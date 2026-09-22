"""Regression checks for the public frozen benchmark execution protocol."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from benchmarks import data
from benchmarks.models import build_model
from benchmarks.prepare import _digest
from benchmarks.run import (
    PROTOCOLS, config_hash, fit, read_protocols, run, seed_streams,
    smoke, synthetic_fixture, validate_arm, verify_cache_payload,
)


def test_all_frozen_arms_match_historical_config_hashes_and_budgets():
    protocols = read_protocols()
    assert {c["dataset"] for c in protocols.values()} == {
        "fi2010", "openml_jannis", "speech_commands", "tiny_imagenet",
    }
    for cohort in protocols.values():
        for arm in cohort["arms"].values():
            validate_arm(cohort, arm)
    assert protocols["tiny_imagenet-mlp-20k"]["split"]["standardization"] == "float32_v15"
    assert protocols["tiny_imagenet-mlp-80k"]["split"]["standardization"] == "chunked_float64_v17"


def test_historical_rng_fixture_and_numeric_type_sensitivity():
    # Independently verified against the recovered historical seed_streams.
    cohort = read_protocols()["openml_jannis-mlp-zero-decay"]
    arm = cohort["arms"]["uniform"]
    assert seed_streams(arm["spec"]) == {
        "scheme": "sha256_named_streams_v1", "base_seed": 100,
        "initialization_seed": 2034274571, "minibatch_seed": 3044408285,
        "dropout_seed": 98623406, "dropout_crn_group": "b173e876aefef39b75ed",
    }
    edited = deepcopy(arm)
    edited["spec"]["mlp_ratio"] = int(edited["spec"]["mlp_ratio"])
    assert config_hash(edited["spec"]) != config_hash(arm["spec"])
    assert seed_streams(edited["spec"])["dropout_seed"] != seed_streams(arm["spec"])["dropout_seed"]
    with pytest.raises(ValueError, match="configuration hash"):
        validate_arm(cohort, edited)


def test_model_forward_for_each_retained_input_family():
    spec, _ = synthetic_fixture()
    torch.set_num_threads(1)
    for dataset, shape in (
        ("fi2010", (2, 100, 40)), ("openml_jannis", (2, 54, 1)),
        ("tiny_imagenet", (2, 3, 64, 64)), ("speech_commands", (2, 1, 64, 64)),
    ):
        for kind in ("mlp", "transformer"):
            small = {**spec, "dataset": dataset, "model_kind": kind,
                     "depth": 2, "width": 8, "heads": 2}
            model = build_model(small, [0.2, 0.0])
            output = model(torch.zeros(shape))
            assert output.shape == (2, data.BENCHMARK_SPECS[dataset].classes)
            assert torch.isfinite(output).all()
            output.sum().backward()
            assert all(p.grad is not None for p in model.parameters())


def test_checkpoint_smoke_reloads_distinct_best_and_final(tmp_path):
    directory = tmp_path / "smoke"
    report = smoke(directory)
    assert report["saved_endpoints_reproduced"] is True
    result = json.loads((directory / "result.json").read_text())
    assert result["selected_epoch"] == np.argmin(result["history"]["validation_loss"]) == 0
    assert result["test"]["fixed_final_epoch"]["epoch"] == 3
    assert result["test"]["best_validation_checkpoint"]["loss"] < result["test"]["fixed_final_epoch"]["loss"]
    assert result["history"]["lr_multiplier"][0] == 1.0
    assert result["history"]["lr_multiplier"][-1] == pytest.approx(0.001)
    with pytest.raises(FileExistsError):
        smoke(directory)


def test_test_labels_cannot_change_training_or_checkpoint_selection(tmp_path):
    spec, bundle = synthetic_fixture()
    first = fit(spec, [0.1, 0.1], bundle, tmp_path / "first", device="cpu", provenance={})
    modified = replace(bundle, test=TensorDataset(bundle.test.tensors[0], torch.zeros(12, dtype=torch.long)))
    second = fit(spec, [0.1, 0.1], modified, tmp_path / "second", device="cpu", provenance={})
    assert first["history"] == second["history"]
    assert first["selected_epoch"] == second["selected_epoch"]
    assert first["test"] != second["test"]
    for name in ("best.pt", "final.pt"):
        a = torch.load(tmp_path / "first" / name, weights_only=True)["model_state_dict"]
        b = torch.load(tmp_path / "second" / name, weights_only=True)["model_state_dict"]
        assert all(torch.equal(a[key], b[key]) for key in a)


def test_splits_are_disjoint_balanced_and_temporally_embargoed():
    labels = np.repeat(np.arange(4), 20)
    parts = data._balanced_split(labels, train_size=24, validation_size=8, test_size=12, seed=20260812)
    assert len(np.unique(np.concatenate(parts))) == 44
    for indices, expected in zip(parts, (6, 2, 3)):
        np.testing.assert_array_equal(np.bincount(labels[indices]), [expected] * 4)
    train, validation, test = data._anchored_split(70_200, train_size=40_000,
        validation_size=10_000, test_size=20_000, embargo=100)
    assert (train[-1], validation[0], validation[-1], test[0], test[-1]) == (
        39_999, 40_100, 50_099, 50_200, 70_199,
    )


def test_standardization_is_train_only_and_tiny_modes_remain_distinct():
    original = np.asarray([[1., 3.], [3., 7.], [100., 200.]], dtype=np.float32)
    changed = original.copy()
    changed[-1] *= 100
    for normalize in (data._standardize, data._standardize_inplace_v15, data._standardize_inplace):
        a, stats_a = normalize(original.copy(), np.asarray([0, 1]))
        b, stats_b = normalize(changed.copy(), np.asarray([0, 1]))
        np.testing.assert_array_equal(a[:2], b[:2])
        assert stats_a == stats_b
        np.testing.assert_allclose(a[:2].mean(axis=0), 0)
        np.testing.assert_allclose(a[:2].std(axis=0), 1)
    # Float32 cancellation exposes the historical precision difference.
    large = np.tile(np.asarray([1e8, 1e8 + 8], dtype=np.float32), 4096).reshape(-1, 1)
    indices = np.arange(len(large))
    old, old_stats = data._standardize_inplace_v15(large.copy(), indices)
    new, new_stats = data._standardize_inplace(large.copy(), indices)
    assert old.dtype == new.dtype == np.float32
    assert old_stats != new_stats
    assert not np.array_equal(old, new)


def make_cache(path: Path):
    mlp = np.arange(80, dtype=np.float32).reshape(20, 4)
    sequence = mlp.reshape(20, 4, 1)
    labels = np.arange(20, dtype=np.int64) % 4
    digest = _digest(mlp, sequence, labels)
    np.savez_compressed(path, features_mlp=mlp, features_sequence=sequence,
                        labels=labels, payload_sha256=digest)
    return mlp, sequence, labels, digest


def test_cache_payload_and_streamed_rows_reject_corruption(tmp_path):
    path = tmp_path / "cache.npz"
    mlp, sequence, labels, digest = make_cache(path)
    assert verify_cache_payload(path) == digest
    selected = np.asarray([18, 1, 15, 0])
    np.testing.assert_array_equal(data._load_compressed_npz_rows(path, "features_mlp", selected, chunk_bytes=31), mlp[selected])
    mlp = mlp.copy()
    mlp[0, 0] = -1
    np.savez_compressed(path, features_mlp=mlp, features_sequence=sequence,
                        labels=labels, payload_sha256=digest)
    with pytest.raises(ValueError, match="payload_sha256"):
        verify_cache_payload(path)


def test_frontloaded_alias_requires_explicit_comparison(capsys):
    args = SimpleNamespace(protocols=PROTOCOLS,
        cohort="speech_commands-transformer-zero-decay", profile=["frontloaded"],
        seed=[100], describe=True)
    run(args)
    assert list(json.loads(capsys.readouterr().out)["arms"]) == ["big_step"]
    args.cohort = "fi2010-transformer-linear-followup"
    with pytest.raises(ValueError, match="no selected frontloaded arm"):
        run(args)
