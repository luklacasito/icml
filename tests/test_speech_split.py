"""The official Speech Commands partitions must separate speakers, not clips."""

import numpy as np
import pytest

from benchmarks.speech_split import audit_speech_split, official_speech_split


def sample_paths():
    paths = [
        f"{word}/{speaker}_nohash_{utterance}.wav"
        for word in ("yes", "no")
        for speaker in ("train_a", "train_b", "val_a", "test_a")
        for utterance in range(2)
    ]
    validation = [p for p in paths if "/val_a_" in p]
    testing = [p for p in paths if "/test_a_" in p]
    return paths, validation, testing


def test_official_groups_and_full_evaluation_sets():
    paths, validation, testing = sample_paths()
    indices, manifest = official_speech_split(paths, validation, testing, train_size=4)
    assert {paths[i] for i in indices["validation"]} == set(validation)
    assert {paths[i] for i in indices["test"]} == set(testing)
    assert manifest["partitions"]["train"]["class_counts"] == {"no": 2, "yes": 2}
    assert all(item["speakers"] == item["clips"] == 0 for item in manifest["overlaps"].values())
    assert manifest["full_official_partitions"]["partitions"]["train"]["clips"] == 8
    assert len(manifest["official_validation_list_sha256"]) == 64


def test_same_selection_despite_cache_row_order():
    paths, validation, testing = sample_paths()
    first, _ = official_speech_split(paths, validation, testing, train_size=4, seed=9)
    reverse = list(reversed(paths))
    second, _ = official_speech_split(reverse, validation, testing, train_size=4, seed=9)
    for name in first:
        assert {paths[i] for i in first[name]} == {reverse[i] for i in second[name]}


def test_existing_training_sample_can_be_preserved():
    paths, validation, testing = sample_paths()
    chosen = [p for p in paths if "/train_a_" in p]
    indices, manifest = official_speech_split(
        paths, validation, testing, train_size=4, training_files=chosen
    )
    assert {paths[i] for i in indices["train"]} == set(chosen)
    assert manifest["training_selection"] == "supplied_filenames"
    with pytest.raises(ValueError, match="official training"):
        official_speech_split(paths, validation, testing, train_size=4, training_files=validation)


def test_rejects_speaker_leak_even_when_no_clip_is_repeated():
    paths, validation, testing = sample_paths()
    validation = [*validation, "yes/train_a_nohash_0.wav"]
    with pytest.raises(ValueError, match="split a speaker"):
        official_speech_split(paths, validation, testing, train_size=4)


def test_old_training_only_cache_is_rejected():
    paths, validation, testing = sample_paths()
    paths = [p for p in paths if p not in validation + testing]
    with pytest.raises(ValueError, match="missing"):
        official_speech_split(paths, validation, testing, train_size=4)


def test_audit_counts_clip_exposure_not_only_unique_speakers():
    paths = ["yes/a_nohash_0.wav", "yes/a_nohash_1.wav", "no/b_nohash_0.wav", "no/a_nohash_2.wav"]
    result = audit_speech_split(
        paths, {"train": np.array([0]), "validation": np.array([1, 2]), "test": np.array([3])}
    )
    assert result["overlaps"]["train_validation"] == {
        "clips": 0,
        "speakers": 1,
        "second_partition_clips_from_first_partition_speakers": 1,
    }


@pytest.mark.parametrize(
    "bad_paths", [["yes/a_nohash_0.wav"] * 2, ["../a_nohash_0.wav"], ["yes/no-speaker.wav"]]
)
def test_rejects_invalid_or_duplicate_metadata(bad_paths):
    with pytest.raises(ValueError):
        audit_speech_split(bad_paths, {})


def make_cache(tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    from benchmarks import data
    from benchmarks.prepare import _save
    from benchmarks.speech_split import SPEECH_PREPROCESSING

    paths, validation, testing = sample_paths()
    chosen = [p for p in paths if "/train_a_" in p]
    parts, manifest = official_speech_split(
        paths, validation, testing, train_size=4, training_files=chosen
    )
    manifest.update(preprocessing=SPEECH_PREPROCESSING, split_seed=20260812)
    partition = np.array(
        [1 if p in validation else 2 if p in testing else 0 for p in paths], dtype=np.int8
    )
    metadata = {
        "filenames": np.asarray(paths),
        "speaker_ids": np.asarray([p.split("/")[1].split("_nohash_")[0] for p in paths]),
        "official_partition": partition,
        "split_manifest": json.dumps(manifest, sort_keys=True),
        **{f"{key}_indices": value for key, value in parts.items()},
    }
    monkeypatch.setitem(
        data.BENCHMARK_SPECS,
        "speech_commands_speakers",
        replace(
            data.BENCHMARK_SPECS["speech_commands_speakers"],
            classes=2,
            mlp_input_dim=4,
            image_size=2,
            patch_size=1,
            sequence_length=4,
            train_size=4,
            validation_size=4,
            test_size=4,
        ),
    )
    features = np.arange(len(paths) * 4, dtype=np.float32).reshape(len(paths), 4)
    labels = np.array([0 if p.startswith("no/") else 1 for p in paths], dtype=np.int64)
    _save(
        tmp_path,
        "speech_commands_speakers",
        features,
        features.reshape(-1, 1, 2, 2),
        labels,
        metadata=metadata,
    )
    return metadata


def test_loader_uses_saved_official_groups_for_both_models(tmp_path, monkeypatch):
    from benchmarks.data import load_benchmark_bundle
    from benchmarks.speech_split import SPEECH_SPLIT_PROTOCOL

    make_cache(tmp_path, monkeypatch)
    first = load_benchmark_bundle("speech_commands_speakers", root=tmp_path, view="mlp")
    second = load_benchmark_bundle("speech_commands_speakers", root=tmp_path, view="sequence")
    assert first.split_protocol == SPEECH_SPLIT_PROTOCOL
    assert first.split_hash == second.split_hash
    assert len(first.train) == len(first.validation) == len(first.test) == 4
    assert first.train.tensors[0].shape == (4, 4)
    assert second.train.tensors[0].shape == (4, 1, 2, 2)
    np.testing.assert_allclose(first.train.tensors[0].mean(0), 0, atol=1e-7)
    with pytest.raises(ValueError, match="sizes or seed"):
        load_benchmark_bundle("speech_commands_speakers", root=tmp_path, split_seed=7)


def test_cached_ids_and_manifest_cannot_disagree(tmp_path, monkeypatch):
    from benchmarks.data import cache_path
    from benchmarks.speech_split import cached_speech_split

    make_cache(tmp_path, monkeypatch)
    with np.load(cache_path("speech_commands_speakers", tmp_path), allow_pickle=False) as cache:
        values = {key: cache[key] for key in cache.files}
    changed = {**values, "speaker_ids": np.full(len(values["speaker_ids"]), "fake")}
    with pytest.raises(ValueError, match="speaker IDs"):
        cached_speech_split(changed)
    with pytest.raises(ValueError, match="labels"):
        cached_speech_split({**values, "labels": 1 - values["labels"]})
    with pytest.raises(ValueError, match="validation indices"):
        cached_speech_split({**values, "validation_indices": values["validation_indices"][:2]})


def test_runner_writes_separate_speaker_experiment(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import torch
    from benchmarks.run import config_hash, read_protocols, run

    torch.set_num_threads(1)
    make_cache(tmp_path, monkeypatch)
    cohort_id = "speech_commands-mlp-zero-decay"
    cohort = read_protocols()[cohort_id]
    arm = cohort["arms"]["uniform"]
    arm["spec"].update(depth=2, width=8, epochs=1, batch_size=4)
    arm["p_layers"] = [0.1, 0.1]
    arm["historical_config_hash"] = config_hash(arm["spec"])
    cohort["arms"] = {"uniform": arm}
    protocol_path = tmp_path / "protocols.json"
    protocol_path.write_text(json.dumps({"schema_version": 1, "protocols": {cohort_id: cohort}}))
    run(
        SimpleNamespace(
            protocols=protocol_path,
            cohort=cohort_id,
            profile=["uniform"],
            seed=[200],
            dataset="speech_commands_speakers",
            describe=False,
            data_root=tmp_path,
            output_dir=tmp_path / "runs",
            device="cpu",
            cache_sha256=None,
        )
    )
    destination = (
        tmp_path
        / "runs"
        / "speech_commands_speakers-mlp-zero-decay-transferred-settings"
        / "uniform"
        / "seed-200"
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["spec"]["dataset"] == "speech_commands_speakers"
    assert manifest["provenance"]["settings_transferred_to_new_split"] is True
    assert manifest["provenance"]["historical_seed"] is False
    assert (destination / "best.pt").exists() and (destination / "final.pt").exists()
    assert json.loads((destination / "result.json").read_text())["status"] == "complete"


def test_new_db_cutoff_does_not_depend_on_neighboring_clips():
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    from benchmarks.prepare import _speech_db

    power = torch.tensor([[[1.0, 1e-12]], [[1.0, 1e-12]]])
    louder = power.clone()
    louder[1] *= 100
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    first = _speech_db(power, to_db, per_clip=True)
    second = _speech_db(louder, to_db, per_clip=True)
    assert torch.equal(first[0], second[0])
    assert not torch.equal(
        _speech_db(power, to_db, per_clip=False)[0],
        _speech_db(louder, to_db, per_clip=False)[0],
    )
