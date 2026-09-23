"""Standalone FI-2010 vanilla MLP: recovered model, data, and trial recipe.

Numerical definitions retain the original study's execution and RNG ordering.
This module is independent of the four-dataset benchmark models and caches.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

ARCHITECTURES = ((6, 256), (6, 512), (12, 256), (12, 512))
LEARNING_RATES = (3e-5, 1e-4, 3e-4, 1e-3)
PROFILES = (
    "none",
    "uniform",
    "step_early",
    "step_late",
    "big_step",
    "linear_early",
    "linear_late",
)


def profile_layers(profile: str, depth: int, mean_dropout: float = 0.1) -> list[float]:
    if depth < 1 or not 0 <= mean_dropout < 1:
        raise ValueError("Invalid depth or dropout mean")
    if profile == "none":
        values = [0.0] * depth
    elif profile == "uniform":
        values = [mean_dropout] * depth
    elif profile in {"step_early", "step_late"}:
        active = [2 * mean_dropout] * (depth // 2)
        values = active + [0.0] * (depth - len(active))
        if profile == "step_late":
            values.reverse()
    elif profile == "big_step":
        count = math.ceil(depth / 3)
        active = depth * mean_dropout / count
        values = [active] * count + [0.0] * (depth - count)
    elif profile in {"linear_early", "linear_late"}:
        values = np.linspace(2 * mean_dropout, 0.0, depth).tolist()
        if profile == "linear_late":
            values.reverse()
    else:
        raise ValueError(f"Unknown profile: {profile}")
    if any(not 0 <= value < 1 for value in values):
        raise ValueError("Dropout probabilities must lie in [0, 1)")
    if profile != "none" and not np.isclose(np.mean(values), mean_dropout):
        raise ValueError("Profile does not preserve the declared mean")
    return list(map(float, values))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def derived_seed(stream, value):
    return int.from_bytes(
        hashlib.sha256(f"{stream}\0{canonical(value)}".encode()).digest()[:4], "big"
    )


def pair_family(profile):
    for suffix in ("_early", "_late"):
        if profile.endswith(suffix):
            return profile[: -len(suffix)]
    return profile


def trial(stage, depth, width, profile, learning_rate, seed, epochs, data_hash, source_hash):
    cell = {"depth": depth, "width": width, "seed": seed}
    return {
        "schema": "fi2010_vanilla_mlp_v1",
        "stage": stage,
        "architecture": "plain_feedforward_relu_mlp",
        "depth": depth,
        "width": width,
        "input_shape": [128, 40],
        "output_dim": 3,
        "profile": profile,
        "mean_dropout": 0.0 if profile == "none" else 0.1,
        "dropout_probabilities": profile_layers(profile, depth),
        "seed": seed,
        "randomization": {
            "scheme": "sha256_named_streams_v1",
            "initialization_seed": derived_seed("initialization", cell),
            "minibatch_seed": derived_seed("minibatch", cell),
            "dropout_seed": derived_seed(
                "dropout_common_random_numbers",
                {**cell, "profile_family": pair_family(profile)},
            ),
        },
        "initialization": {
            "activation": "relu",
            "sigma_w_sq": 1.98,
            "sigma_b_sq": 0.02,
        },
        "training": {
            "epochs": epochs,
            "batch_size": 128,
            "optimizer": "adam",
            "learning_rate": learning_rate,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "clip_norm": 1.0,
            "cosine_floor": 0.01,
        },
        "data": {
            "train_days": [6],
            "validation_days": [7],
            "test_days": [8, 9, 10] if stage == "confirmation" else [],
            "sequence": 128,
            "features": 40,
        },
        "data_sha256": data_hash,
        "source_sha256": source_hash,
    }


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Windows(Dataset):
    def __init__(self, segments, mean, std, sequence=128, tail=10):
        self.segments = segments
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.sequence, self.tail = sequence, tail
        self.counts = [len(segment["x"]) - tail - sequence + 1 for segment in segments]
        if not self.counts or min(self.counts) < 1:
            raise ValueError("Empty or too-short segment")
        self.ends = np.cumsum(self.counts)

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        segment = int(np.searchsorted(self.ends, index, side="right"))
        previous = self.ends[segment - 1] if segment else 0
        return segment, int(index - previous)

    def __getitem__(self, index):
        segment_index, start = self.locate(index)
        segment = self.segments[segment_index]
        endpoint = start + self.sequence - 1
        inputs = (segment["x"][start : endpoint + 1] - self.mean) / self.std
        return inputs, segment["y"][endpoint, 0], int(index)

    def example_ids(self, indexes):
        rows = []
        for index in indexes:
            segment_index, start = self.locate(int(index))
            segment = self.segments[segment_index]
            rows.append((segment["day"], segment["stock"], start, start + self.sequence - 1))
        return np.asarray(rows, dtype=np.int64)


def load_data(root, include_test=False, sequence=128):
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "fi2010_segments_v1":
        raise ValueError("Unrecognized dataset manifest")
    splits = {"train": [6], "validation": [7]}
    if include_test:
        splits["test"] = [8, 9, 10]
    segments = {name: [] for name in splits}
    for item in manifest["segments"]:
        for split, days in splits.items():
            if item["day"] not in days:
                continue
            path = root / item["file"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"Data checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                segments[split].append(
                    {
                        **item,
                        "x": torch.from_numpy(archive["x"].copy()),
                        "y": torch.from_numpy(archive["y"].copy()),
                    }
                )
    tail = manifest["tail_purge_representations"]
    training_rows = np.concatenate([segment["x"].numpy()[:-tail] for segment in segments["train"]])
    mean = training_rows.mean(axis=0, dtype=np.float64)
    std = training_rows.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    datasets = {name: Windows(value, mean, std, sequence, tail) for name, value in segments.items()}
    labels = np.concatenate(
        [segment["y"].numpy()[sequence - 1 : -tail, 0] for segment in segments["train"]]
    )
    counts = np.bincount(labels, minlength=3)
    prior = (counts + 1) / (counts.sum() + 3)
    metadata = {
        "manifest_sha256": digest(manifest_path),
        "train_days": [6],
        "validation_days": [7],
        "test_days": [8, 9, 10] if include_test else [],
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "counts": {name: len(value) for name, value in datasets.items()},
        "train_class_counts": counts.tolist(),
        "train_prior": prior.tolist(),
        "train_prior_logloss": float(-np.log(prior[labels]).mean()),
    }
    metadata["preprocessing_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    return datasets, metadata


class StageDropout(nn.Module):
    def __init__(self, probability: float, seed: int):
        super().__init__()
        if not math.isfinite(probability) or not 0 <= probability < 1:
            raise ValueError("Dropout probability must be finite and in [0, 1)")
        self.probability = float(probability)
        self.seed = int(seed)
        self.generator = None
        self.pending_state = None

    def forward(self, inputs):
        if not self.training or self.probability == 0:
            return inputs
        if self.generator is None:
            self.generator = torch.Generator(device=inputs.device).manual_seed(self.seed)
            if self.pending_state is not None:
                self.generator.set_state(self.pending_state)
                self.pending_state = None
        keep = (
            torch.rand(inputs.shape, device=inputs.device, generator=self.generator)
            >= self.probability
        )
        return inputs * keep.to(inputs.dtype) / (1 - self.probability)

    def rng_state(self):
        return self.generator.get_state() if self.generator is not None else self.pending_state

    def restore_rng(self, state):
        self.generator, self.pending_state = None, state


class VanillaMLP(nn.Module):
    """Depth counts hidden affine/ReLU/dropout blocks; readout is separate."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        probabilities: list[float],
        dropout_seed: int,
        output_dim: int = 3,
        sigma_w_sq: float = 1.98,
        sigma_b_sq: float = 0.02,
    ):
        super().__init__()
        if len(probabilities) != depth:
            raise ValueError("One dropout probability is required per hidden layer")
        self.hidden = nn.ModuleList(
            [nn.Linear(input_dim, width)] + [nn.Linear(width, width) for _ in range(depth - 1)]
        )
        self.dropouts = nn.ModuleList(
            StageDropout(value, dropout_seed + index) for index, value in enumerate(probabilities)
        )
        self.activation = nn.ReLU()
        self.readout = nn.Linear(width, output_dim)
        for layer in [*self.hidden, self.readout]:
            nn.init.normal_(
                layer.weight,
                mean=0.0,
                std=math.sqrt(sigma_w_sq / layer.weight.shape[1]),
            )
            nn.init.normal_(layer.bias, mean=0.0, std=math.sqrt(sigma_b_sq))

    def forward(self, inputs):
        hidden = inputs.reshape(inputs.shape[0], -1)
        for layer, dropout in zip(self.hidden, self.dropouts, strict=True):
            hidden = dropout(self.activation(layer(hidden)))
        return self.readout(hidden)


def read_release(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".txt")]
        assert len(names) == 1, names
        rows = []
        with archive.open(names[0]) as f:
            for i, line in enumerate(f):
                if i < 40 or 144 <= i < 149:
                    rows.append(np.fromstring(line.decode("ascii"), sep=" ", dtype=np.float64))
        assert i == 148 and len(rows) == 45
        data = np.stack(rows).T
    assert np.isfinite(data).all()
    assert np.isin(data[:, 40:], [1, 2, 3]).all()
    return data, names[0]


def stock_cuts(data):
    jumps = np.abs(np.diff(data[:, 0]))
    ranked = np.argsort(jumps)[::-1]
    # Fail instead of treating ambiguous movement as a stock boundary.
    margin = float(jumps[ranked[3]] / max(jumps[ranked[4]], 1e-15))
    assert margin >= 10, f"Ambiguous stock discontinuities: margin={margin}"
    cuts = [0] + sorted((ranked[:4] + 1).tolist()) + [len(data)]
    assert min(np.diff(cuts)) > 256
    return cuts, margin


def prepare_segments(raw, output):
    output.mkdir(parents=True, exist_ok=True)
    days, records = {}, []
    for day in range(1, 11):
        path = raw / f"day{day:02d}.zip"
        data, member = read_release(path)
        expected = (
            "Train_Dst_NoAuction_DecPre_CF_1.txt"
            if day == 1
            else f"Test_Dst_NoAuction_DecPre_CF_{day - 1}.txt"
        )
        assert member == expected, (member, expected)
        cuts, margin = stock_cuts(data)
        blocks = [data[a:b] for a, b in zip(cuts, cuts[1:])]
        days[day] = blocks
        records.append(
            {
                "day": day,
                "source_file": path.name,
                "member": member,
                "sha256": digest(path),
                "rows": len(data),
                "stock_cuts": cuts,
                "jump_margin": margin,
                "ask_price_ranges": [[float(b[:, 0].min()), float(b[:, 0].max())] for b in blocks],
            }
        )
    cumulative_path = raw / "cumulative06.zip"
    cumulative, member = read_release(cumulative_path)
    assert member == "Train_Dst_NoAuction_DecPre_CF_6.txt"
    # Daily cuts determine expected cumulative offsets. Exact row equality
    # below independently verifies them, including gaps between trading days.
    cuts = [0] + np.cumsum([sum(len(days[d][s]) for d in range(1, 7)) for s in range(5)]).tolist()
    assert cuts[-1] == len(cumulative)
    comparisons = []
    for stock, (a, b) in enumerate(zip(cuts, cuts[1:])):
        expected = np.concatenate([days[day][stock] for day in range(1, 7)])
        observed = cumulative[a:b]
        assert expected.shape == observed.shape, (stock, expected.shape, observed.shape)
        assert np.array_equal(expected, observed), f"Cumulative/day mismatch stock {stock + 1}"
        comparisons.append({"stock_ordinal": stock + 1, "rows": len(expected), "exact_equal": True})
    # Test/validation days use the same ordinal order. Require prices to remain
    # compatible with each stock's training-period range, even if two ranges overlap.
    for stock in range(5):
        reference = np.median(np.concatenate([days[d][stock][:, 0] for d in range(1, 7)]))
        for day in range(7, 11):
            ratio = float(np.median(days[day][stock][:, 0]) / reference)
            assert 0.8 < ratio < 1.2, (day, stock, ratio)
    segments = []
    for day, blocks in days.items():
        for stock, block in enumerate(blocks, 1):
            filename = f"day{day:02d}_stock{stock}.npz"
            target = output / filename
            np.savez_compressed(
                target,
                x=block[:, :40].astype(np.float32),
                y=block[:, 40:].astype(np.int64) - 1,
            )
            segments.append(
                {
                    "file": filename,
                    "sha256": digest(target),
                    "day": day,
                    "stock": stock,
                    "rows": len(block),
                }
            )
    manifest = {
        "schema": "fi2010_segments_v1",
        "dataset": "FI-2010",
        "normalization": "release decimal precision; train-only z-score in runner",
        "source": "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
        "mirror": "https://www.kaggle.com/datasets/ulfricirons/fi-2010",
        "license": "CC-BY-4.0",
        "attribution": "Ntakaris et al., Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data, 2017/2018",
        "stock_identity": "ordinal 1..5; no unverified ticker mapping",
        "boundary_method": "four isolated ask-price jumps; stock-major cumulative CF6 equality on 40 features and all 5 labels; day7-10 ordinal price continuity",
        "cumulative_audit": {
            "file": cumulative_path.name,
            "sha256": digest(cumulative_path),
            "comparisons": comparisons,
        },
        "class_mapping": {"0": "up", "1": "stationary", "2": "down"},
        "label_row_zero_based": 144,
        "horizon_representations": 1,
        "upstream_horizon_name": 10,
        "representation": "10 consecutive order events",
        "tail_purge_representations": 10,
        "sources": records,
        "segments": segments,
        "split_days": {
            "small": [6],
            "medium": [4, 5, 6],
            "large": [1, 2, 3, 4, 5, 6],
            "validation": [7],
            "test": [8, 9, 10],
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "audited",
                "segments": len(segments),
                "rows": sum(s["rows"] for s in segments),
                "manifest": str(output / "manifest.json"),
            }
        )
    )
