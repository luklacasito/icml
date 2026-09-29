"""Recovered CIFAR source must agree with the hashes saved by the experiments."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_archived_training_and_probe_match_recorded_sources():
    block = json.loads((ROOT / "results/sliding_block.json").read_text())
    correlations = json.loads((ROOT / "results/cifar_correlations.json").read_text())
    expected = {
        "sliding-block": next(iter(block["provenance"]["training_sources"].values())),
        "dropout-rates": correlations["source"]["files"],
    }
    source = ROOT / "benchmarks/sources/cifar-depth"
    for directory, hashes in expected.items():
        for name, digest in hashes.items():
            content = (source / directory / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == digest
    probe = (source / "dropout-rates/probe_rates.py").read_bytes()
    assert hashlib.sha256(probe).hexdigest() == correlations["provenance"]["probe_code_sha256"]
    config = json.loads((source / "dropout-rates/sweep.json").read_text())
    assert config == correlations["config"]
