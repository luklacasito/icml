"""Recovered CIFAR source must agree with the hashes saved by the experiments."""

import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def test_archived_training_and_probe_match_recorded_sources():
    block = json.loads((ROOT / "results/sliding_block.json").read_text())
    correlations = json.loads((ROOT / "results/cifar_correlations.json").read_text())
    expected = {
        "sliding-block": next(iter(block["provenance"]["training_sources"].values())),
        "dropout-rates": correlations["source"]["files"],
    }
    with tarfile.open(ROOT / "benchmarks/sources/cifar-depth.tar.gz") as archive:
        for directory, hashes in expected.items():
            for name, digest in hashes.items():
                content = archive.extractfile(f"{directory}/{name}").read()
                assert hashlib.sha256(content).hexdigest() == digest
        probe = archive.extractfile("dropout-rates/probe_rates.py").read()
        assert hashlib.sha256(probe).hexdigest() == correlations["provenance"]["probe_code_sha256"]
        config = json.load(archive.extractfile("dropout-rates/sweep.json"))
        assert config == correlations["config"]
