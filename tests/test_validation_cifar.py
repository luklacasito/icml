"""Prevent accidental re-pairing or relabelling of the CIFAR followups."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.validation_cifar_results import comparisons

ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_pairs_cannot_be_reordered_silently():
    evidence = json.loads((ROOT / "results/cifar_validation_evidence.json").read_text())
    changed = copy.deepcopy(evidence)
    changed["studies"][0]["recipes"]["tuned_frontloaded"]["seeds"].reverse()
    with pytest.raises(ValueError, match="matching ordered"):
        comparisons(changed)


def test_validation_cifar_export_reproduces_archived_followups(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validation_cifar_results.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    for name in ("cifar_validation_intervals.json", "cifar_validation.md"):
        assert (tmp_path / name).read_bytes() == (ROOT / "results" / name).read_bytes()
    rows = json.loads((tmp_path / "cifar_validation_intervals.json").read_text())["comparisons"]
    assert [row["n"] for row in rows] == [10, 10, 3, 3]
    for row in rows:
        assert set(row["metrics"]) == {
            "checkpoint_loss",
            "checkpoint_accuracy_percent",
            "final_loss",
            "final_accuracy_percent",
        }
