"""Check paired inference and reproduction of the published tables."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import t

from scripts.confidence_intervals import accuracy_gain, fieller_reduction


ROOT = Path(__file__).resolve().parents[1]


def test_loss_interval_inverts_paired_t_and_preserves_covariance():
    uniform = np.array([1.7, 2.0, 1.9, 2.2, 2.1])
    frontloaded = np.array([1.3, 1.8, 1.7, 1.9, 1.6])
    result = fieller_reduction(uniform, frontloaded)
    for bound in result["ci95"]:
        difference = frontloaded - (1 - bound / 100) * uniform
        statistic = difference.mean() / (difference.std(ddof=1) / np.sqrt(5))
        assert abs(statistic) == pytest.approx(t.ppf(0.975, 4), abs=1e-10)
    shuffled = fieller_reduction(uniform, frontloaded[::-1])
    assert shuffled["estimate"] == pytest.approx(result["estimate"])
    assert not np.allclose(shuffled["ci95"], result["ci95"])


def test_uncertain_denominator_is_not_reported_as_a_finite_interval():
    result = fieller_reduction([0.01, 0.01, 10], [1, 1.01, 0.99])
    assert result["kind"] in ("unbounded", "disjoint")
    assert result["ci95"] is None
    json.dumps(result, allow_nan=False)


def test_accuracy_interval_uses_percentage_point_differences():
    # Paired gains are [1, 2, 3] percentage points, with sample SD equal to 1.
    result = accuracy_gain([50, 70, 90], [51, 72, 93])
    radius = t.ppf(0.975, 2) / np.sqrt(3)
    assert result["estimate"] == 2
    assert result["ci95"] == pytest.approx([2 - radius, 2 + radius])


@pytest.mark.parametrize(
    "uniform,frontloaded",
    [([1], [1]), ([1, 2], [1]), ([1, float("nan")], [1, 2]), ([0, 0], [1, 2])],
)
def test_invalid_loss_pairs_are_rejected(uniform, frontloaded):
    with pytest.raises(ValueError):
        fieller_reduction(uniform, frontloaded)


def test_standalone_command_reproduces_checked_in_results_and_tables(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/confidence_intervals.py"),
            "--data", str(ROOT / "results/confidence_seed_metrics.json"),
            "--output-dir", str(tmp_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    for extension in ("json", "csv", "md"):
        name = f"confidence_intervals.{extension}"
        assert (tmp_path / name).read_bytes() == (ROOT / "results" / name).read_bytes()
    for name in ("original_results_table.tex", "frontloaded_table.tex"):
        assert (tmp_path / name).read_bytes() == (
            ROOT / "manuscript/sections" / name
        ).read_bytes()
