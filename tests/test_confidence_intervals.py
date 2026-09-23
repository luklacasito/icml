"""Check paired inference and reproduction of the published tables."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import t

from scripts.confidence_intervals import (
    accuracy_gain,
    fieller_reduction,
    render_frontloaded_table,
    summarize,
    write_outputs,
)


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


def test_disjoint_fieller_bounds_still_invert_the_paired_t_statistic():
    uniform = np.array([0.01, 0.01, 10])
    frontloaded = np.array([1, 1.01, 0.99])
    result = fieller_reduction(uniform, frontloaded)
    assert result["kind"] == "disjoint"
    for interval in result["intervals"]:
        bound = next(value for value in interval if value is not None)
        difference = frontloaded - (1 - bound / 100) * uniform
        statistic = difference.mean() / (difference.std(ddof=1) / np.sqrt(3))
        assert abs(statistic) == pytest.approx(t.ppf(0.975, 2))


@pytest.mark.parametrize("seeds,values", [([0, 0], [1, 2]), ([0, 1], [1])])
def test_evidence_rejects_duplicate_or_incomplete_seed_pairs(seeds, values):
    evidence = {
        "inference": "test",
        "original": [
            {
                "n": 2,
                "seeds": seeds,
                "metrics": {"final_loss": {"uniform": [1, 2], "frontloaded": values}},
            }
        ],
        "benchmarks": [],
    }
    with pytest.raises(ValueError, match="[Ss]eed"):
        summarize(evidence)


def test_accuracy_interval_uses_percentage_point_differences():
    # Paired gains are [1, 2, 3] percentage points, with sample SD equal to 1.
    result = accuracy_gain([50, 70, 90], [51, 72, 93])
    radius = t.ppf(0.975, 2) / np.sqrt(3)
    assert result["estimate"] == 2
    assert result["ci95"] == pytest.approx([2 - radius, 2 + radius])


@pytest.fixture
def partial_endpoint_evidence():
    """Historical checkpoint values exist; only the fresh seeds have final values."""
    seeds = list(range(100, 110))
    row = {
        "dataset": "Jannis",
        "model": "Transformer",
        "train_size": 3840,
        "weight_decay": 0,
        "profile": "Big step",
        "n": 10,
        "seeds": seeds,
        "metrics": {
            "checkpoint_loss": {
                "uniform": [1 + 0.1 * i for i in range(10)],
                "frontloaded": [0.9 + 0.08 * i for i in range(10)],
            },
            "checkpoint_accuracy_percent": {
                "uniform": [70 + 0.1 * i for i in range(10)],
                "frontloaded": [71 + 0.1 * i for i in range(10)],
            },
            "final_loss": {
                "seeds": seeds[5:],
                "uniform": [1.4 + 0.1 * i for i in range(5)],
                "frontloaded": [1.2 + 0.09 * i for i in range(5)],
            },
            "final_accuracy_percent": {
                "seeds": seeds[5:],
                "uniform": [68 + 0.1 * i for i in range(5)],
                "frontloaded": [70 + 0.2 * i for i in range(5)],
            },
        },
    }
    return {
        "schema_version": 1,
        "inference": "Fixed profiles and split; paired seed variation.",
        "endpoint_definitions": {},
        "original": [],
        "benchmarks": [row],
    }


def test_endpoint_subset_uses_its_own_pairs_and_degrees_of_freedom(
    partial_endpoint_evidence,
):
    row = summarize(partial_endpoint_evidence)["benchmarks"][0]
    checkpoint = row["metrics"]["checkpoint_loss"]
    final = row["metrics"]["final_loss"]
    assert (checkpoint["n"], checkpoint["df"]) == (10, 9)
    assert checkpoint["seeds"] == row["seeds"]
    assert (final["n"], final["df"]) == (5, 4)
    assert final["seeds"] == [105, 106, 107, 108, 109]
    assert final["uniform_mean"] == pytest.approx(1.6)
    assert final["frontloaded_mean"] == pytest.approx(1.38)
    assert final["estimate"] == pytest.approx(13.75)


@pytest.mark.parametrize(
    "endpoint_seeds",
    [
        [105, 105, 107, 108, 109],  # Duplicate identifiers cannot identify pairs.
        [105, 106, 107, 108, 999],  # Every pair must belong to the row's cohort.
        [105, 106],  # Five outcomes cannot describe two identified pairs.
    ],
)
def test_invalid_endpoint_seed_subsets_are_rejected(partial_endpoint_evidence, endpoint_seeds):
    partial_endpoint_evidence["benchmarks"][0]["metrics"]["final_loss"]["seeds"] = endpoint_seeds
    with pytest.raises(ValueError, match="[Ss]eed"):
        summarize(partial_endpoint_evidence)


def test_exports_use_endpoint_counts_and_identify_extended_jannis_by_recipe(
    tmp_path, partial_endpoint_evidence
):
    extended = copy.deepcopy(partial_endpoint_evidence["benchmarks"][0])
    extended["weight_decay"] = 1e-7
    extended["metrics"]["final_loss"] = None
    extended["metrics"]["final_accuracy_percent"] = None
    partial_endpoint_evidence["benchmarks"].append(extended)
    write_outputs(tmp_path, partial_endpoint_evidence)

    with (tmp_path / "confidence_intervals.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert [record["n"] for record in records] == [
        "10",
        "10",
        "5",
        "5",
        "10",
        "10",
        "",
        "",
    ]

    markdown = (tmp_path / "confidence_intervals.md").read_text()
    checkpoints, final = markdown.split("## Recorded final-epoch test results")
    assert "| Jannis / Transformer (N=3,840) | 10 |" in checkpoints
    assert "| Jannis extended / Transformer (N=3,840) | 10 |" in checkpoints
    assert "| Jannis / Transformer (N=3,840) | 5 |" in final
    assert "| Jannis extended /" not in final

    latex = render_frontloaded_table(summarize(partial_endpoint_evidence)["benchmarks"])
    assert "Jannis (3,840)" in latex
    assert latex.count("Jannis extended (3,840)") == 1
    assert " & 10/5 & " in latex
    assert " & 10/-- & " in latex


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
            "--data",
            str(ROOT / "results/confidence_seed_metrics.json"),
            "--output-dir",
            str(tmp_path),
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
        assert (tmp_path / name).read_bytes() == (ROOT / "manuscript/sections" / name).read_bytes()
