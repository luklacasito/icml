"""Keep published paired endpoints tied to the new per-fit measurements."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from utils.results import load_npz_result


ROOT = Path(__file__).resolve().parents[1]


def test_extension_archive_matches_every_published_new_endpoint():
    evidence = json.loads((ROOT / "results/confidence_seed_metrics.json").read_text())
    path = ROOT / "results/seed_extension.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["extension"]["artifact_sha256"]
    archive = load_npz_result(path)
    assert len(archive["fits"]) == 114
    seen = set()
    pairs = {}
    for fit in archive["fits"]:
        task = fit["task"]
        assert task["task_id"] not in seen
        seen.add(task["task_id"])
        if task["kind"] == "benchmark":
            section, index = "benchmarks", task["row_index"]
            checkpoint = fit["test"]["best_validation_checkpoint"]
            final = fit["test"]["fixed_final_epoch"]
            assert checkpoint["epoch"] == int(np.argmin(fit["curves"]["validation_loss"]))
            assert final["epoch"] == fit["epochs"] - 1
            endpoints = {
                "checkpoint_loss": checkpoint["loss"],
                "checkpoint_accuracy_percent": 100 * checkpoint["accuracy"],
                "final_loss": final["loss"],
                "final_accuracy_percent": 100 * final["accuracy"],
            }
        else:
            section = "original"
            index = {"relu": 2, "vit": 5}[task["study"]]
            endpoints = {
                "minimum_loss": min(fit["curves"]["test_loss"]),
                "final_loss": fit["curves"]["test_loss"][-1],
                "final_accuracy_percent": fit["curves"]["test_accuracy_percent"][-1],
            }
        row = evidence[section][index]
        assert task["seed"] in row["extension_seeds"]
        for endpoint, value in endpoints.items():
            published = row["metrics"][endpoint]
            position = published["seeds"].index(task["seed"])
            assert published[task["arm"]][position] == pytest.approx(value, abs=1e-12)
        pair = (section, index, task["seed"])
        pairs.setdefault(pair, set()).add(task["arm"])
    assert len(pairs) == 57
    assert all(arms == {"uniform", "frontloaded"} for arms in pairs.values())
