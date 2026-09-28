"""Check original-extension subset order, provenance, and retained endpoints."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from benchmarks import run_original


class OriginalExtensionTests(unittest.TestCase):
    def test_subset_rng_uses_train_then_test(self):
        expected_rng = np.random.RandomState(0)
        expected_train = expected_rng.choice(100, 12, replace=False)
        expected_test = expected_rng.choice(40, 8, replace=False)
        train, test = run_original.subset_indices(100, 40, 12, 8)
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)
        train, test = run_original.subset_indices(100, 40, 100, 40)
        np.testing.assert_array_equal(train, np.arange(100))
        np.testing.assert_array_equal(test, np.arange(40))

    def test_source_identity_includes_reused_utils_and_public_runner(self):
        source = run_original.source_provenance()
        for name in (
            "benchmarks/run_original.py",
            "benchmarks/original/relu.py",
            "benchmarks/original/vit.py",
            "benchmarks/original/recipes.json",
            "utils/training.py",
            "utils/schedules.py",
        ):
            self.assertIn(name, source["files"])
        self.assertTrue(all(not Path(name).is_absolute() for name in source["files"]))

    def test_final_weights_reproduce_metrics_and_completed_fit_is_reused(self):
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                root = Path(directory)
                for study in ("relu", "vit"):
                    result = run_original.run_trial(
                        study, "frontloaded", 900001, None, root, "cpu", smoke=True
                    )
                    output = root / "original-smoke" / study / "frontloaded" / "seed-900001"
                    checkpoint_path = output / "final.pt"
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    self.assertEqual(checkpoint["endpoint"], "final_epoch")
                    self.assertEqual(checkpoint["epoch_zero_based"], 1)
                    model = run_original.build_model(result["spec"])
                    model.load_state_dict(checkpoint["model_state_dict"])
                    _, test, _ = run_original.load_data(result["spec"], None, torch.device("cpu"))
                    criterion = torch.nn.CrossEntropyLoss()
                    if study == "relu":
                        loss, accuracy = run_original.relu.evaluate(model, test, criterion, 3)
                    else:
                        loss, accuracy = run_original.vit_training.evaluate(
                            model, test, criterion, 3
                        )
                    np.testing.assert_allclose(
                        [loss, accuracy],
                        [
                            result["metrics"]["final_epoch_test_loss"],
                            result["metrics"]["final_epoch_test_accuracy_percent"],
                        ],
                        rtol=1e-6,
                    )
                    before = checkpoint_path.stat().st_mtime_ns
                    repeated = run_original.run_trial(
                        study, "frontloaded", 900001, None, root, "cpu", smoke=True
                    )
                    self.assertEqual(result, repeated)
                    self.assertEqual(before, checkpoint_path.stat().st_mtime_ns)
                    saved = json.loads((output / "result.json").read_text())
                    self.assertEqual(
                        saved["metrics"]["retrospective_min_test_loss"],
                        min(saved["curves"]["test_loss"]),
                    )
        finally:
            torch.set_num_threads(original_threads)


if __name__ == "__main__":
    unittest.main()
