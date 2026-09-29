"""Check the rate sweep, validation selection, and diagnostic evaluation isolation."""

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from benchmarks import run_original as base
from benchmarks.original import relu, validation
from benchmarks.original.rate_sweep import DEFAULT_CONFIG, rate_spec


class RateSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fit(self, root, p=0.001):
        spec = rate_spec(p, 47, smoke=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = validation.run_trial("relu", spec["arm"], 47, "/unused", root,
                                          "cpu", smoke=True, spec=spec)
        directory = Path(root) / "original-smoke/relu" / spec["arm"] / "seed-47"
        return result, directory

    def test_frozen_recipe_and_shared_initialization(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        self.assertEqual(len(config["rates"]) * len(config["seeds"]), 50)
        hashes = []
        for p in (0, 0.001, 0.25):
            spec = rate_spec(p, 47)
            recipe = spec["recipe"]
            self.assertEqual(recipe["epochs"], 35)
            self.assertEqual((recipe["train_size"], recipe["validation_size"], recipe["test_size"]), (5000, 1000, 5000))
            self.assertEqual((recipe["depth"], recipe["width"], recipe["batch_size"]), (12, 256, 100))
            self.assertEqual(recipe["dropout_probabilities"], [p] * 12)
            self.assertEqual((recipe["learning_rate"], recipe["weight_decay"]), (1e-4, 1e-7))
            self.assertNotIn("frontloaded_dropout", recipe)
            torch.manual_seed(47)
            model = base.build_model(spec)
            self.assertTrue(all(torch.count_nonzero(layer.bias) == 0 for layer in model.layers))
            hashes.append(base.initial_state_hash(model))
        self.assertEqual(len(set(hashes)), 1)
        early = rate_spec(0.25, 47, profile="early")["recipe"]
        self.assertEqual(early["dropout_probabilities"], [0.5] * 6 + [0] * 6)
        self.assertEqual(sum(early["dropout_probabilities"]) / 12, early["mean_dropout"])
        for p in (-0.1, 1, float("nan")):
            with self.assertRaises(ValueError):
                rate_spec(p, 47)

    def test_extreme_rates_train_with_finite_gradients_and_verified_restart(self):
        real_train = relu.train_epoch
        checked_epochs = []

        def checked_train(model, *args, **kwargs):
            metrics = real_train(model, *args, **kwargs)
            gradients = [parameter.grad for parameter in model.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertTrue(any(torch.count_nonzero(gradient) > 0 for gradient in gradients))
            checked_epochs.append(1)
            return metrics

        with tempfile.TemporaryDirectory() as temporary, patch.object(relu, "train_epoch", side_effect=checked_train):
            for p in (0.001, 0.25):
                result, directory = self.fit(temporary, p)
                self.assertTrue(all(len(result["curves"][key]) == 2 for key in validation.CURVES))
                self.assertTrue(all(math.isfinite(value) for values in result["curves"].values() for value in values))
                self.assertEqual(result, validation.verified_result(directory, result["spec"], result["source"], result["data"]))
                self.assertEqual(result, self.fit(temporary, p)[0])
                self.assertEqual(result["curves"]["learning_rate_after_step"][-1], 1e-7)
        self.assertEqual(len(checked_epochs), 4)

    def test_first_validation_minimum_restores_matching_weights(self):
        real_evaluate = validation.evaluate
        validation_states, test_states = [], []

        def tied_validation(model, data, *args):
            fingerprint = base.initial_state_hash(model)
            if len(data[0]) == 10:
                validation_states.append(fingerprint)
                return 2.0, 10.0
            test_states.append(fingerprint)
            return real_evaluate(model, data, *args)

        with tempfile.TemporaryDirectory() as temporary, patch.object(validation, "evaluate", side_effect=tied_validation):
            result, directory = self.fit(temporary)
            self.assertEqual(result["metrics"]["best_validation_epoch_zero_based"], 0)
            self.assertEqual(test_states, [validation_states[0], validation_states[1], validation_states[0]])
            self.assertNotEqual(validation_states[0], validation_states[1])
            best = torch.load(directory / "best.pt", weights_only=True)
            self.assertEqual(best["epoch_zero_based"], 0)
            self.assertEqual(result["metrics"]["validation_selected_test_loss"], result["curves"]["test_loss"][0])
            result["curves"]["test_loss"][0] += 1
            base.atomic_json(directory / "result.json", result)
            with self.assertRaisesRegex(RuntimeError, "differs from its curve"):
                validation.verified_result(directory, result["spec"], result["source"], result["data"])

    def test_diagnostic_test_passes_leave_training_and_final_weights_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            full, full_dir = self.fit(Path(temporary) / "full", 0.25)
            real_evaluate = validation.evaluate
            test_calls = 0

            def skip_diagnostic_forward(model, data, *args):
                nonlocal test_calls
                if len(data[0]) == 4:
                    test_calls += 1
                    if test_calls <= 2:
                        return (full["curves"]["test_loss"][test_calls - 1],
                                full["curves"]["test_accuracy_percent"][test_calls - 1])
                return real_evaluate(model, data, *args)

            with patch.object(validation, "evaluate", side_effect=skip_diagnostic_forward):
                skipped, skipped_dir = self.fit(Path(temporary) / "skipped", 0.25)
            self.assertEqual(full["curves"], skipped["curves"])
            self.assertEqual(full["metrics"], skipped["metrics"])
            full_state = torch.load(full_dir / "final.pt", weights_only=True)["model_state_dict"]
            skipped_state = torch.load(skipped_dir / "final.pt", weights_only=True)["model_state_dict"]
            self.assertTrue(all(torch.equal(value, skipped_state[key]) for key, value in full_state.items()))

    def test_evaluation_detects_rng_consumption(self):
        relu.SHUFFLE_GENERATOR = torch.Generator().manual_seed(47)

        def consuming_evaluation(*args):
            torch.rand(1)
            return 1.0, 10.0

        with patch.object(relu, "evaluate", side_effect=consuming_evaluation):
            with self.assertRaisesRegex(RuntimeError, "random stream"):
                validation.evaluate(None, None, None, 3, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
