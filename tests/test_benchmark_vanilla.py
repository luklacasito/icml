"""Standalone vanilla recipe: pairing, chronology, sealed test, and exact resume."""

import json
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import numpy as np
import torch

from benchmarks.vanilla import (
    ARCHITECTURES,
    PROFILES,
    StageDropout,
    VanillaMLP,
    Windows,
    digest,
    fingerprint,
    load_data,
    profile_layers,
    trial,
)
from benchmarks.run_vanilla import (
    make_confirmation,
    make_plans,
    make_smoke_data,
    metrics,
    run,
    select_rates,
    source_hash,
)


torch.set_num_threads(1)


class ScheduleAndModelTests(unittest.TestCase):
    def test_profiles_preserve_mean_and_direction(self):
        for depth in (6, 12):
            for profile in PROFILES:
                values = profile_layers(profile, depth)
                expected = 0.0 if profile == "none" else 0.1
                self.assertAlmostEqual(float(np.mean(values)), expected)
            self.assertEqual(
                profile_layers("step_early", depth)[::-1],
                profile_layers("step_late", depth),
            )
            self.assertEqual(
                profile_layers("linear_early", depth)[::-1],
                profile_layers("linear_late", depth),
            )

    def test_plain_architectures_and_parameter_counts(self):
        counts = {
            (6, 256): 1_640_707,
            (12, 256): 2_035_459,
            (6, 512): 3_936_771,
            (12, 512): 5_512_707,
        }
        for depth, width in ARCHITECTURES:
            model = VanillaMLP(128 * 40, width, depth, [0.1] * depth, 42)
            self.assertEqual(
                sum(p.numel() for p in model.parameters()), counts[depth, width]
            )
            self.assertEqual(model(torch.randn(2, 128, 40)).shape, (2, 3))
            self.assertFalse(
                any("Norm" in type(module).__name__ for module in model.modules())
            )

    def test_dropout_statistics_eval_and_resume(self):
        module = StageDropout(0.2, 42)
        inputs = torch.ones(200_000)
        output = module(inputs)
        self.assertAlmostEqual(float((output == 0).float().mean()), 0.2, delta=0.004)
        self.assertAlmostEqual(float(output.mean()), 1.0, delta=0.005)
        state = module.rng_state()
        expected = module(inputs)
        resumed = StageDropout(0.2, 42)
        resumed.restore_rng(state)
        self.assertTrue(torch.equal(expected, resumed(inputs)))
        module.eval()
        state = module.rng_state().clone()
        self.assertIs(module(inputs), inputs)
        self.assertTrue(torch.equal(state, module.rng_state()))

    def test_initialization_variance_is_near_declared_value(self):
        torch.manual_seed(1)
        model = VanillaMLP(5120, 512, 6, [0.0] * 6, 9)
        first = model.hidden[0]
        self.assertAlmostEqual(
            float(first.weight.detach().var()) * 5120, 1.98, delta=0.03
        )
        self.assertAlmostEqual(float(first.bias.detach().var()), 0.02, delta=0.004)


class ProtocolAndDataTests(unittest.TestCase):
    def test_plan_counts_pairing_and_test_policy(self):
        plans = make_plans("data", "code")
        self.assertEqual(len(plans["canary"]["trials"]), 4)
        self.assertEqual(len(plans["lr_search"]["trials"]), 48)
        self.assertTrue(
            all(not item["data"]["test_days"] for item in plans["lr_search"]["trials"])
        )
        early = trial("confirmation", 6, 256, "linear_early", 1e-4, 100, 2, "d", "s")
        late = trial("confirmation", 6, 256, "linear_late", 1e-4, 100, 2, "d", "s")
        uniform = trial("confirmation", 6, 256, "uniform", 1e-4, 100, 2, "d", "s")
        self.assertEqual(
            early["randomization"]["dropout_seed"],
            late["randomization"]["dropout_seed"],
        )
        self.assertEqual(
            early["randomization"]["initialization_seed"],
            uniform["randomization"]["initialization_seed"],
        )
        self.assertEqual(
            early["randomization"]["minibatch_seed"],
            uniform["randomization"]["minibatch_seed"],
        )
        self.assertEqual(early["data"]["test_days"], [8, 9, 10])

    def test_windows_do_not_cross_segments_or_future_tail(self):
        segments = [
            {
                "x": torch.full((150, 40), float(index)),
                "y": torch.zeros(150, 1, dtype=torch.long),
                "day": index + 1,
                "stock": 1,
            }
            for index in range(2)
        ]
        dataset = Windows(segments, np.zeros(40), np.ones(40))
        self.assertEqual(len(dataset), 26)
        for index in range(len(dataset)):
            inputs, _, _ = dataset[index]
            self.assertEqual(len(inputs.unique()), 1)
            segment, start = dataset.locate(index)
            self.assertLess(start + 127 + 10, len(segments[segment]["x"]))

    def test_metrics_keep_absent_classes(self):
        labels = np.array([0, 0, 1])
        probabilities = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.7, 0.2, 0.1]])
        result = metrics(labels, probabilities)
        self.assertAlmostEqual(
            result["logloss"], -(np.log(0.8) + np.log(0.6) + np.log(0.2)) / 3
        )
        self.assertAlmostEqual(result["macro_f1"], 0.8 / 3)
        self.assertEqual(result["confusion"], [[2, 0, 0], [1, 0, 0], [0, 0, 0]])


class ResumeTest(unittest.TestCase):
    def make_data(self, root):
        entries = []
        rng = np.random.default_rng(7)
        for day in range(6, 11):
            path = root / f"day{day}.npz"
            x = rng.normal(size=(30, 40)).astype(np.float32)
            y = rng.integers(0, 3, size=(30, 1), dtype=np.int64)
            np.savez(path, x=x, y=y)
            entries.append(
                {"day": day, "stock": 1, "file": path.name, "sha256": digest(path)}
            )
        manifest = {
            "schema": "fi2010_segments_v1",
            "tail_purge_representations": 10,
            "segments": entries,
        }
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_exact_epoch_resume_and_sealed_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_data(root)
            spec = trial(
                "test_fixture",
                6,
                8,
                "linear_early",
                1e-4,
                0,
                2,
                digest(root / "manifest.json"),
                source_hash(),
            )
            spec["input_shape"] = [8, 40]
            spec["training"]["batch_size"] = 8
            spec["canary_examples"] = 8
            full_path, resume_path = root / "full", root / "resume"
            complete = run(spec, root, full_path, "cpu")
            partial = run(spec, root, resume_path, "cpu", stop_after_epoch=1)
            resumed = run(spec, root, resume_path, "cpu")
            self.assertEqual(partial, {"status": "checkpointed", "epoch": 1})
            self.assertEqual(complete["metrics"], resumed["metrics"])
            self.assertEqual(complete["best_epoch"], resumed["best_epoch"])
            self.assertFalse(complete["test_evaluated"])
            self.assertFalse((full_path / "test_predictions.npz").exists())
            self.assertFalse((resume_path / "latest.pt").exists())
            self.assertEqual(
                json.loads((resume_path / "result.json").read_text())["fingerprint"],
                fingerprint(spec),
            )


class ReproductionTests(unittest.TestCase):
    def test_learning_rate_selection_uses_validation_and_lower_rate_tiebreak(self):
        plans = make_plans("data", "source")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = plans["lr_search"]
            for index, spec in enumerate(plan["trials"]):
                path = root / f"{index:03d}_{fingerprint(spec)[:12]}"
                path.mkdir()
                result = {
                    "status": "complete",
                    "fingerprint": fingerprint(spec),
                    "spec": spec,
                    "test_evaluated": False,
                    "epoch_seconds": [0.0] * 50,
                    "metrics": {"validation": {"logloss": 1.0}},
                }
                (path / "result.json").write_text(json.dumps(result))
            selected, _ = select_rates(plan, root)
            self.assertEqual(set(selected.values()), {3e-5})
            confirmation = make_confirmation(plan, selected)
            self.assertEqual(len(confirmation["trials"]), 140)
            self.assertEqual(
                set(s["seed"] for s in confirmation["trials"]), set(range(100, 105))
            )
            self.assertTrue(
                all(s["training"]["epochs"] == 100 for s in confirmation["trials"])
            )
            # A completed run that touched test is ineligible for validation selection.
            first = plan["trials"][0]
            path = root / f"000_{fingerprint(first)[:12]}" / "result.json"
            result = json.loads(path.read_text())
            result["test_evaluated"] = True
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "premature test"):
                select_rates(plan, root)

    def test_normalization_is_training_only_and_endpoint_labels_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_smoke_data(root)
            datasets, info = load_data(root, include_test=True, sequence=8)
            self.assertEqual(
                info["counts"], {"train": 13, "validation": 13, "test": 39}
            )
            segment = datasets["train"].segments[0]
            expected_mean = segment["x"].numpy()[:-10].mean(0, dtype=np.float64)
            np.testing.assert_array_equal(info["normalizer_mean"], expected_mean)
            x, y, index = datasets["train"][0]
            self.assertTrue(torch.equal(y, segment["y"][7, 0]))
            self.assertEqual(index, 0)
            self.assertEqual(tuple(x.shape), (8, 40))
            np.testing.assert_array_equal(
                datasets["test"].example_ids([0, 12, 13, 38]),
                [[8, 1, 0, 7], [8, 1, 12, 19], [9, 1, 0, 7], [10, 1, 12, 19]],
            )
            # Changing a validation segment must not change the normalizer.
            manifest = json.loads((root / "manifest.json").read_text())
            path = root / "day7.npz"
            with np.load(path, allow_pickle=False) as archive:
                values = {key: archive[key].copy() for key in archive.files}
            values["x"] += 1000
            np.savez(path, **values)
            for item in manifest["segments"]:
                if item["day"] == 7:
                    item["sha256"] = digest(path)
            (root / "manifest.json").write_text(json.dumps(manifest))
            _, changed = load_data(root, include_test=False, sequence=8)
            self.assertEqual(info["normalizer_mean"], changed["normalizer_mean"])
            self.assertEqual(info["normalizer_std"], changed["normalizer_std"])

    def test_confirmation_evaluates_test_once_at_validation_checkpoint(self):
        from benchmarks import run_vanilla as implementation

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_smoke_data(root)
            spec = trial(
                "confirmation",
                6,
                8,
                "step_early",
                1e-4,
                100,
                2,
                digest(root / "manifest.json"),
                source_hash(),
            )
            spec["input_shape"] = [8, 40]
            spec["training"]["batch_size"] = 8
            calls = []
            evaluate = implementation.evaluate

            def record(model, dataset, *args, **kwargs):
                calls.append(tuple(segment["day"] for segment in dataset.segments))
                return evaluate(model, dataset, *args, **kwargs)

            output = root / "confirmation-fixture"
            with patch.object(implementation, "evaluate", side_effect=record):
                result = run(spec, root, output, "cpu")
            self.assertEqual(calls.count((8, 9, 10)), 1)
            self.assertEqual(calls.count((7,)), 3)
            self.assertTrue(result["test_evaluated"])
            self.assertEqual(result["metrics"]["test"]["n"], 39)
            history = json.loads((output / "history.json").read_text())
            best = min(history, key=lambda epoch: epoch["validation"]["logloss"])
            self.assertEqual(result["best_epoch"], best["epoch"])
            self.assertAlmostEqual(history[-1]["learning_rate"], 1e-6)
            with np.load(
                output / "test_predictions.npz", allow_pickle=False
            ) as predictions:
                self.assertEqual(predictions["probabilities"].shape, (39, 3))
                self.assertEqual(
                    predictions["day_stock_window_start_label_endpoint"].shape, (39, 4)
                )


if __name__ == "__main__":
    unittest.main()
