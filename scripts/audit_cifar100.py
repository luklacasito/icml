"""Check the saved CIFAR-100 curves and their conditional dataset-size inference."""

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = json.loads((ROOT / "results/cifar100_configuration.json").read_text())
    raw = (ROOT / evidence["result_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence["result_sha256"]:
        raise ValueError("CIFAR-100 results differ from the audited archive")
    data = json.loads(raw)
    confirmed = evidence["confirmed"]
    if list(data) != confirmed["profiles"]:
        raise ValueError("Unexpected dropout profiles")
    for profile in data.values():
        for history in profile.values():
            if len(history) != confirmed["runs_per_profile"] or any(
                len(row) != confirmed["epochs_per_run"] for row in history
            ):
                raise ValueError("Unexpected number of runs or epochs")

    checks = {}
    for metric in ("train_acc", "test_acc"):
        values = [v for profile in data.values() for row in profile[metric] for v in row]
        if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
            raise ValueError(f"Invalid percentages in {metric}")
        denominator = math.lcm(*((Fraction(str(v)) / 100).denominator for v in values))
        checks[metric] = {
            "values_checked": len(values),
            "minimum_common_sample_denominator": denominator,
        }
    if checks != evidence["dataset_size_inference"]["checks"]:
        raise ValueError("Accuracy denominators differ from the documented inference")
    print(json.dumps(checks, indent=2))
    print("Fingerprint and curve dimensions match the audited archive.")
    print("Dataset sizes follow under the evaluator assumptions in the evidence file.")
    print("This does not recover the settings used for the saved runs.")


if __name__ == "__main__":
    main()
