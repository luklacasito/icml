#!/usr/bin/env python
"""Rebuild the separate CIFAR validation-followup table from paired measurements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.confidence_intervals import _cell, _table, summarize  # noqa: E402

ENDPOINTS = {
    "checkpoint_loss": ("validation_checkpoint", "test_loss"),
    "checkpoint_accuracy_percent": ("validation_checkpoint", "test_accuracy_percent"),
    "final_loss": ("final_epoch", "test_loss"),
    "final_accuracy_percent": ("final_epoch", "test_accuracy_percent"),
}


def comparisons(evidence):
    """Validate seed pairing before passing endpoint vectors to the shared estimator."""
    if evidence.get("schema_version") != 1:
        raise ValueError("Unsupported CIFAR evidence schema")
    rows = []
    for study in evidence["studies"]:
        seeds = study["confirmation_seeds"]
        for comparison in study["comparisons"]:
            arms = [study["recipes"][comparison[key]] for key in ("baseline", "treatment")]
            if comparison["n"] != study["n"] or any(
                arm["seeds"] != seeds or arm["n"] != study["n"] for arm in arms
            ):
                raise ValueError("Recipes must have matching ordered confirmation seeds")
            rows.append(
                {
                    "comparison_id": comparison["comparison_id"],
                    "study": study["label"],
                    "recipe": "Validation-tuned"
                    if comparison["baseline"] == "tuned_uniform"
                    else "Common recipe",
                    "n": study["n"],
                    "seeds": seeds,
                    "profile": study["profiles"]["frontloaded"]["name"],
                    "metrics": {
                        name: {
                            label: arm[endpoint][metric]
                            for label, arm in zip(("uniform", "frontloaded"), arms)
                        }
                        for name, (endpoint, metric) in ENDPOINTS.items()
                    },
                }
            )
    return summarize({"inference": evidence["inference"], "original": [], "benchmarks": rows})[
        "benchmarks"
    ]


def render_table(rows):
    caption = (
        r"CIFAR followups with validation-based selection. Profiles are fixed before tuning: "
        r"big step for the MLP and linearly decreasing dropout for the ViT, each with mean "
        r"dropout $0.1$. The common recipe uses the original learning rate and weight decay; "
        r"the tuned recipe selects them separately for each profile by validation CE. "
        r"Both recipes use fresh confirmation seeds and minimum-validation-CE checkpoints. "
        r"Positive values favor frontloading over uniform; brackets give nominal paired "
        r"95\% intervals (Fieller for relative CE reductions, Student-$t$ for accuracy gains). "
        r"Final evaluations use epoch 75. These cohorts differ from the historical CIFAR "
        r"runs in Table~\ref{tab:loss_improvements}; the ViT row is a small-data pilot. "
        r"The fixed and tuned comparisons share seeds and splits, and the test datasets "
        r"were used in the earlier experiments."
    )
    header = "\n".join(
        [
            r"& & & \multicolumn{2}{c}{Validation-selected test} & \multicolumn{2}{c}{Final-epoch test} \\",
            r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
            r"Experiment & Recipe & $n$ & \shortstack{CE reduction\\(\%)} & \shortstack{Accuracy gain\\(pp)} & \shortstack{CE reduction\\(\%)} & \shortstack{Accuracy gain\\(pp)} \\",
        ]
    )
    cells = []
    for row in rows:
        label = row["study"].split(",")[0].replace(" MLP", r"\\MLP").replace(" ViT", r"\\ViT pilot")
        cells.append(
            [rf"\shortstack[l]{{{label}}}", row["recipe"], str(row["n"])]
            + [_cell(row["metrics"][endpoint]) for endpoint in ENDPOINTS]
        )
    return _table(cells, caption, "tab:cifar_validation", "@{}llrrrrr@{}", header)


def render_markdown(rows):
    lines = [
        "# CIFAR followups with validation-based selection",
        "",
        "Profiles were fixed before tuning: big step for the MLP and linear decreasing for the ViT, "
        "both at mean dropout 0.1. Tuned recipes select learning rate and weight decay separately "
        "for each profile using validation loss. Confirmation uses fresh training seeds on the "
        "recorded split; the test datasets had appeared in historical experiments.",
        "",
        "Positive = improvement over uniform. CE reductions are percentages of the uniform mean; "
        "accuracy gains are percentage points. Brackets show nominal paired 95% intervals "
        "(Fieller for CE, Student-t for accuracy), conditional on the split and selected settings.",
        "",
        "| Experiment | Recipe | n | Checkpoint CE % | Checkpoint accuracy pp | Final CE % | Final accuracy pp |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [row["study"].split(",")[0], row["recipe"], str(row["n"])]
        for metric in row["metrics"].values():
            bounds = metric["ci95"]
            interval = "unbounded" if bounds is None else f"{bounds[0]:.2f}, {bounds[1]:.2f}"
            cells.append(f"{metric['estimate']:+.2f} [{interval}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Checkpoint = test evaluation at the first minimum-validation-CE epoch; final = epoch 75. "
        "These cohorts have no per-epoch test histories. The ViT is a 1,600-training-example pilot. "
        "Common and tuned recipes share seed IDs and data, so they are not independent replications.",
        "",
        "[Measurements and protocol](cifar_validation_evidence.json) · "
        "[Full-precision intervals](cifar_validation_intervals.json)",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "results/supplementary/cifar_validation_evidence.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/confidence")
    args = parser.parse_args()
    evidence = json.loads(args.data.read_text())
    rows = comparisons(evidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cifar_validation_table.tex").write_text(render_table(rows))
    (args.output_dir / "cifar_validation.md").write_text(render_markdown(rows))
    (args.output_dir / "cifar_validation_intervals.json").write_text(
        json.dumps(
            {
                "inference": {
                    **evidence["inference"],
                    "intervals_included": True,
                    "interval_method": "Paired Fieller CE reductions and paired Student-t accuracy gains; nominal 95% intervals with n-1 degrees of freedom.",
                },
                "comparisons": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
