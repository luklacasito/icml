#!/usr/bin/env python
"""Reproduce the paper's paired-seed intervals and two comparison tables.

Only NumPy and SciPy are required. The portable input contains endpoint vectors
in matching seed order; it does not select profiles or checkpoints again. An
endpoint can name a subset of the row's seeds when historical values are absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t


def _paired(uniform, frontloaded):
    arrays = tuple(np.asarray(values, dtype=float) for values in (uniform, frontloaded))
    if (
        any(values.ndim != 1 for values in arrays)
        or arrays[0].shape != arrays[1].shape
        or len(arrays[0]) < 2
        or not all(np.isfinite(values).all() for values in arrays)
    ):
        raise ValueError("Need at least two finite observations in matching seed order")
    return arrays


def fieller_reduction(uniform, frontloaded):
    """95% paired Fieller set for 100 * (1 - mean(frontloaded)/mean(uniform)).

    Invert the paired t statistic for F - r U, retaining its covariance. An
    uncertain denominator can give an unbounded set; never turn that into a
    finite interval. A null bound in ``intervals`` represents infinity.
    """
    uniform, frontloaded = _paired(uniform, frontloaded)
    if np.any(uniform < 0) or np.any(frontloaded < 0) or uniform.mean() <= 0:
        raise ValueError("Losses must be nonnegative with a positive uniform mean")
    n = len(uniform)
    u, f = float(uniform.mean()), float(frontloaded.mean())
    covariance = np.cov([uniform, frontloaded], ddof=1) / n
    q = float(t.ppf(0.975, n - 1)) ** 2
    ratio = f / u
    a = u * u - q * covariance[0, 0]
    # Center the quadratic at the estimated ratio to avoid cancellation when
    # paired outcomes are nearly proportional.
    b = 2 * q * (covariance[0, 1] - ratio * covariance[0, 0])
    residual_variance = float(np.var(frontloaded - ratio * uniform, ddof=1) / n)
    c = -q * residual_variance
    discriminant = max(0.0, float(b * b - 4 * a * c))
    estimate = 100 * (u - f) / u
    result = {"estimate": estimate, "df": n - 1, "ci95": None}
    if a == 0:
        if b == 0:
            result.update(kind="unbounded", intervals=[[None, None]])
        else:
            boundary = 100 * (1 - (ratio - c / b))
            bounds = [boundary, None] if b > 0 else [None, boundary]
            result.update(kind="unbounded", intervals=[bounds])
    elif a < 0 and b * b - 4 * a * c <= 0:
        result.update(kind="unbounded", intervals=[[None, None]])
    else:
        root = math.sqrt(discriminant)
        roots = sorted((ratio + (-b - root) / (2 * a), ratio + (-b + root) / (2 * a)))
        low, high = 100 * (1 - roots[1]), 100 * (1 - roots[0])
        if a > 0:
            result.update(kind="bounded", ci95=[low, high], intervals=[[low, high]])
        else:
            result.update(kind="disjoint", intervals=[[None, low], [high, None]])
    return result


def accuracy_gain(uniform, frontloaded):
    """95% paired Student-t interval; inputs are accuracies in percent units."""
    uniform, frontloaded = _paired(uniform, frontloaded)
    if any(np.any((values < 0) | (values > 100)) for values in (uniform, frontloaded)):
        raise ValueError("Accuracy vectors must be in percent units between 0 and 100")
    differences = frontloaded - uniform
    estimate = float(differences.mean())
    sem = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    radius = float(t.ppf(0.975, len(differences) - 1)) * sem
    return {
        "estimate": estimate,
        "ci95": [estimate - radius, estimate + radius],
        "df": len(differences) - 1,
        "kind": "bounded",
    }


def default_data_path():
    candidate = Path(__file__).resolve().parents[1] / "results/confidence_seed_metrics.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("Pass --data pointing to confidence_seed_metrics.json")


def load_evidence(path=None):
    data = json.loads(Path(path or default_data_path()).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported confidence evidence schema")
    return data


def summarize(evidence):
    """Compute intervals from vectors, preserving the input's fixed row order."""
    result = {"schema_version": 1, "inference": evidence["inference"]}
    for section in ("original", "benchmarks"):
        result[section] = []
        for row in evidence[section]:
            seeds = row["seeds"]
            if len(seeds) != row["n"] or len(set(seeds)) != len(seeds):
                raise ValueError("Seed identifiers must be unique and match n")
            summary = {key: value for key, value in row.items() if key != "metrics"}
            summary["metrics"] = {}
            for name, pair in row["metrics"].items():
                if pair is None:
                    summary["metrics"][name] = None
                    continue
                endpoint_seeds = pair.get("seeds", seeds)
                if len(set(endpoint_seeds)) != len(endpoint_seeds) or not set(
                    endpoint_seeds
                ).issubset(seeds):
                    raise ValueError("Endpoint seeds must be a unique subset of row seeds")
                if any(len(pair[arm]) != len(endpoint_seeds) for arm in ("uniform", "frontloaded")):
                    raise ValueError("Endpoint vectors must match their paired seed identifiers")
                function = accuracy_gain if name.endswith("accuracy_percent") else fieller_reduction
                metric = function(pair["uniform"], pair["frontloaded"])
                metric.update(n=len(endpoint_seeds), seeds=list(endpoint_seeds))
                for arm in ("uniform", "frontloaded"):
                    metric[f"{arm}_mean"] = float(np.mean(pair[arm]))
                summary["metrics"][name] = metric
            result[section].append(summary)
    return result


def _tex(text):
    return str(text).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _cell(metric):
    if metric is None:
        return "--"
    interval = metric["ci95"]
    bounds = "unbounded" if interval is None else f"$[{interval[0]:.2f}, {interval[1]:.2f}]$"
    return rf"\shortstack{{${metric['estimate']:+.2f}$\\[-1pt]{{\scriptsize {bounds}}}}}"


def _dataset_label(row):
    extended = row["dataset"] == "Jannis" and row.get("weight_decay") == 1e-7
    return row["dataset"] + (" extended" if extended else "")


def _sample_sizes(row, endpoints):
    """Use the observations available for the displayed endpoints."""
    counts = [
        str(row["metrics"][name]["n"]) if row["metrics"][name] else "--" for name in endpoints
    ]
    return counts[0] if len(set(counts)) == 1 else "/".join(counts)


def _start(caption, label, columns):
    return [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
    ]


def _finish(lines):
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]) + "\n"


def render_original_table(comparisons):
    caption = (
        r"Original paper experiments. Positive changes favor the named profile over uniform. "
        r"CE reduction is $100(L_U-L_F)/L_U$ using seed means; accuracy gain is in percentage points. "
        r"Brackets give nominal paired 95\% confidence intervals (Fieller for CE; Student-$t$ for accuracy), "
        r"conditional on the selected profiles and fixed data split. Profiles were selected by mean final "
        r"test CE among the historical nonuniform runs, then frozen for the additional seeds; they stay "
        r"fixed across columns. Pooled intervals do not account for the earlier selection. Final CE and accuracy "
        r"use the last epoch. Minimum CE averages each seed's lowest recorded test loss; this "
        r"retrospective endpoint uses test data to choose an epoch, not validation. The first two rows "
        r"share the same uniform runs. See the main text for the assumptions and selection limits."
    )
    lines = _start(caption, "tab:loss_improvements", "@{}lrlrrr@{}")
    lines += [
        r"Experiment & $n$ & Profile & \shortstack{Final CE\\reduction (\%)} & \shortstack{Min. test CE\\reduction (\%)} & \shortstack{Final accuracy\\gain (pp)} \\",
        r"\midrule",
    ]
    for row in comparisons:
        model = _tex(row["model"])
        cells = [
            rf"\shortstack[l]{{{_tex(row['dataset'])}\\{model}}}",
            str(row["n"]),
            _tex(row["profile"]),
        ]
        cells.extend(
            _cell(row["metrics"][name])
            for name in ("final_loss", "minimum_loss", "final_accuracy_percent")
        )
        lines.append(" & ".join(cells) + r" \\")
    return _finish(lines)


def render_frontloaded_table(comparisons):
    caption = (
        r"Frontloaded dropout across tasks. Profiles were selected by lowest mean minimum validation CE "
        r"among complete paired arms in the historical cohorts, then frozen for the additional seeds. "
        r"Each profile stays fixed across columns; pooled intervals do not account for the earlier selection. "
        r"Positive values favor frontloading. CE reduction is $100(L_U-L_F)/L_U$ using seed means; "
        r"accuracy gain is in percentage points. Brackets give nominal paired 95\% confidence intervals "
        r"(Fieller for CE; Student-$t$ for accuracy); see the main text for assumptions and selection limits. "
        r"Checkpoint metrics use each run's minimum-validation-loss checkpoint; final metrics use its last epoch. "
        r"The $n$ column gives checkpoint/final (C/F) paired seed counts. Where historical final evaluations were "
        r"absent, final results use only the five additional pairs. "
        r"The extended Jannis Transformer uses weight decay $10^{-7}$ and fixed mean dropout $0.10$; "
        r"other rows use zero weight decay and allow profile-specific dropout budgets. Learning rate and "
        r"mean dropout follow each cohort's validation protocol. Per-epoch test histories were not recorded. "
        r"Dashes mean missing final test evaluations, not zero change."
    )
    lines = _start(caption, "tab:frontloaded_results", "@{}lrlrrrr@{}")
    lines += [
        r"& & & \multicolumn{2}{c}{Validation-selected test} & \multicolumn{2}{c}{Final-epoch test} \\",
        r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Experiment (training $N$) & \shortstack{$n$\\C/F} & Profile & \shortstack{CE reduction\\(\%)} & \shortstack{Accuracy gain\\(pp)} & \shortstack{CE reduction\\(\%)} & \shortstack{Accuracy gain\\(pp)} \\",
        r"\midrule",
    ]
    for index, row in enumerate(comparisons):
        if index and comparisons[index - 1]["dataset"] != row["dataset"]:
            lines.append(r"\addlinespace[3pt]")
        dataset = _dataset_label(row)
        checkpoint_n = _sample_sizes(row, ("checkpoint_loss", "checkpoint_accuracy_percent"))
        final_n = _sample_sizes(row, ("final_loss", "final_accuracy_percent"))
        cells = [
            rf"\shortstack[l]{{{_tex(dataset)} ({row['train_size']:,})\\{_tex(row['model'])}}}",
            f"{checkpoint_n}/{final_n}",
            _tex(row["profile"]),
        ]
        cells.extend(
            _cell(row["metrics"][name])
            for name in (
                "checkpoint_loss",
                "checkpoint_accuracy_percent",
                "final_loss",
                "final_accuracy_percent",
            )
        )
        lines.append(" & ".join(cells) + r" \\")
    return _finish(lines)


def write_outputs(output_dir, evidence):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(evidence)
    (output_dir / "original_results_table.tex").write_text(
        render_original_table(summary["original"])
    )
    (output_dir / "frontloaded_table.tex").write_text(
        render_frontloaded_table(summary["benchmarks"])
    )
    (output_dir / "confidence_intervals.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    records = []
    for section in ("original", "benchmarks"):
        for row in summary[section]:
            for endpoint, metric in row["metrics"].items():
                interval = metric["ci95"] if metric else None
                records.append(
                    {
                        "table": section,
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "train_size": row.get("train_size"),
                        "weight_decay": row.get("weight_decay"),
                        "n": metric["n"] if metric else None,
                        "profile": row["profile"],
                        "endpoint": endpoint,
                        "units": "percentage points"
                        if endpoint.endswith("accuracy_percent")
                        else "percent reduction",
                        "uniform_mean": metric["uniform_mean"] if metric else None,
                        "frontloaded_mean": metric["frontloaded_mean"] if metric else None,
                        "estimate": metric["estimate"] if metric else None,
                        "ci95_lower": interval[0] if interval else None,
                        "ci95_upper": interval[1] if interval else None,
                        "interval_kind": metric["kind"] if metric else "missing",
                    }
                )
    with (output_dir / "confidence_intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    markdown = ["# Paired-seed confidence intervals", "", evidence["inference"], ""]
    markdown += [
        f"- **{key.replace('_', ' ')}:** {value}"
        for key, value in evidence["endpoint_definitions"].items()
    ]
    markdown += [
        "",
        "Positive values favor frontloading. Loss changes are percentages; accuracy changes are percentage points.",
        "Seed counts refer to the displayed endpoints. If counts differ within a table, they follow the endpoint column order.",
        "",
    ]
    for title, section, endpoints in (
        (
            "Original paper",
            "original",
            [
                ("final_loss", "Final CE reduction %"),
                ("minimum_loss", "Minimum test CE reduction %"),
                ("final_accuracy_percent", "Final accuracy gain pp"),
            ],
        ),
        (
            "Validation-selected test checkpoints",
            "benchmarks",
            [
                ("checkpoint_loss", "Checkpoint CE reduction %"),
                ("checkpoint_accuracy_percent", "Checkpoint accuracy gain pp"),
            ],
        ),
        (
            "Recorded final-epoch test results",
            "benchmarks",
            [
                ("final_loss", "Final CE reduction %"),
                ("final_accuracy_percent", "Final accuracy gain pp"),
            ],
        ),
    ):
        markdown += [f"## {title}", ""]
        headings = ["Experiment", "n", "Profile"] + [label for _, label in endpoints]
        markdown += [
            "| " + " | ".join(headings) + " |",
            "|---|---:|---|" + "---:|" * len(endpoints),
        ]
        for row in summary[section]:
            if all(row["metrics"][endpoint] is None for endpoint, _ in endpoints):
                continue
            dataset = _dataset_label(row)
            experiment = f"{dataset} / {row['model']}"
            if row.get("train_size"):
                experiment += f" (N={row['train_size']:,})"
            cells = [
                experiment,
                _sample_sizes(row, [endpoint for endpoint, _ in endpoints]),
                row["profile"],
            ]
            for endpoint, _ in endpoints:
                metric = row["metrics"][endpoint]
                if metric is None:
                    cells.append("—")
                elif metric["ci95"] is None:
                    cells.append(f"{metric['estimate']:+.2f} [unbounded]")
                else:
                    low, high = metric["ci95"]
                    cells.append(f"{metric['estimate']:+.2f} [{low:.2f}, {high:.2f}]")
            markdown.append("| " + " | ".join(cells) + " |")
        markdown.append("")
    markdown += [
        "Extended Jannis uses weight decay 1e-7; the other benchmark rows use zero weight decay.",
        "",
        "[Full-precision CSV](confidence_intervals.csv) · [Structured results](confidence_intervals.json)",
        "",
    ]
    (output_dir / "confidence_intervals.md").write_text("\n".join(markdown).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Portable paired seed metrics JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs/confidence",
    )
    args = parser.parse_args()
    write_outputs(args.output_dir, load_evidence(args.data))


if __name__ == "__main__":
    main()
