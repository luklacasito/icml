"""Build the paper's plots from saved measurements and mean-field calculations."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/figures")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    jobs = [
        ("plot_mean_field.py", "theory"),
        ("plot_paper_results.py", "experiments"),
        ("plot_sliding_block.py", "experiments/mlp"),
        ("plot_cifar_correlations.py", "experiments/mlp"),
        ("plot_benchmarks.py", "experiments/benchmarks"),
    ]
    for script, directory in jobs:
        print(f"Building {script}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / script),
                "--output-dir",
                str(output / directory),
            ],
            cwd=ROOT,
            check=True,
        )

    # These supplied vector illustrations are retained, not recalculated.
    for name in ("river_ink.pdf", "regularization_reach.pdf"):
        source = ROOT / "manuscript/figures/theory" / name
        destination = output / "theory" / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    figures = set()
    for section in (ROOT / "manuscript/sections").glob("*.tex"):
        figures.update(re.findall(r"\\PaperRoot/figures/([^}]+\.pdf)", section.read_text()))
    missing = sorted(name for name in figures if not (output / name).is_file())
    if missing:
        raise FileNotFoundError(f"Paper figures not produced: {missing}")
    print(f"Ready: {len(figures)} figure files in {output} (including two supplied illustrations).")


if __name__ == "__main__":
    main()
