"""Summarise the full training runs (our ViT, without vs with registers).

Reads the run folders written by src.train, then:
    - copies the logs (not the model weights) into results/runs/ so they go on GitHub
    - plots train loss and validation accuracy per epoch
    - writes a results table and copies it into README.md

Usage (from the repo root):
    python -m scripts.summarise_training --runs /path/to/tiny_reg0 /path/to/tiny_reg4

The paper's matching claim (Table 2a): adding registers does not hurt accuracy
(ImageNet top-1 changes by -0.1 to +0.5 points across DeiT-III, OpenCLIP, DINOv2).
"""

import argparse
import json
import shutil
from pathlib import Path

from src.plots import plot_training_curves
from src.report import inject_into_readme, markdown_table
from src.utils import REPO_ROOT

LOG_FILES = ("config.json", "log.csv", "final.json")
PAPER_TABLE_2A = "DeiT-III 84.7 → 84.7, OpenCLIP 78.2 → 78.1, DINOv2 84.3 → 84.8"


def load_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    final = json.loads((run_dir / "final.json").read_text())
    with open(run_dir / "log.csv") as f:
        epoch_times = [float(line.split(",")[-1]) for line in f.read().splitlines()[1:] if line]
    return {"config": config, "final": final, "minutes": sum(epoch_times) / 60}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise full training runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="run folders, no-register run first")
    args = parser.parse_args()

    runs = {}
    for run_dir in map(Path, args.runs):
        if not (run_dir / "final.json").exists():
            raise SystemExit(f"{run_dir} has no final.json: the run did not finish.")
        runs[run_dir.name] = load_run(run_dir)
        # Copy the logs into the repo (the model weights stay where they are; too big for GitHub).
        target = REPO_ROOT / "results" / "runs" / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for name in LOG_FILES:
            shutil.copy(run_dir / name, target / name)

    labels = {name: f"{r['config']['registers']} registers" for name, r in runs.items()}
    plot_training_curves(
        {labels[name]: REPO_ROOT / "results" / "runs" / name for name in runs},
        REPO_ROOT / "results" / "figures" / "full_training_runs.png",
        "Full training: ViT-Tiny on Tiny-ImageNet (100k images)",
    )

    rows = []
    for name, r in runs.items():
        c, f = r["config"], r["final"]
        rows.append([f"ViT-{c['size'].capitalize()}", c["registers"], f"{c['params']:,}", f["epochs_run"],
                     f"**{100 * f['val_acc']:.2f}%**", f"{f['val_loss']:.3f}", f"{r['minutes']:.0f}"])
    table = markdown_table(
        ["Model", "Registers", "Parameters", "Epochs", "Val accuracy", "Val loss", "Training time (min)"], rows)

    accs = [r["final"]["val_acc"] for r in runs.values()]
    comparison = ""
    if len(accs) == 2:
        diff = 100 * (accs[1] - accs[0])
        comparison = (
            f"Effect of adding registers: **{diff:+.2f} percentage points**. "
            f"Paper (Table 2a, ImageNet top-1): {PAPER_TABLE_2A}, i.e. registers do not hurt accuracy. "
            "Our absolute accuracies are not comparable with the paper's (different dataset, much smaller "
            "model, shorter training); only the with/without-registers difference is. Each setting was "
            "trained once (seed 0), so differences of a few tenths of a point may be random variation."
        )

    summary = "\n\n".join(filter(None, [
        "Validation accuracy after the last epoch (we do not pick the best epoch, because the validation "
        "set is also our test set).",
        table,
        comparison,
        "![Full training runs](results/figures/full_training_runs.png)",
    ]))
    (REPO_ROOT / "results" / "runs" / "full_training_summary.md").write_text(summary + "\n")
    inject_into_readme("FULL_TRAINING", summary)
    print(summary)


if __name__ == "__main__":
    main()
