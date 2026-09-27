"""Summarise the full training runs of our ViT (registers, augmentation and amount of training data).

Reads the run folders written by src.train, then:
    - copies the logs (not the model weights) into results/runs/ so they go on GitHub
    - plots train loss and validation accuracy per epoch
    - writes a results table and copies it into README.md

Usage (from the repo root):
    python -m scripts.summarise_training --runs RUN_DIR/tiny_reg0 RUN_DIR/tiny_reg4 \
        RUN_DIR/tiny_reg0_strong RUN_DIR/tiny_reg4_strong RUN_DIR/tiny_reg0_strong_50pct

The paper's matching claim (Table 2a): adding registers does not hurt accuracy
(ImageNet top-1 changes by -0.1 to +0.5 points across DeiT-III, OpenCLIP, DINOv2).
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.plots import NO_REG_COLOUR, REG_COLOUR, plot_training_curves
from src.report import inject_into_readme, markdown_table
from src.utils import REPO_ROOT

LOG_FILES = ("config.json", "log.csv", "final.json")
FULL_TRAIN_SIZE = 100_000  # Tiny-ImageNet training images
PAPER_TABLE_2A = "DeiT-III 84.7 → 84.7, OpenCLIP 78.2 → 78.1, DINOv2 84.3 → 84.8"


def load_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    final = json.loads((run_dir / "final.json").read_text())
    with open(run_dir / "log.csv") as f:
        rows = list(csv.DictReader(f))
    val_losses = [float(r["val_loss"]) for r in rows]
    return {
        "name": run_dir.name,
        "registers": config["registers"],
        "aug": config.get("aug", "basic"),  # runs made before --aug existed used basic augmentation
        "train_images": config.get("train_subset") or FULL_TRAIN_SIZE,
        "size": config["size"],
        "params": config["params"],
        "epochs": final["epochs_run"],
        "val_acc": final["val_acc"],
        "train_acc": float(rows[-1]["train_acc"]),
        "val_loss": final["val_loss"],
        "min_val_loss": min(val_losses),
        "minutes": sum(float(r["epoch_time_s"]) for r in rows) / 60,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise full training runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="run folders")
    args = parser.parse_args()

    runs = []
    for run_dir in map(Path, args.runs):
        if not (run_dir / "final.json").exists():
            raise SystemExit(f"{run_dir} has no final.json: the run did not finish.")
        runs.append(load_run(run_dir))
        # Copy the logs into the repo (the model weights stay where they are; too big for GitHub).
        target = REPO_ROOT / "results" / "runs" / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for name in LOG_FILES:
            shutil.copy(run_dir / name, target / name)

    # Graph: colour = registers (orange none, blue 4); line style = augmentation and data
    # (solid basic, dashed strong, dotted strong on part of the data).
    def line_style(r):
        if r["train_images"] < FULL_TRAIN_SIZE:
            return ":"
        return "-" if r["aug"] == "basic" else "--"

    def data_label(r):
        return "" if r["train_images"] == FULL_TRAIN_SIZE else f", {100 * r['train_images'] // FULL_TRAIN_SIZE}% data"

    curves = {
        f"{r['registers']} registers, {r['aug']} aug{data_label(r)}": (
            REPO_ROOT / "results" / "runs" / r["name"],
            NO_REG_COLOUR if r["registers"] == 0 else REG_COLOUR,
            line_style(r),
        )
        for r in runs
    }
    plot_training_curves(curves, REPO_ROOT / "results" / "figures" / "full_training_runs.png",
                         "Full training: ViT-Tiny on Tiny-ImageNet")

    table = markdown_table(
        ["Model", "Augmentation", "Registers", "Train images", "Epochs", "Val accuracy", "Train accuracy",
         "Val loss (final / lowest)", "Time (min)"],
        [[f"ViT-{r['size'].capitalize()}", r["aug"], r["registers"], f"{r['train_images']:,}", r["epochs"],
          f"**{100 * r['val_acc']:.2f}%**", f"{100 * r['train_acc']:.1f}%",
          f"{r['val_loss']:.3f} / {r['min_val_loss']:.3f}", f"{r['minutes']:.0f}"] for r in runs],
    )

    # Compare runs that differ in exactly one thing.
    notes = []
    by_key = {(r["aug"], r["registers"], r["train_images"]): r for r in runs}
    full = FULL_TRAIN_SIZE

    def diff(a, b):  # accuracy difference b - a, in percentage points
        return 100 * (by_key[b]["val_acc"] - by_key[a]["val_acc"])

    for aug in ("basic", "strong"):
        a, b = (aug, 0, full), (aug, 4, full)
        if a in by_key and b in by_key:
            notes.append(f"Effect of adding registers ({aug} augmentation): **{diff(a, b):+.2f} points**.")
    for regs in (0, 4):
        a, b = ("basic", regs, full), ("strong", regs, full)
        if a in by_key and b in by_key:
            notes.append(f"Effect of strong augmentation ({regs} registers): **{diff(a, b):+.2f} points**.")
    for r in runs:
        a, b = (r["aug"], r["registers"], r["train_images"]), (r["aug"], r["registers"], full)
        if r["train_images"] < full and b in by_key:
            notes.append(f"Effect of going from {r['train_images']:,} to {full:,} training images "
                         f"({r['aug']} augmentation, {r['registers']} registers): **{diff(a, b):+.2f} points**.")
    notes.append(
        f"Paper (Table 2a, ImageNet top-1): {PAPER_TABLE_2A}, i.e. registers do not hurt accuracy. "
        "Our absolute accuracies are not comparable with the paper's (different dataset, much smaller "
        "model, shorter training); only the with/without-registers differences are. Each setting was "
        "trained once (seed 0), so differences of a few tenths of a point may be random variation."
    )
    notes.append(
        "How to read the overfitting columns: a large gap between train and validation accuracy, and a final "
        "validation loss well above its lowest value, both mean the model is memorising the training images. "
        "With strong augmentation, train accuracy is measured on mixed (MixUp/CutMix) images, so it is only "
        "roughly comparable with the basic runs; the validation loss columns are directly comparable."
    )

    summary = "\n\n".join([
        "Validation accuracy after the last epoch (we do not pick the best epoch, because the validation "
        "set is also our test set).",
        table,
        "\n".join(f"- {n}" for n in notes),
        "![Full training runs](results/figures/full_training_runs.png)",
    ])
    (REPO_ROOT / "results" / "runs" / "full_training_summary.md").write_text(summary + "\n")
    inject_into_readme("FULL_TRAINING", summary)
    print(summary)


if __name__ == "__main__":
    main()
