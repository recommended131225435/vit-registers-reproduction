"""Experiment, step 3: combine all runs into tables and figures.

Question: do high-norm artifacts appear in small ViTs trained from scratch, or only
once models are large enough? And do registers do anything below that size?

Reads results/experiment/<run>/stats.json and arrays.npz (written by
scripts.analyse_run) for our own trained models, plus results/part_a/summary.json
for Meta's DINOv2 models (Part A), and puts them on one model-size axis.

Usage (from the repo root):
    python -m scripts.summarise_experiment

Writes results/figures/experiment_*.png and results/experiment/summary.md,
and copies the summary into README.md.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plots import NO_REG_COLOUR, REG_COLOUR, _save
from src.report import inject_into_readme, markdown_table
from src.utils import REPO_ROOT

EXP_DIR = REPO_ROOT / "results" / "experiment"
FIG_DIR = REPO_ROOT / "results" / "figures"
PART_A = REPO_ROOT / "results" / "part_a" / "summary.json"
SIZE_ORDER = ["tiny", "small", "base"]
# Parameter counts of Meta's DINOv2 backbones (DINOv2 paper, Table 16; patch size 14).
DINOV2_PARAMS = {"S": 21e6, "B": 86e6, "L": 300e6, "g": 1100e6}


def load_runs() -> list[dict]:
    runs = []
    for stats_path in sorted(EXP_DIR.glob("*/stats.json")):
        stats = json.loads(stats_path.read_text())
        if stats.get("fake"):
            continue
        arrays = np.load(stats_path.parent / "arrays.npz")
        stats["arrays"] = {k: arrays[k] for k in arrays.files}
        runs.append(stats)
    runs.sort(key=lambda r: (SIZE_ORDER.index(r["size"]), r["registers"]))
    if not runs:
        raise SystemExit(f"No analysed runs in {EXP_DIR}; run scripts.analyse_run first.")
    return runs


def label(r: dict) -> str:
    return f"ViT-{r['size'].capitalize()}, {r['registers']} reg"


def colour(registers: int) -> str:
    return NO_REG_COLOUR if registers == 0 else REG_COLOUR


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_scaling(runs: list[dict], part_a: dict | None, path: Path) -> None:
    """Main figure: artifacts against model size, our models and DINOv2 together."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, part_a_key, title in [
        (axes[0], "pct_outliers", "pct_outliers", "% of patch tokens that are outliers"),
        (axes[1], "max_over_median", "max_over_median", "largest token norm / median norm"),
    ]:
        for regs, tag in ((0, "no_reg"), (4, "reg")):
            ours = [r for r in runs if r["registers"] == regs]
            ax.plot([r["params"] for r in ours], [r["last_block"][key] for r in ours], "o-",
                    color=colour(regs), label=f"our ViTs (from scratch), {regs} reg")
            if part_a:
                sizes = [s for s in DINOV2_PARAMS if s in part_a]
                ax.plot([DINOV2_PARAMS[s] for s in sizes], [part_a[s][f"{part_a_key}_{tag}"] for s in sizes],
                        "^--", color=colour(regs), label=f"DINOv2 (Meta), {regs} reg")
                if regs == 0:
                    for s in sizes:
                        ax.annotate(f"DINOv2-{s}", (DINOV2_PARAMS[s], part_a[s][f"{part_a_key}_no_reg"]),
                                    textcoords="offset points", xytext=(4, 4), fontsize=7)
        for r in runs:
            if r["registers"] == 0:
                ax.annotate(f"ours-{r['size'].capitalize()}", (r["params"], r["last_block"][key]),
                            textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.set(xscale="log", xlabel="parameters (log scale)", title=title)
    axes[1].axhline(3.0, ls=":", c="grey", lw=1)
    axes[1].text(axes[1].get_xlim()[0], 3.05, " outlier cutoff (3x median)", fontsize=7, color="grey")
    axes[0].legend(fontsize=7)
    fig.suptitle("Do high-norm artifacts appear with model size? (norms before the final LayerNorm)")
    _save(fig, path)


def plot_layers(runs: list[dict], path: Path) -> None:
    """Paper Fig 4a analogue: how the largest/median norm develops through the blocks."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    styles = {"tiny": "-", "small": "--", "base": ":"}
    for r in runs:
        values = [layer["max_over_median"] for layer in r["layers"]]
        ax.plot(range(1, len(values) + 1), values, styles[r["size"]], marker="o", markersize=3,
                color=colour(r["registers"]), label=label(r))
    ax.axhline(3.0, ls=":", c="grey", lw=1)
    ax.set(xlabel="block", ylabel="largest / median patch-token norm",
           title="Patch-token norms through the network (cf. paper Fig 4a)")
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_histograms(runs: list[dict], path: Path) -> None:
    sizes = [s for s in SIZE_ORDER if any(r["size"] == s for r in runs)]
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.4), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        group = [r for r in runs if r["size"] == size]
        top = max(r["arrays"]["patch_norm"].astype(np.float32).max() for r in group)
        bins = np.linspace(0, top * 1.02, 80)
        for r in group:
            ax.hist(r["arrays"]["patch_norm"].astype(np.float32).ravel(), bins=bins, density=True, alpha=0.6,
                    color=colour(r["registers"]), label=f"{r['registers']} registers")
        ax.set(yscale="log", title=f"our ViT-{size.capitalize()}", xlabel="L2 norm of patch token")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("density (log scale)")
    fig.suptitle("Patch-token norms of our trained models (last block, before the final LayerNorm)")
    _save(fig, path)


def plot_attention(runs: list[dict], path: Path) -> None:
    images = runs[0]["arrays"]["map_images"]
    n = len(images)
    fig, axes = plt.subplots(n, 1 + len(runs), figsize=(1.8 * (1 + len(runs)), 1.8 * n + 0.6), squeeze=False)
    for row in range(n):
        axes[row][0].imshow(images[row])
        for col, r in enumerate(runs, start=1):
            axes[row][col].imshow(r["arrays"]["attn_maps"][row], cmap="viridis")
            if row == 0:
                axes[row][col].set_title(f"ViT-{r['size'].capitalize()}\n{r['registers']} reg", fontsize=8)
    axes[0][0].set_title("input", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Last-block [CLS] attention of our trained models")
    _save(fig, path)


def plot_accuracy_and_registers(runs: list[dict], path: Path) -> None:
    sizes = [s for s in SIZE_ORDER if any(r["size"] == s for r in runs)]
    by_key = {(r["size"], r["registers"]): r for r in runs}
    x = np.arange(len(sizes))
    fig, (ax_acc, ax_reg) = plt.subplots(1, 2, figsize=(10, 3.6))
    for offset, regs in ((-0.2, 0), (0.2, 4)):
        accs = [100 * by_key[(s, regs)]["val_acc"] if (s, regs) in by_key and by_key[(s, regs)]["val_acc"]
                else np.nan for s in sizes]  # missing runs are left out of the plot
        bars = ax_acc.bar(x + offset, accs, 0.4, color=colour(regs), label=f"{regs} registers")
        ax_acc.bar_label(bars, fmt="%.1f", fontsize=7)
    ax_acc.set_xticks(x, [f"ViT-{s.capitalize()}" for s in sizes])
    ax_acc.set(ylabel="validation accuracy (%)", title="Tiny-ImageNet accuracy")
    ax_acc.legend(fontsize=7)

    reg_runs = [r for r in runs if r["registers"] > 0]
    xr = np.arange(len(reg_runs))
    ax_reg.bar(xr - 0.2, [100 * r["attn_patch_to_reg"] for r in reg_runs], 0.4, color=REG_COLOUR,
               label="patches → registers")
    ax_reg.bar(xr + 0.2, [100 * r["attn_cls_to_reg"] for r in reg_runs], 0.4, color="tab:green",
               label="[CLS] → registers")
    for i, r in enumerate(reg_runs):
        ax_reg.hlines(100 * r["attn_uniform_share"], i - 0.45, i + 0.45, colors="k", linestyles=":")
    ax_reg.set_xticks(xr, [f"ViT-{r['size'].capitalize()}" for r in reg_runs])
    ax_reg.set(ylabel="% of attention (last block)", title="How much the registers are used (dotted: uniform)")
    ax_reg.legend(fontsize=7)
    _save(fig, path)


# ----------------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------------
def format_summary(runs: list[dict], part_a: dict | None) -> str:
    def pct(v, d=2):
        return "n/a" if v is None else f"{100 * v:.{d}f}%"

    ours = markdown_table(
        ["Model", "Parameters", "Registers", "Val accuracy", "Median norm", "Largest / median",
         "% outliers", "Peak largest/median over blocks", "Patch attention to registers (uniform)"],
        [[f"ViT-{r['size'].capitalize()}", f"{r['params'] / 1e6:.1f}M", r["registers"], pct(r["val_acc"]),
          f"{r['last_block']['median']:.1f}", f"{r['last_block']['max_over_median']:.2f}x",
          f"{r['last_block']['pct_outliers']:.2f}%",
          f"{max(layer['max_over_median'] for layer in r['layers']):.2f}x",
          "n/a" if not r["registers"] else f"{pct(r['attn_patch_to_reg'], 1)} ({pct(r['attn_uniform_share'], 1)})"]
         for r in runs],
    )
    parts = [
        f"Our models: analysed on the same {runs[0]['n_images']:,} Tiny-ImageNet validation images. Norms are "
        "taken before the final LayerNorm; an outlier is a patch token whose norm is more than "
        f"{runs[0]['cutoff_mult']:g}x the median norm of the same model (same rule as Part A).",
        ours,
    ]
    if part_a:
        parts += [
            "Meta's DINOv2 models (from Part A), on the same axis:",
            markdown_table(
                ["Model", "Parameters", "% outliers (no reg / 4 reg)", "Largest / median (no reg / 4 reg)"],
                [[f"DINOv2 ViT-{s}/14", f"≈{DINOV2_PARAMS[s] / 1e6:.0f}M",
                  f"{part_a[s]['pct_outliers_no_reg']:.2f}% / {part_a[s]['pct_outliers_reg']:.2f}%",
                  f"{part_a[s]['max_over_median_no_reg']:.2f}x / {part_a[s]['max_over_median_reg']:.2f}x"]
                 for s in DINOV2_PARAMS if s in part_a],
            ),
        ]
    parts += [
        "![Scaling](results/figures/experiment_scaling.png)",
        "![Norms through the blocks](results/figures/experiment_layer_norms.png)",
        "![Norm histograms](results/figures/experiment_norm_histograms.png)",
        "![Accuracy and register use](results/figures/experiment_accuracy_registers.png)",
        "![Attention maps](results/figures/experiment_attention_maps.png)",
    ]
    return "\n\n".join(parts)


def main() -> None:
    runs = load_runs()
    part_a = json.loads(PART_A.read_text())["results"] if PART_A.exists() else None
    plot_scaling(runs, part_a, FIG_DIR / "experiment_scaling.png")
    plot_layers(runs, FIG_DIR / "experiment_layer_norms.png")
    plot_histograms(runs, FIG_DIR / "experiment_norm_histograms.png")
    plot_attention(runs, FIG_DIR / "experiment_attention_maps.png")
    plot_accuracy_and_registers(runs, FIG_DIR / "experiment_accuracy_registers.png")
    summary = format_summary(runs, part_a)
    (EXP_DIR / "summary.md").write_text(summary + "\n")
    inject_into_readme("EXPERIMENT", summary)
    print(summary)


if __name__ == "__main__":
    main()
