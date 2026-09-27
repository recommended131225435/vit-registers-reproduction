"""Stage 5 of the pipeline: turn measurements into figures.

Every function saves one PNG file. Colours are consistent everywhere:
orange = no registers, blue = with registers.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import MEAN, STD

NO_REG_COLOUR, REG_COLOUR = "tab:orange", "tab:blue"


def _save(fig, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def to_displayable(image: torch.Tensor) -> np.ndarray:
    """Undo the normalisation so a (3, H, W) tensor can be shown as a picture."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ----------------------------------------------------------------------------
# Data and training
# ----------------------------------------------------------------------------
def plot_sample_batch(images: torch.Tensor, labels: torch.Tensor, path, title: str) -> None:
    """Grid of the first 16 images of a batch with their labels."""
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    for ax, image, label in zip(axes.flat, images[:16], labels[:16]):
        ax.imshow(to_displayable(image))
        ax.set_title(f"class {int(label)}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    _save(fig, path)


def plot_training_curves(runs: dict, path, title: str) -> None:
    """Train loss and validation accuracy per epoch.

    `runs` maps a label to either a run folder, or a tuple (run folder, colour, line style).
    With exactly two plain folders (no registers first, registers second) the usual colours are used.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.5))
    for i, (label, run) in enumerate(runs.items()):
        if isinstance(run, tuple):
            run_dir, colour, style = run
        else:
            run_dir, style = run, "-"
            colour = [NO_REG_COLOUR, REG_COLOUR][i] if len(runs) == 2 else None
        with open(Path(run_dir) / "log.csv") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        kwargs = dict(color=colour, linestyle=style, marker="o", markersize=3, label=label)
        ax_loss.plot(epochs, [float(r["train_loss"]) for r in rows], **kwargs)
        ax_acc.plot(epochs, [float(r["val_acc"]) for r in rows], **kwargs)
    ax_loss.set(title="train loss", xlabel="epoch")
    ax_acc.set(title="validation accuracy", xlabel="epoch")
    ax_loss.legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, path)


# ----------------------------------------------------------------------------
# Part A: DINOv2 analysis
# ----------------------------------------------------------------------------
def plot_norm_histograms(stats: dict, sizes: list[str], key: str, title: str, path,
                         cutoffs: dict | None = None) -> None:
    """Paper Fig 3 / Fig 7: distribution of patch-token norms, without vs with registers.

    stats[(size, has_registers)][key] holds the norms. The y-axis is logarithmic
    so that the few outliers stay visible next to the many normal tokens.
    cutoffs[(size, has_registers)], if given, is drawn as a dashed line per model.
    """
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.4), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        no_reg, reg = stats[(size, False)][key].ravel(), stats[(size, True)][key].ravel()
        bins = np.linspace(0, max(no_reg.max(), reg.max()) * 1.02, 80)
        ax.hist(no_reg, bins=bins, density=True, alpha=0.6, color=NO_REG_COLOUR, label="no registers")
        ax.hist(reg, bins=bins, density=True, alpha=0.6, color=REG_COLOUR, label="4 registers")
        if cutoffs is not None:
            ax.axvline(cutoffs[(size, False)], ls="--", c=NO_REG_COLOUR, lw=1.2, label="cutoff, no reg")
            ax.axvline(cutoffs[(size, True)], ls="--", c=REG_COLOUR, lw=1.2, label="cutoff, 4 reg")
        ax.set(yscale="log", title=f"DINOv2 ViT-{size}/14", xlabel="L2 norm of patch token")
    axes[0][0].set_ylabel("density (log scale)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title)
    _save(fig, path)


def plot_outlier_bars(table: dict, sizes: list[str], path) -> None:
    """Paper Fig 4c + Fig 7: percentage of outlier tokens for each model size."""
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(x - 0.2, [table[s]["pct_outliers_no_reg"] for s in sizes], 0.4, color=NO_REG_COLOUR, label="no registers")
    ax.bar(x + 0.2, [table[s]["pct_outliers_reg"] for s in sizes], 0.4, color=REG_COLOUR, label="4 registers")
    ax.set_xticks(x, [f"ViT-{s}" for s in sizes])
    ax.set(ylabel="% of patch tokens that are outliers", title="High-norm outlier tokens by model size")
    ax.legend()
    _save(fig, path)


def plot_neighbour_cosine(norms: np.ndarray, cosine: np.ndarray, cutoff: float, title: str, path) -> None:
    """Paper Fig 5a: are outlier patches more similar to their neighbours than normal patches?"""
    is_outlier = norms.ravel() > cutoff
    cosine = cosine.ravel()
    bins = np.linspace(min(cosine.min(), 0.0), 1.0, 60)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(cosine[~is_outlier], bins=bins, density=True, histtype="step", lw=2,
            label=f"normal patches ({(~is_outlier).sum():,})")
    if is_outlier.any():
        ax.hist(cosine[is_outlier], bins=bins, density=True, histtype="step", lw=2,
                label=f"outlier patches ({is_outlier.sum():,})")
    ax.set(xlabel="cosine similarity to 4 neighbours (after patch embedding)", ylabel="density", title=title)
    ax.legend()
    _save(fig, path)


def plot_maps(stats: dict, sizes: list[str], images: torch.Tensor, kind: str, title: str, path) -> None:
    """Paper Fig 1 / Fig 21: one row per image, one column per model (no reg, reg for each size).

    kind = "attention" shows [CLS] attention maps; kind = "norm" shows patch-token norms
    (before the final LayerNorm).
    """
    if kind not in ("attention", "norm"):
        raise ValueError("kind must be 'attention' or 'norm'")
    num_images, num_cols = len(images), 1 + 2 * len(sizes)
    fig, axes = plt.subplots(num_images, num_cols, figsize=(1.9 * num_cols, 1.9 * num_images + 0.6),
                             squeeze=False)
    for row in range(num_images):
        axes[row][0].imshow(to_displayable(images[row]))
        col = 1
        for size in sizes:
            for has_registers in (False, True):
                s = stats[(size, has_registers)]
                grid = s["grid_size"]
                heatmap = s["attn_maps"][row] if kind == "attention" else s["patch_norm"][row].reshape(grid, grid)
                axes[row][col].imshow(heatmap, cmap="viridis")
                if row == 0:
                    axes[row][col].set_title(f"ViT-{size}\n{'4 reg' if has_registers else 'no reg'}", fontsize=8)
                col += 1
    axes[0][0].set_title("input", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title)
    _save(fig, path)
