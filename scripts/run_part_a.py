"""Part A: reproduce the paper's findings on Meta's pretrained DINOv2 models. No training needed.

For each model size (S, B, L, g) we compare the model without registers and the
model trained with 4 registers, on the same images. Claims checked:

    Fig 3   some output patch tokens have a much larger norm than the rest
    Fig 4c  these outliers appear in larger models
    Fig 5a  outliers sit on patches that look like their neighbours (background)
    Fig 7   models with registers have no outlier patch tokens
    Fig 1   with registers, [CLS] attention maps are clean
    Fig 15  with registers, the high norms move into the register tokens

Usage (from the repo root; a GPU is recommended, e.g. Colab T4):
    python -m scripts.run_part_a                    # all sizes, ~15-25 min (mostly downloads)
    python -m scripts.run_part_a --sizes S,B,L      # skip giant if Colab runs out of memory
    python -m scripts.run_part_a --smoke            # random weights + random images: code test only
    python -m scripts.run_part_a --reuse            # redo tables/figures from saved measurements (no models)

How outliers are measured:
    - Token norms are taken at the output of the last transformer block, BEFORE the
      final LayerNorm (the LayerNorm rescales every token and hides the outliers).
    - A token is an outlier if its norm is more than 3x the median norm of the same model.
    - For ViT-g we also report the paper's own hand-picked cutoff of 150.

Outputs: results/figures/part_a_*.png, results/part_a/summary.{json,md},
results/part_a/raw_stats.npz (all measurements). The summary is also copied into README.md.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.analysis import DINOV2_ARCHS, collect_dinov2_stats, load_dinov2, outlier_cutoff
from src.data import MEAN, STD
from src.plots import plot_maps, plot_neighbour_cosine, plot_norm_histograms, plot_outlier_bars
from src.report import inject_into_readme, markdown_table
from src.utils import REPO_ROOT, get_device

FIG_DIR = REPO_ROOT / "results" / "figures"
OUT_DIR = REPO_ROOT / "results" / "part_a"
PATCH_SIZE = 14             # all DINOv2 models use 14x14 patches
PAPER_CUTOFF_VIT_G = 150.0  # the paper's hand-picked outlier cutoff for DINOv2 ViT-g (Sec 2.1)
PAPER_OUTLIER_PCT_VIT_G = 2.37  # % of ViT-g tokens above that cutoff, reported in the paper (Fig 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Artifact analysis on pretrained DINOv2 models.")
    p.add_argument("--sizes", default="S,B,L,g", help="comma-separated, from S,B,L,g")
    p.add_argument("--n-images", type=int, default=256, help="number of images to analyse")
    p.add_argument("--img-size", type=int, default=224, help="must be a multiple of 14")
    p.add_argument("--n-maps", type=int, default=4, help="images shown in the attention / norm-map figures")
    p.add_argument("--cutoff-mult", type=float, default=3.0,
                   help="outlier = norm > this x the median norm of the same model")
    p.add_argument("--data-root", default="data")
    p.add_argument("--smoke", action="store_true", help="random weights + random images (code test only)")
    p.add_argument("--reuse", action="store_true",
                   help="skip the models; reuse measurements saved in results/part_a/raw_stats.npz")
    args = p.parse_args()
    args.sizes = [s.strip() for s in args.sizes.split(",")]
    if any(s not in DINOV2_ARCHS for s in args.sizes):
        p.error(f"--sizes must be from {list(DINOV2_ARCHS)}")
    if args.img_size % PATCH_SIZE != 0:
        p.error("--img-size must be a multiple of 14")
    return args


def get_image_loader(n_images: int, img_size: int, root: str, smoke: bool) -> DataLoader:
    """A fixed set of images, the same for every model.

    We use Imagenette: 10 easy ImageNet classes of object-centred photos,
    similar to the images in the paper's figures.
    """
    preprocess = transforms.Compose([
        transforms.Resize(img_size), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    if smoke:
        dataset = datasets.FakeData(n_images, (3, img_size, img_size), 10, preprocess)
    else:
        already_downloaded = (Path(root) / "imagenette2-320" / "val").is_dir()
        dataset = datasets.Imagenette(root, split="val", size="320px",
                                      download=not already_downloaded, transform=preprocess)
        chosen = np.random.RandomState(0).permutation(len(dataset))[:n_images]
        dataset = Subset(dataset, sorted(chosen.tolist()))
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)


def run_all_models(sizes: list[str], loader: DataLoader, device, n_maps: int, smoke: bool) -> dict:
    """Run every model once. Returns stats[(size, has_registers)]."""
    stats = {}
    for size in sizes:
        for has_registers in (False, True):
            model, name = load_dinov2(size, has_registers, pretrained=not smoke)
            print(f"Running {name} ...", flush=True)
            stats[(size, has_registers)] = collect_dinov2_stats(model.to(device), loader, device, n_maps)
            # Free memory before loading the next model (the giant one needs a lot).
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return stats


def save_stats(stats: dict, path: Path) -> None:
    """Save every measurement so tables and figures can be remade without rerunning the models."""
    arrays = {}
    for (size, has_registers), s in stats.items():
        for key, value in s.items():
            arrays[f"{size}|{int(has_registers)}|{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_stats(path: Path) -> dict:
    stats = {}
    with np.load(path) as data:
        for name in data.files:
            size, has_registers, key = name.split("|")
            value = data[name]
            stats.setdefault((size, bool(int(has_registers))), {})[key] = int(value) if key == "grid_size" else value
    return stats


def summarise(stats: dict, sizes: list[str], cutoff_mult: float) -> tuple[dict, dict]:
    """Turn raw measurements into the numbers we report.

    Returns (table, cutoffs): table[size] holds the numbers for one model size,
    cutoffs[(size, has_registers)] holds each model's own outlier cutoff.
    """
    cutoffs = {key: outlier_cutoff(s["patch_norm"], cutoff_mult) for key, s in stats.items()}
    table = {}
    for size in sizes:
        row = {}
        for has_registers, tag in ((False, "no_reg"), (True, "reg")):
            s = stats[(size, has_registers)]
            norms = s["patch_norm"]
            is_outlier = norms > cutoffs[(size, has_registers)]
            row[f"median_{tag}"] = float(np.median(norms))
            row[f"max_{tag}"] = float(norms.max())
            row[f"cutoff_{tag}"] = cutoffs[(size, has_registers)]
            row[f"pct_outliers_{tag}"] = 100 * float(is_outlier.mean())
            # After the final LayerNorm: how much bigger is the largest token than a typical one?
            after_ln = s["patch_norm_after_ln"]
            row[f"max_over_median_after_ln_{tag}"] = float(after_ln.max() / np.median(after_ln))
            row[f"max_over_median_{tag}"] = float(norms.max() / np.median(norms))

        no_reg = stats[(size, False)]
        is_outlier = no_reg["patch_norm"] > cutoffs[(size, False)]
        row["cosine_outlier"] = float(no_reg["cosine"][is_outlier].mean()) if is_outlier.any() else None
        row["cosine_normal"] = float(no_reg["cosine"][~is_outlier].mean())

        reg = stats[(size, True)]
        row["cls_norm_reg"] = float(reg["cls_norm"].mean())
        row["register_norms_reg"] = [float(v) for v in reg["reg_norm"].mean(axis=0)]
        if size == "g":  # the paper's own cutoff, only defined for ViT-g
            row["pct_above_paper_cutoff_no_reg"] = 100 * float((no_reg["patch_norm"] > PAPER_CUTOFF_VIT_G).mean())
        table[size] = row
    return table, cutoffs


def make_figures(stats: dict, table: dict, cutoffs: dict, sizes: list[str], images: torch.Tensor) -> None:
    largest = sizes[-1]
    plot_norm_histograms(stats, sizes, "patch_norm", "Patch-token norms, before the final LayerNorm (paper Fig 3 / Fig 7)",
                         FIG_DIR / "part_a_norm_histograms.png", cutoffs=cutoffs)
    plot_norm_histograms(stats, sizes, "patch_norm_after_ln",
                         "Patch-token norms AFTER the final LayerNorm (outliers are hidden here)",
                         FIG_DIR / "part_a_norm_histograms_after_layernorm.png")
    plot_outlier_bars(table, sizes, FIG_DIR / "part_a_outliers_by_size.png")
    plot_neighbour_cosine(stats[(largest, False)]["patch_norm"], stats[(largest, False)]["cosine"],
                          cutoffs[(largest, False)], f"DINOv2 ViT-{largest}/14, no registers (paper Fig 5a)",
                          FIG_DIR / "part_a_neighbour_cosine.png")
    plot_maps(stats, sizes, images, "attention", "Last-block [CLS] attention (paper Fig 1)",
              FIG_DIR / "part_a_attention_maps.png")
    plot_maps(stats, sizes, images, "norm", "Patch-token norm maps (paper Fig 21)",
              FIG_DIR / "part_a_norm_maps.png")


def format_summary(table: dict, sizes: list[str], args: argparse.Namespace, num_patches: int) -> str:
    """The markdown text that goes into the README."""
    def fmt(value, digits=1):
        return "n/a" if value is None else f"{value:.{digits}f}"

    setup = (
        f"Setup: official DINOv2 checkpoints (torch.hub), {args.n_images} Imagenette validation images at "
        f"{args.img_size}x{args.img_size} ({num_patches} patches each). Norms are measured on the output of the "
        f"last transformer block, before the final LayerNorm. An outlier is a patch token whose norm is more than "
        f"{args.cutoff_mult:g} x the median norm of the same model."
    )
    if args.smoke:
        setup = "**SMOKE TEST: random weights and random images, numbers are meaningless.**\n\n" + setup

    main_table = markdown_table(
        ["Model", "Median norm (no reg / 4 reg)", "Max norm (no reg / 4 reg)",
         "% outliers, no reg", "% outliers, 4 reg", "Neighbour similarity, outlier vs normal (no reg)"],
        [[f"ViT-{s}/14",
          f"{fmt(t['median_no_reg'])} / {fmt(t['median_reg'])}",
          f"{fmt(t['max_no_reg'])} / {fmt(t['max_reg'])}",
          f"{fmt(t['pct_outliers_no_reg'], 2)}%", f"{fmt(t['pct_outliers_reg'], 2)}%",
          f"{fmt(t['cosine_outlier'], 3)} vs {fmt(t['cosine_normal'], 3)}"]
         for s, t in ((s, table[s]) for s in sizes)],
    )
    parts = [setup, main_table]

    if "g" in sizes:
        t = table["g"]
        parts.append(
            f"With the paper's own cutoff (norm > {PAPER_CUTOFF_VIT_G:g}) on ViT-g: "
            f"**{fmt(t['pct_above_paper_cutoff_no_reg'], 2)}%** of patch tokens without registers "
            f"(paper reports {PAPER_OUTLIER_PCT_VIT_G}%). This cutoff is only meaningful for the no-register model, "
            f"which is what the paper applied it to."
        )

    parts += [
        "Why before the final LayerNorm: the LayerNorm rescales every token to a similar length. "
        "Largest token norm divided by the median norm:",
        markdown_table(
            ["Model", "before LayerNorm (no reg / 4 reg)", "after LayerNorm (no reg / 4 reg)"],
            [[f"ViT-{s}/14",
              f"{fmt(table[s]['max_over_median_no_reg'], 2)}x / {fmt(table[s]['max_over_median_reg'], 2)}x",
              f"{fmt(table[s]['max_over_median_after_ln_no_reg'], 2)}x / "
              f"{fmt(table[s]['max_over_median_after_ln_reg'], 2)}x"] for s in sizes],
        ),
        "Average norms in the register models, before the final LayerNorm (where do the high norms go? cf. paper Fig 15b):",
        markdown_table(
            ["Model", "[CLS]", "reg_0", "reg_1", "reg_2", "reg_3", "median patch"],
            [[f"ViT-{s}/14 + reg", fmt(table[s]["cls_norm_reg"]),
              *[fmt(v) for v in table[s]["register_norms_reg"]], fmt(table[s]["median_reg"])] for s in sizes],
        ),
        "Figures: `results/figures/part_a_*.png`",
    ]
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    device = get_device()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / ("raw_stats_smoke.npz" if args.smoke else "raw_stats.npz")

    loader = get_image_loader(args.n_images, args.img_size, args.data_root, args.smoke)
    images_for_maps = next(iter(loader))[0][: args.n_maps]  # same images every model sees first

    if args.reuse:
        if not raw_path.exists():
            raise SystemExit(f"--reuse given but {raw_path} does not exist; run without --reuse first.")
        stats = load_stats(raw_path)
        missing = [s for s in args.sizes if (s, False) not in stats or (s, True) not in stats]
        if missing:
            raise SystemExit(f"Saved measurements have no results for sizes {missing}; run without --reuse.")
    else:
        stats = run_all_models(args.sizes, loader, device, args.n_maps, args.smoke)
        save_stats(stats, raw_path)

    table, cutoffs = summarise(stats, args.sizes, args.cutoff_mult)
    make_figures(stats, table, cutoffs, args.sizes, images_for_maps)

    num_patches = stats[(args.sizes[0], False)]["grid_size"] ** 2
    summary = format_summary(table, args.sizes, args, num_patches)
    (OUT_DIR / "summary.md").write_text(summary + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps({"settings": vars(args), "results": table}, indent=2))
    if not args.smoke:
        inject_into_readme("PART_A", summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
