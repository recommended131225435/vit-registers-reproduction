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

Outputs: results/figures/part_a_*.png, results/part_a/summary.{json,md}.
The summary is also copied into README.md.
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
PATCH_SIZE = 14  # all DINOv2 models use 14x14 patches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Artifact analysis on pretrained DINOv2 models.")
    p.add_argument("--sizes", default="S,B,L,g", help="comma-separated, from S,B,L,g")
    p.add_argument("--n-images", type=int, default=256, help="number of images to analyse")
    p.add_argument("--img-size", type=int, default=224, help="must be a multiple of 14")
    p.add_argument("--n-maps", type=int, default=4, help="images shown in the attention / norm-map figures")
    p.add_argument("--cutoff-mult", type=float, default=3.0,
                   help="outlier = norm > this x median norm of the no-register model")
    p.add_argument("--data-root", default="data")
    p.add_argument("--smoke", action="store_true", help="random weights + random images (code test only)")
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


def summarise(stats: dict, sizes: list[str], cutoff_mult: float) -> dict:
    """Turn raw measurements into the numbers we report, one entry per model size.

    The cutoff is computed on the no-register model and applied to both
    versions, so both are judged by the same standard.
    """
    table = {}
    for size in sizes:
        no_reg, reg = stats[(size, False)], stats[(size, True)]
        cutoff = outlier_cutoff(no_reg["norm_post"], cutoff_mult)
        cutoff_pre = outlier_cutoff(no_reg["norm_pre"], cutoff_mult)
        is_outlier = no_reg["norm_post"] > cutoff
        table[size] = {
            "median_norm_no_reg": float(np.median(no_reg["norm_post"])),
            "max_norm_no_reg": float(no_reg["norm_post"].max()),
            "max_norm_reg": float(reg["norm_post"].max()),
            "cutoff": cutoff,
            "cutoff_prenorm": cutoff_pre,
            "pct_outliers_no_reg": 100 * float(is_outlier.mean()),
            "pct_outliers_reg": 100 * float((reg["norm_post"] > cutoff).mean()),
            "pct_outliers_no_reg_prenorm": 100 * float((no_reg["norm_pre"] > cutoff_pre).mean()),
            "pct_outliers_reg_prenorm": 100 * float((reg["norm_pre"] > cutoff_pre).mean()),
            "cosine_outlier": float(no_reg["cosine"][is_outlier].mean()) if is_outlier.any() else None,
            "cosine_normal": float(no_reg["cosine"][~is_outlier].mean()),
            "cls_norm_reg": float(reg["cls_norm"].mean()),
            "register_norms_reg": [float(v) for v in reg["reg_norm"].mean(axis=0)],
            "median_patch_norm_reg": float(np.median(reg["norm_post"])),
        }
    return table


def make_figures(stats: dict, table: dict, sizes: list[str], images: torch.Tensor) -> None:
    largest = sizes[-1]
    cutoffs = {s: table[s]["cutoff"] for s in sizes}
    cutoffs_pre = {s: table[s]["cutoff_prenorm"] for s in sizes}
    plot_norm_histograms(stats, sizes, "norm_post", cutoffs,
                         "Output patch-token norms (paper Fig 3 / Fig 7)", FIG_DIR / "part_a_norm_histograms.png")
    plot_norm_histograms(stats, sizes, "norm_pre", cutoffs_pre,
                         "Patch-token norms before the final LayerNorm",
                         FIG_DIR / "part_a_norm_histograms_prenorm.png")
    plot_outlier_bars(table, sizes, FIG_DIR / "part_a_outliers_by_size.png")
    plot_neighbour_cosine(stats[(largest, False)]["norm_post"], stats[(largest, False)]["cosine"],
                          cutoffs[largest], f"DINOv2 ViT-{largest}/14, no registers (paper Fig 5a)",
                          FIG_DIR / "part_a_neighbour_cosine.png")
    plot_maps(stats, sizes, images, "attention", "Last-block [CLS] attention (paper Fig 1)",
              FIG_DIR / "part_a_attention_maps.png")
    plot_maps(stats, sizes, images, "norm", "Output patch-token norm maps (paper Fig 21)",
              FIG_DIR / "part_a_norm_maps.png")


def format_summary(table: dict, sizes: list[str], args: argparse.Namespace, num_patches: int) -> str:
    """The markdown text that goes into the README."""
    def fmt(value, digits=1):
        return "n/a" if value is None else f"{value:.{digits}f}"

    setup = (
        f"Setup: official DINOv2 checkpoints (torch.hub), {args.n_images} Imagenette validation images at "
        f"{args.img_size}x{args.img_size} ({num_patches} patches each). An outlier is an output patch token "
        f"whose norm is more than {args.cutoff_mult:g} x the median norm of the no-register model of the "
        f"same size; the same cutoff is used for its register version. (The paper used a hand-picked 150 for ViT-g.)"
    )
    if args.smoke:
        setup = "**SMOKE TEST: random weights and random images, numbers are meaningless.**\n\n" + setup

    main_table = markdown_table(
        ["Model", "Median norm", "Cutoff", "Max norm (no reg / reg)", "% outliers, no reg",
         "% outliers, 4 reg", "Neighbour similarity, outlier vs normal"],
        [[f"ViT-{s}/14", fmt(t["median_norm_no_reg"]), fmt(t["cutoff"]),
          f"{fmt(t['max_norm_no_reg'])} / {fmt(t['max_norm_reg'])}",
          f"{fmt(t['pct_outliers_no_reg'], 2)}%", f"{fmt(t['pct_outliers_reg'], 2)}%",
          f"{fmt(t['cosine_outlier'], 3)} vs {fmt(t['cosine_normal'], 3)}"]
         for s, t in ((s, table[s]) for s in sizes)],
    )
    prenorm_table = markdown_table(
        ["Model", "% outliers, no reg", "% outliers, 4 reg"],
        [[f"ViT-{s}/14", f"{fmt(table[s]['pct_outliers_no_reg_prenorm'], 2)}%",
          f"{fmt(table[s]['pct_outliers_reg_prenorm'], 2)}%"] for s in sizes],
    )
    register_table = markdown_table(
        ["Model", "[CLS]", "reg_0", "reg_1", "reg_2", "reg_3", "median patch"],
        [[f"ViT-{s}/14 + reg", fmt(table[s]["cls_norm_reg"]),
          *[fmt(v) for v in table[s]["register_norms_reg"]], fmt(table[s]["median_patch_norm_reg"])]
         for s in sizes],
    )
    return "\n\n".join([
        setup,
        main_table,
        "Same rule applied to the norms *before* the final LayerNorm:",
        prenorm_table,
        "Average output norms in the register models (where do the high norms go? cf. paper Fig 15b):",
        register_table,
        "Figures: `results/figures/part_a_*.png`",
    ])


def main() -> None:
    args = parse_args()
    device = get_device()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loader = get_image_loader(args.n_images, args.img_size, args.data_root, args.smoke)
    images_for_maps = next(iter(loader))[0][: args.n_maps]  # same images every model sees first

    stats = run_all_models(args.sizes, loader, device, args.n_maps, args.smoke)
    table = summarise(stats, args.sizes, args.cutoff_mult)
    make_figures(stats, table, args.sizes, images_for_maps)

    num_patches = stats[(args.sizes[0], False)]["grid_size"] ** 2
    summary = format_summary(table, args.sizes, args, num_patches)
    (OUT_DIR / "summary.md").write_text(summary + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps({"settings": vars(args), "results": table}, indent=2))
    if not args.smoke:
        inject_into_readme("PART_A", summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
