"""Experiment, step 2: measure artifacts in one of OUR trained ViTs.

Loads a finished training run (the folder written by src.train), runs the trained
model over a fixed set of Tiny-ImageNet validation images, and measures the same
things as Part A: patch-token norms (after every block), outlier tokens, register
norms, how much attention goes to the registers, and [CLS] attention maps.

Usage (from the repo root):
    python -m scripts.analyse_run --run RUN_DIR/small_reg0
    python -m scripts.analyse_run --run some/folder --fake      # random weights + images: code test only

Writes (small files, safe for GitHub):
    results/experiment/<run name>/stats.json    all summary numbers
    results/experiment/<run name>/arrays.npz    norms, attention maps and the images used
It also copies the run's logs (config.json, log.csv, final.json) into results/runs/<run name>/
if they are not there yet. The model weights are NOT copied (too big for GitHub).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.analysis import collect_own_vit_stats, summarise_norms
from src.data import IMG_SIZE, MEAN, NUM_CLASSES, STD, TinyImageNet, download_tiny_imagenet, get_transforms
from src.utils import REPO_ROOT, count_params, get_device
from src.vit import build_vit

LOG_FILES = ("config.json", "log.csv", "final.json")
ANALYSIS_SEED = 0  # the same validation images are used for every run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure high-norm artifacts in one of our trained ViTs.")
    p.add_argument("--run", required=True, help="run folder with config.json, final.json and checkpoint.pt")
    p.add_argument("--out", default=None, help="default: results/experiment/<run folder name>")
    p.add_argument("--n-images", type=int, default=2000, help="validation images to analyse")
    p.add_argument("--n-maps", type=int, default=6, help="images kept for the attention-map figure")
    p.add_argument("--cutoff-mult", type=float, default=3.0, help="outlier = norm > this x median norm")
    p.add_argument("--data-root", default="data")
    p.add_argument("--fake", action="store_true", help="random weights + random images (code test only)")
    return p.parse_args()


def load_trained_model(run_dir: Path, fake: bool, device: torch.device):
    """Rebuild the model from the run's settings and load its trained weights."""
    config = json.loads((run_dir / "config.json").read_text())
    model = build_vit(config["size"], num_registers=config["registers"], num_classes=NUM_CLASSES,
                      drop_path_rate=config.get("drop_path", 0.1))
    if not fake:
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config


def get_analysis_images(n_images: int, root: str, fake: bool) -> DataLoader:
    """A fixed random subset of the validation set (the same images for every run)."""
    if fake:
        dataset = datasets.FakeData(n_images, (3, IMG_SIZE, IMG_SIZE), NUM_CLASSES, get_transforms(train=False))
    else:
        dataset = TinyImageNet(download_tiny_imagenet(root), "val", get_transforms(train=False))
        chosen = np.random.RandomState(ANALYSIS_SEED).permutation(len(dataset))[:n_images]
        dataset = torch.utils.data.Subset(dataset, sorted(chosen.tolist()))
    return DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)


def to_uint8_images(images: torch.Tensor) -> np.ndarray:
    """Undo the normalisation, so the images can be shown next to their attention maps."""
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    pixels = (images.cpu() * std + mean).clamp(0, 1).permute(0, 2, 3, 1)
    return (pixels.numpy() * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else REPO_ROOT / "results" / "experiment" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    model, config = load_trained_model(run_dir, args.fake, device)
    loader = get_analysis_images(args.n_images, args.data_root, args.fake)
    map_images = next(iter(loader))[0][: args.n_maps]
    print(f"Analysing {run_dir.name}: {config['size']} ViT, {config['registers']} registers, "
          f"{len(loader.dataset):,} images, device {device}")

    stats = collect_own_vit_stats(model, loader, device, args.n_maps, args.cutoff_mult)

    final_path = run_dir / "final.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    R = config["registers"]
    summary = {
        "run": run_dir.name,
        "size": config["size"],
        "registers": R,
        "params": count_params(model),
        "val_acc": final.get("val_acc"),
        "epochs": final.get("epochs_run"),
        "n_images": len(loader.dataset),
        "cutoff_mult": args.cutoff_mult,
        "fake": args.fake,
        "last_block": summarise_norms(stats["patch_norm"], args.cutoff_mult),
        "layers": stats["layer_summary"],
        "cls_norm": float(stats["cls_norm"].mean()),
        "register_norms": [float(v) for v in stats["reg_norm"].mean(axis=0)] if R else [],
        "attn_patch_to_reg": float(stats["attn_patch_to_reg"].mean()) if R else None,
        "attn_cls_to_reg": float(stats["attn_cls_to_reg"].mean()) if R else None,
        "attn_uniform_share": R / stats["num_tokens"],  # share the registers would get if attention were uniform
    }
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "arrays.npz",
        patch_norm=stats["patch_norm"].astype(np.float16),
        cosine=stats["cosine"].astype(np.float16),
        attn_maps=stats["attn_maps"].astype(np.float32),
        map_images=to_uint8_images(map_images),
    )

    # Keep the training logs next to the results (never overwrite logs already in the repo).
    log_target = REPO_ROOT / "results" / "runs" / run_dir.name
    if not args.fake and not (log_target / "final.json").exists():
        log_target.mkdir(parents=True, exist_ok=True)
        for name in LOG_FILES:
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, log_target / name)

    lb = summary["last_block"]
    print(f"median norm {lb['median']:.1f} | largest/median {lb['max_over_median']:.2f}x | "
          f"outliers {lb['pct_outliers']:.2f}%"
          + (f" | patch attention to registers {100 * summary['attn_patch_to_reg']:.1f}% "
             f"(uniform would be {100 * summary['attn_uniform_share']:.1f}%)" if R else ""))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
