"""Export a trained ViT for the deployment app.

Takes a training run folder (from src.train) and saves a small file with only what
the app needs: the model weights (stored in half precision to halve the size), the
model settings, and the 200 Tiny-ImageNet class names.

Usage (from the repo root):
    python -m scripts.export_model --run RUN_DIR/tiny_reg0 --out app/weights/vit_tiny_reg0.pt
    python -m scripts.export_model --run some/folder --out /tmp/x.pt --fake   # random weights: code test only
"""

import argparse
import json
from pathlib import Path

import torch

from src.data import NUM_CLASSES, download_tiny_imagenet
from src.vit import build_vit


def class_names(data_root: str) -> list[str]:
    """Readable names for the 200 classes, in the same order as the training labels.

    wnids.txt lists the 200 class IDs (we sort them, exactly like src/data.py does);
    words.txt maps every ImageNet ID to names such as "goldfish, Carassius auratus".
    We keep the first name.
    """
    path = download_tiny_imagenet(data_root)
    ids = sorted(line.strip() for line in (path / "wnids.txt").read_text().splitlines() if line.strip())
    words = {}
    for line in (path / "words.txt").read_text().splitlines():
        if "\t" in line:
            wnid, names = line.split("\t", 1)
            words[wnid] = names.split(",")[0].strip()
    return [words.get(i, i) for i in ids]


def main() -> None:
    p = argparse.ArgumentParser(description="Export a trained ViT for the deployment app.")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--fake", action="store_true", help="random weights and dummy class names (code test only)")
    args = p.parse_args()

    run_dir = Path(args.run)
    config = json.loads((run_dir / "config.json").read_text())
    if args.fake:
        weights = build_vit(config["size"], num_registers=config["registers"]).state_dict()
        names = [f"class {i}" for i in range(NUM_CLASSES)]
    else:
        weights = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
        names = class_names(args.data_root)
    final_path = run_dir / "final.json"
    export = {
        "model": {k: v.half() if v.is_floating_point() else v for k, v in weights.items()},
        "size": config["size"],
        "registers": config["registers"],
        "classes": names,
        "val_acc": json.loads(final_path.read_text())["val_acc"] if final_path.exists() else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
