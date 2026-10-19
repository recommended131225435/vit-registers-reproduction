"""The models behind the app, and what the app computes for an uploaded image.

Two kinds of model are used:

1. OUR ViT-Tiny classifiers (trained in this project on Tiny-ImageNet, without and with
   4 registers). They give the prediction: the top-5 of the 200 Tiny-ImageNet classes.
   Weights: app/weights/vit_tiny_reg0.pt and vit_tiny_reg4.pt (made by scripts.export_model).

2. Meta's pretrained DINOv2 ViT-B/14, without and with registers. Our small models have no
   artifacts to show (see the experiment), so these large models are used to show the
   paper's effect: the [CLS] attention map and the patch-token norm map, side by side.

Settings (environment variables):
    DINOV2_SIZE   S, B (default), L or G. Strong norm outliers only appear in G (see Part A); G needs
                  about 9 GB of RAM and a 9 GB download for the two models.
    DINOV2_REPO   where torch.hub gets the DINOv2 code (default: the official GitHub repo)
    APP_FAKE=1    random weights and no downloads; only for testing the code
"""

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.analysis import cls_attention_map
from src.data import MEAN, NUM_CLASSES, STD
from src.vit import build_vit

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DINOV2_ARCHS = {"S": "vits14", "B": "vitb14", "L": "vitl14", "G": "vitg14"}
OUTLIER_MULT = 3.0  # same outlier rule as the rest of the project: norm > 3x the median norm

CLASSIFIER_INPUT = transforms.Compose([   # exactly the validation preprocessing used in training
    transforms.Resize(64), transforms.CenterCrop(64),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
DINOV2_INPUT = transforms.Compose([       # 224x224 -> 16x16 patches of 14x14 pixels
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


@dataclass
class Classifier:
    name: str
    model: nn.Module
    classes: list[str]
    val_acc: float | None


@dataclass
class Explainer:
    name: str
    model: nn.Module
    num_registers: int


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def is_fake() -> bool:
    return os.environ.get("APP_FAKE") == "1"


def load_classifier(registers: int) -> Classifier:
    """Our ViT-Tiny, trained with `registers` registers."""
    name = f"Our ViT-Tiny, {registers} registers"
    if is_fake():
        model = build_vit("tiny", num_registers=registers, num_classes=NUM_CLASSES).eval()
        return Classifier(name, model, [f"class {i}" for i in range(NUM_CLASSES)], None)
    path = WEIGHTS_DIR / f"vit_tiny_reg{registers}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; create it with scripts.export_model (see README).")
    export = torch.load(path, map_location="cpu", weights_only=False)
    model = build_vit(export["size"], num_registers=export["registers"], num_classes=len(export["classes"]))
    model.load_state_dict({k: v.float() for k, v in export["model"].items()})
    return Classifier(name, model.eval(), export["classes"], export.get("val_acc"))


def load_explainer(registers: bool) -> Explainer:
    """Meta's DINOv2 (size from DINOV2_SIZE), without or with 4 registers."""
    size = os.environ.get("DINOV2_SIZE", "B").upper()
    name = f"DINOv2 ViT-{size}/14, {4 if registers else 0} registers"
    if is_fake():  # stand-in with the same image size and patch size, random weights, no download
        model = build_vit("tiny", num_registers=4 if registers else 0, img_size=224, patch_size=14).eval()
        return Explainer(name, model, model.num_registers)
    hub_repo = os.environ.get("DINOV2_REPO", "facebookresearch/dinov2")
    model = torch.hub.load(
        hub_repo, f"dinov2_{DINOV2_ARCHS[size]}" + ("_reg" if registers else ""),
        source="local" if Path(hub_repo).is_dir() else "github",
        pretrained=os.environ.get("APP_DINOV2_PRETRAINED", "1") == "1",
        **({} if Path(hub_repo).is_dir() else {"trust_repo": True, "skip_validation": True}),
    )
    return Explainer(name, model.eval(), model.num_register_tokens)


def load_all() -> dict:
    """Load every model once, when the app starts."""
    return {
        "classifiers": [load_classifier(0), load_classifier(4)],
        "explainers": [load_explainer(False), load_explainer(True)],
    }


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
@torch.no_grad()
def classify(clf: Classifier, image: Image.Image, top_k: int = 5) -> dict:
    probs = clf.model(CLASSIFIER_INPUT(image).unsqueeze(0)).softmax(dim=-1)[0]
    values, indices = probs.topk(top_k)
    return {
        "model": clf.name,
        "validation_accuracy": clf.val_acc,
        "top5": [{"class": clf.classes[i], "probability": round(float(p), 4)} for p, i in zip(values, indices)],
    }


@torch.no_grad()
def explain(exp: Explainer, image: Image.Image) -> dict:
    """[CLS] attention map, patch-token norm map and artifact numbers for one model."""
    x = DINOV2_INPUT(image).unsqueeze(0)
    grid = x.shape[-1] // 14
    captured = {}
    last = exp.model.blocks[-1]
    hooks = [
        last.attn.register_forward_pre_hook(lambda m, args: captured.update(attn_in=args[0])),
        # DINOv2 blocks return a tensor; ours return (tokens, attention).
        last.register_forward_hook(lambda m, args, out: captured.update(out=out[0] if isinstance(out, tuple) else out)),
    ]
    try:
        exp.model(x)
    finally:
        for h in hooks:
            h.remove()

    R = exp.num_registers
    attention = cls_attention_map(last.attn, captured["attn_in"], R, grid)[0].numpy()
    norms = captured["out"][0, 1 + R:].norm(dim=-1).reshape(grid, grid).numpy()  # before the final LayerNorm
    median = float(np.median(norms))
    return {
        "model": exp.name,
        "registers": R,
        "outlier_patches": int((norms > OUTLIER_MULT * median).sum()),
        "total_patches": grid * grid,
        "largest_over_median_norm": round(float(norms.max() / median), 2),
        "attention_map_png": overlay_png(image, attention),
        # Same colour scale for every model: norm divided by the model's median, 0 to 4.
        # Yellow (3x the median or more) marks the outlier patches.
        "norm_map_png": heatmap_png(norms / median, vmin=0.0, vmax=OUTLIER_MULT + 1),
    }


def predict(models: dict, image: Image.Image) -> dict:
    image = image.convert("RGB")
    return {
        "classification": [classify(c, image) for c in models["classifiers"]],
        "artifacts": [explain(e, image) for e in models["explainers"]],
    }


# ----------------------------------------------------------------------------
# Pictures for the browser (PNG, base64-encoded)
# ----------------------------------------------------------------------------
def _to_png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _colour(values: np.ndarray, size: int = 224, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Scale to 0..1, colour with viridis, and enlarge with sharp patch edges.

    Without vmin/vmax the map is stretched to its own range (used for attention, which
    sums to 1 anyway); with them, every map shares one fixed scale and values outside are clipped.
    """
    low = values.min() if vmin is None else vmin
    high = values.max() if vmax is None else vmax
    scaled = np.clip((values - low) / (high - low + 1e-12), 0.0, 1.0)
    rgb = (matplotlib.colormaps["viridis"](scaled)[..., :3] * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.NEAREST))


def heatmap_png(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> str:
    return _to_png_base64(_colour(values, vmin=vmin, vmax=vmax))


def overlay_png(image: Image.Image, values: np.ndarray, alpha: float = 0.6) -> str:
    photo = np.asarray(transforms.CenterCrop(224)(transforms.Resize(224)(image)), dtype=np.float32)
    mixed = (1 - alpha) * photo + alpha * _colour(values).astype(np.float32)
    return _to_png_base64(mixed.clip(0, 255).astype(np.uint8))
