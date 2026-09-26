"""Milestone check (Week 3): data loads, and the model runs a correct forward + backward pass.

Checks, for our ViT with 0 and with 4 registers:
    - the data comes out in the right shape, with valid labels and normalised pixels
    - the model produces one score per class and a finite loss
    - the starting loss is close to ln(200), as expected for a model that hasn't learned anything
    - gradients reach every parameter, including the registers
    - the token sequence has length 1 + R + 64, and registers are split off at the output
    - our step-by-step attention gives the same result as PyTorch's fast attention

Usage (from the repo root):
    python -m scripts.check_forward_pass          # real Tiny-ImageNet (downloads ~237 MB)
    python -m scripts.check_forward_pass --fake   # random images, code test only

Writes results/logs/forward_pass_check.md and results/figures/tinyimagenet_samples.png,
and copies the report into README.md. Exits with an error if any check fails.
"""

import argparse
import math
import platform
import sys

import torch
import torch.nn.functional as F

from src.data import IMG_SIZE, NUM_CLASSES, get_dataloaders
from src.plots import plot_sample_batch
from src.report import inject_into_readme, markdown_table
from src.utils import REPO_ROOT, count_params, get_device, set_seed
from src.vit import build_vit

NUM_REGISTERS_TO_TEST = (0, 4)
REAL_TRAIN_SIZE, REAL_VAL_SIZE = 100_000, 10_000


class Checker:
    """Records pass/fail results and prints them as it goes."""

    def __init__(self):
        self.rows: list[list[str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        self.rows.append([name, status, detail])
        print(f"[{status}] {name}  {detail}")

    @property
    def failures(self) -> list[str]:
        return [row[0] for row in self.rows if row[1] == "FAIL"]


def check_data(c: Checker, train_loader, val_loader, batch_size: int, fake: bool):
    """Stage 1: does the data pipeline produce correct batches?"""
    images, labels = next(iter(train_loader))
    val_images, _ = next(iter(val_loader))
    n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)

    c.check("Train set size", fake or n_train == REAL_TRAIN_SIZE, f"{n_train:,} images")
    c.check("Val set size", fake or n_val == REAL_VAL_SIZE, f"{n_val:,} images")
    c.check("Train batch shape", tuple(images.shape) == (batch_size, 3, IMG_SIZE, IMG_SIZE), f"{tuple(images.shape)}")
    c.check("Val batch shape", tuple(val_images.shape[1:]) == (3, IMG_SIZE, IMG_SIZE), f"{tuple(val_images.shape)}")
    c.check("Labels in [0, 199]", 0 <= int(labels.min()) and int(labels.max()) < NUM_CLASSES,
            f"min {int(labels.min())}, max {int(labels.max())}")
    c.check("Pixels normalised (mean near 0, std near 1)",
            abs(images.mean().item()) < 1.0 and 0.3 < images.std().item() < 2.0,
            f"mean {images.mean():.3f}, std {images.std():.3f}")
    return images, labels


def check_model(c: Checker, num_registers: int, images, labels) -> int:
    """Stage 2: does the model run forward and backward correctly? Returns its parameter count."""
    tag = f"[{num_registers} reg]"
    model = build_vit("tiny", num_registers=num_registers).to(images.device)
    num_patches, dim = model.num_patches, model.embed_dim

    # Forward + backward in training mode
    model.train()
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    c.check(f"{tag} logits shape", tuple(logits.shape) == (len(images), NUM_CLASSES), f"{tuple(logits.shape)}")
    c.check(f"{tag} loss is finite", bool(torch.isfinite(loss)), f"{loss.item():.4f}")
    c.check(f"{tag} starting loss close to ln(200)", abs(loss.item() - math.log(NUM_CLASSES)) < 0.5,
            f"{loss.item():.3f} vs {math.log(NUM_CLASSES):.3f}")
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    c.check(f"{tag} every parameter gets a finite gradient", grads_ok)
    if num_registers > 0:
        grad_norm = model.register_tokens.grad.norm().item()
        c.check(f"{tag} registers receive gradient", grad_norm > 0, f"grad norm {grad_norm:.2e}")

    # Token bookkeeping and attention in evaluation mode
    model.eval()
    with torch.no_grad():
        tokens = model.prepare_tokens(images)
        with_attn = model.forward_features(images, return_attn=True)
        fast = model.forward_features(images)
    c.check(f"{tag} sequence length = 1 + R + 64", tokens.shape[1] == 1 + num_registers + num_patches,
            f"{tokens.shape[1]} = 1 + {num_registers} + {num_patches}")
    split_ok = (tuple(with_attn["x_norm_reg"].shape[1:]) == (num_registers, dim)
                and tuple(with_attn["x_norm_patch"].shape[1:]) == (num_patches, dim))
    c.check(f"{tag} registers split off at output", split_ok,
            f"reg {tuple(with_attn['x_norm_reg'].shape)}, patch {tuple(with_attn['x_norm_patch'].shape)}")
    row_sums = with_attn["attn_last"].sum(dim=-1)
    c.check(f"{tag} attention weights sum to 1", torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4),
            f"attention shape {tuple(with_attn['attn_last'].shape)}")
    max_diff = (with_attn["x_prenorm"] - fast["x_prenorm"]).abs().max().item()
    c.check(f"{tag} step-by-step attention == fast attention", max_diff < 1e-3, f"max difference {max_diff:.1e}")
    return count_params(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="random images, code test only")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    set_seed(0)
    device = get_device()
    c = Checker()

    # Stage 1: data
    train_loader, val_loader = get_dataloaders(args.data_root, args.batch_size, fake=args.fake)
    images, labels = check_data(c, train_loader, val_loader, args.batch_size, args.fake)
    plot_sample_batch(images, labels, REPO_ROOT / "results" / "figures" / "tinyimagenet_samples.png",
                      "Tiny-ImageNet training batch after augmentation (64x64)")

    # Stage 2: model, with and without registers
    images, labels = images.to(device), labels.to(device)
    params = {R: check_model(c, R, images, labels) for R in NUM_REGISTERS_TO_TEST}
    dim = build_vit("tiny").embed_dim
    c.check("Registers add exactly R x 192 parameters", params[4] - params[0] == 4 * dim,
            f"{params[0]:,} -> {params[4]:,} (+{params[4] - params[0]})")

    # Report
    gpu = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    data_name = "**random FakeData (code test only)**" if args.fake else "Tiny-ImageNet"
    passed = len(c.rows) - len(c.failures)
    report = "\n".join([
        f"Data: {data_name}. Model: ViT-Tiny (192-dim, 12 blocks, 3 heads, 8x8 patches, 64 patch tokens). "
        f"Batch size {args.batch_size}. Device `{device}`{gpu}, Python {platform.python_version()}, "
        f"torch {torch.__version__}.",
        "",
        markdown_table(["Check", "Result", "Detail"], c.rows),
        "",
        f"**{passed}/{len(c.rows)} checks passed.**",
    ])
    log_path = REPO_ROOT / "results" / "logs" / "forward_pass_check.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report + "\n")
    if not args.fake:
        inject_into_readme("FORWARD_CHECK", report)

    print(f"\n{passed}/{len(c.rows)} checks passed")
    if c.failures:
        sys.exit(f"FAILED: {c.failures}")


if __name__ == "__main__":
    main()
