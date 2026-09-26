"""Stage 3 of the pipeline: train the model and evaluate it.

Training recipe (standard for training small ViTs from scratch, following DeiT):
    - Optimiser:  AdamW, learning rate 1e-3, weight decay 0.05
    - Schedule:   learning rate rises linearly for 5 epochs (warmup), then decays
                  smoothly to 0 following a cosine curve
    - Loss:       cross-entropy with label smoothing 0.1
    - Extras:     stochastic depth 0.1, gradient clipping at 1.0,
                  mixed precision on GPU (faster, uses less memory)

We report the model from the LAST epoch. We do not pick the "best" epoch,
because the validation set is also our test set; picking the best epoch on it
would make the score look better than it really is.

Usage (from the repo root):
    # quick check that training works (a few minutes)
    python -m src.train --registers 0 --epochs 5 --train-subset 10000 --val-subset 2000 --out results/runs/sanity_tiny_reg0

    # full runs
    python -m src.train --size tiny --registers 0 --epochs 50 --out results/runs/tiny_reg0
    python -m src.train --size tiny --registers 4 --epochs 50 --out results/runs/tiny_reg4

Saved in --out:
    config.json     all settings of the run
    log.csv         one row per epoch (loss, accuracy, learning rate, time)
    final.json      final validation accuracy and loss
    checkpoint.pt   model + optimiser state after the last finished epoch

Resuming: with --resume, if --out already has a checkpoint.pt, training continues
from the last finished epoch instead of starting again (useful if Colab
disconnects). A resumed run gives the same schedule, but not bit-identical
numbers, because the random data order after the restart differs.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import NUM_CLASSES, get_dataloaders
from src.utils import count_params, get_device, set_seed
from src.vit import MODEL_SIZES, build_vit

LOG_COLUMNS = ["epoch", "step", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a ViT with or without registers on Tiny-ImageNet.")
    # model
    p.add_argument("--size", default="tiny", choices=list(MODEL_SIZES))
    p.add_argument("--registers", type=int, default=0, help="number of register tokens (paper uses 4)")
    p.add_argument("--drop-path", type=float, default=0.1)
    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (for code tests)")
    p.add_argument("--seed", type=int, default=0)
    # data
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-subset", type=int, default=None, help="use only this many training images")
    p.add_argument("--val-subset", type=int, default=None, help="use only this many validation images")
    p.add_argument("--fake", action="store_true", help="random images instead of real data (code test only)")
    p.add_argument("--num-workers", type=int, default=2)
    # output
    p.add_argument("--out", default="results/runs/run")
    p.add_argument("--resume", action="store_true", help="continue from --out/checkpoint.pt if it exists")
    args = p.parse_args(argv)
    if args.epochs < 1:
        p.error("--epochs must be at least 1")
    return args


# ----------------------------------------------------------------------------
# Optimiser and learning-rate schedule
# ----------------------------------------------------------------------------
def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW. Weight decay (which pulls weights towards 0) is only applied to the
    weight matrices, not to biases, LayerNorm parameters or the learnable tokens."""
    no_decay_names = {"cls_token", "pos_embed", "register_tokens"}
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim < 2 or name in no_decay_names:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup for `warmup_steps`, then cosine decay to 0 at `total_steps`."""

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


# ----------------------------------------------------------------------------
# One pass over the data
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device,
                    step: int, max_steps: int | None) -> tuple[float, float, int]:
    """Train for one epoch. Returns (average loss, accuracy, step count so far)."""
    model.train()
    use_amp = device.type == "cuda"
    seen, correct, loss_sum = 0, 0, 0.0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Forward pass (in mixed precision on GPU)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became {loss.item()} at step {step}; stopping.")

        # Backward pass and weight update
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                               # so clipping sees the true gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # avoid huge update steps
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Running statistics
        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
        step += 1
        if max_steps is not None and step >= max_steps:
            break

    return loss_sum / seen, correct / seen, step


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device) -> tuple[float, float]:
    """Average loss and accuracy on a dataset, without updating the model."""
    model.eval()
    use_amp = device.type == "cuda"
    seen, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return loss_sum / seen, correct / seen


# ----------------------------------------------------------------------------
# Saving and resuming
# ----------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, config: dict,
                    epoch: int, step: int, val_loss: float, val_acc: float) -> None:
    """Save everything needed to continue training. Written to a temporary file first,
    so a disconnect during saving can never leave a broken checkpoint behind."""
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "config": config, "epoch": epoch, "step": step, "val_loss": val_loss, "val_acc": val_acc,
    }
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def trim_log(log_path: Path, last_epoch: int) -> None:
    """Keep only the log rows up to `last_epoch` (drops a row written just before a disconnect)."""
    with open(log_path, newline="") as f:
        rows = list(csv.reader(f))
    kept = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= last_epoch]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(kept)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
        args.train_subset, args.val_subset, fake=args.fake, seed=args.seed,
    )

    # 2. Model
    model = build_vit(args.size, num_registers=args.registers,
                      num_classes=NUM_CLASSES, drop_path_rate=args.drop_path).to(device)

    # 3. Loss, optimiser, schedule
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    # Warmup is 5 epochs, but never more than 10% of training (matters only for very short runs).
    warmup_steps = min(int(args.warmup_epochs * steps_per_epoch), max(total_steps // 10, 1))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")  # needed for mixed precision

    # 4. Save the settings of this run (or pick up where a previous run stopped)
    config = vars(args) | {
        "device": str(device), "params": count_params(model), "torch": torch.__version__,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup_steps,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    log_path, ckpt_path = out_dir / "log.csv", out_dir / "checkpoint.pt"

    start_epoch, step = 1, 0
    val_loss = val_acc = float("nan")
    if args.resume and ckpt_path.exists() and log_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        val_loss, val_acc = ckpt["val_loss"], ckpt["val_acc"]
        trim_log(log_path, ckpt["epoch"])
        print(f"Resuming from epoch {ckpt['epoch']} (step {step})")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)
    print(f"{args.size} ViT | {args.registers} registers | {count_params(model):,} params | device {device}")

    # 5. Train
    last_epoch = start_epoch - 1  # last fully finished epoch
    for epoch in range(start_epoch, args.epochs + 1):
        if args.max_steps is not None and step >= args.max_steps:
            break
        start = time.time()
        train_loss, train_acc, step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, device, step, args.max_steps)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step, scheduler.get_last_lr()[0], train_loss, train_acc,
                                    val_loss, val_acc, epoch_time])
        print(f"epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} | {epoch_time:.0f}s")
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, config, epoch, step, val_loss, val_acc)
        last_epoch = epoch

    # 6. Final result
    final = {"val_acc": val_acc, "val_loss": val_loss, "epochs_run": last_epoch, "steps": step}
    (out_dir / "final.json").write_text(json.dumps(final, indent=2))
    print("final:", final)
    return final


if __name__ == "__main__":
    main()
