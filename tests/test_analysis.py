"""Tests for the measurements (src/analysis.py). Run from the repo root: python -m pytest"""

import numpy as np
import torch

from src.analysis import cls_attention_map, neighbour_cosine, outlier_cutoff
from src.vit import build_vit


def test_identical_patches_have_similarity_one():
    patches = torch.ones(1, 16, 8)  # 4x4 grid, every patch the same
    assert torch.allclose(neighbour_cosine(patches, grid_size=4), torch.ones(1, 16))


def test_checkerboard_patches_have_similarity_minus_one():
    grid = 4
    signs = torch.tensor([[(-1.0) ** (r + c) for c in range(grid)] for r in range(grid)])
    patches = signs.reshape(1, grid * grid, 1) * torch.ones(1, 1, 8)  # every neighbour points the opposite way
    assert torch.allclose(neighbour_cosine(patches, grid), -torch.ones(1, grid * grid))


def test_outlier_cutoff_is_multiple_of_median():
    assert outlier_cutoff(np.array([1.0, 2.0, 3.0, 100.0]), multiplier=3.0) == 7.5


def test_cls_attention_map_matches_the_model_own_attention():
    """Recomputing attention from the layer's weights must give the model's real attention."""
    R = 4
    model = build_vit("tiny", num_registers=R).eval()
    images = torch.randn(2, 3, 64, 64)
    captured = {}
    hook = model.blocks[-1].attn.register_forward_pre_hook(lambda m, args: captured.update(x=args[0]))
    with torch.no_grad():
        out = model.forward_features(images, return_attn=True)
        ours = cls_attention_map(model.blocks[-1].attn, captured["x"], R, model.grid_size)
    hook.remove()
    expected = out["attn_last"][:, :, 0, 1 + R:].mean(dim=1).reshape(2, 8, 8)
    assert torch.allclose(ours, expected, atol=1e-6)
