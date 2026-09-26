"""Tests for the model (src/vit.py). Run from the repo root: python -m pytest"""

import torch
import torch.nn.functional as F

from src.utils import count_params
from src.vit import build_vit

BATCH, IMG, NUM_PATCHES, DIM, HEADS, CLASSES = 2, 64, 64, 192, 3, 200


def random_images(n: int = BATCH) -> torch.Tensor:
    return torch.randn(n, 3, IMG, IMG)


def test_output_shapes_for_different_register_counts():
    for R in (0, 1, 4, 16):
        model = build_vit("tiny", num_registers=R).eval()
        out = model.forward_features(random_images(), return_attn=True)
        tokens = 1 + R + NUM_PATCHES
        assert model(random_images()).shape == (BATCH, CLASSES)
        assert out["x_prenorm"].shape == (BATCH, tokens, DIM)
        assert out["x_norm_reg"].shape == (BATCH, R, DIM)
        assert out["x_norm_patch"].shape == (BATCH, NUM_PATCHES, DIM)
        assert out["attn_last"].shape == (BATCH, HEADS, tokens, tokens)


def test_registers_come_after_cls_and_have_no_position_embedding():
    model = build_vit("tiny", num_registers=4).eval()
    with torch.no_grad():
        model.register_tokens.fill_(7.0)
        tokens = model.prepare_tokens(random_images(1))
    assert torch.all(tokens[0, 1:5] == 7.0)       # unchanged -> nothing was added to them
    assert not torch.all(tokens[0, 0] == 7.0)     # position 0 is still [CLS]


def test_registers_only_add_their_own_parameters():
    assert count_params(build_vit("tiny", 4)) - count_params(build_vit("tiny", 0)) == 4 * DIM


def test_step_by_step_attention_matches_fast_attention():
    model = build_vit("tiny", num_registers=4).eval()
    images = random_images()
    with torch.no_grad():
        slow = model.forward_features(images, return_attn=True)["x_prenorm"]
        fast = model.forward_features(images, return_attn=False)["x_prenorm"]
    assert torch.allclose(slow, fast, atol=1e-4)


def test_registers_receive_gradients():
    model = build_vit("tiny", num_registers=4)
    loss = F.cross_entropy(model(random_images(4)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert model.register_tokens.grad is not None
    assert model.register_tokens.grad.abs().sum() > 0
