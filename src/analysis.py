"""Stage 4 of the pipeline: measure what the paper measures.

The paper finds "artifact" tokens by their norm (vector length): artifact
patch tokens are about 10x longer than normal ones. This file contains the
measurements used in our experiments:

    - token norms                     (paper Fig 3, 4, 7, 15)
    - similarity of each patch to its 4 neighbours, right after patch embedding
                                      (paper Fig 5a: artifacts sit on "boring" patches)
    - [CLS] attention map of the last block
                                      (paper Fig 1: the bright spots)

The functions work both for Meta's pretrained DINOv2 models and for our own ViT.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Meta's released DINOv2 models (all use 14x14 patches). "_reg" versions have 4 registers.
DINOV2_ARCHS = {"S": "vits14", "B": "vitb14", "L": "vitl14", "g": "vitg14"}


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def outlier_cutoff(norms: np.ndarray, multiplier: float = 3.0) -> float:
    """Norm above which a token counts as an outlier: `multiplier` x the median norm.

    Always computed on the SAME model's norms: different models have very
    different typical token lengths, so each model is judged against its own
    typical token. The paper hand-picked 150 for DINOv2 ViT-g and notes the value
    depends on the model; for ViT-g this rule gives a similar value.
    """
    return multiplier * float(np.median(norms))


def neighbour_cosine(patch_tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    """For every patch: average cosine similarity to its up/down/left/right neighbours.

    Cosine similarity is 1 when two vectors point the same way (patches look alike)
    and 0 when they are unrelated. Edge patches have fewer than 4 neighbours, so we
    average over the neighbours they do have.

    Args:
        patch_tokens: (batch, grid_size * grid_size, dim)
    Returns:
        (batch, grid_size * grid_size)
    """
    batch, num_patches, dim = patch_tokens.shape
    unit = F.normalize(patch_tokens.float(), dim=-1).reshape(batch, grid_size, grid_size, dim)

    sim_sum = torch.zeros(batch, grid_size, grid_size, device=unit.device)
    num_neighbours = torch.zeros(1, grid_size, grid_size, device=unit.device)

    left_right = (unit[:, :, :-1] * unit[:, :, 1:]).sum(-1)  # each patch with the one to its right
    up_down = (unit[:, :-1] * unit[:, 1:]).sum(-1)           # each patch with the one below it

    # Every pair counts for both patches in the pair.
    sim_sum[:, :, :-1] += left_right
    sim_sum[:, :, 1:] += left_right
    sim_sum[:, :-1] += up_down
    sim_sum[:, 1:] += up_down
    num_neighbours[:, :, :-1] += 1
    num_neighbours[:, :, 1:] += 1
    num_neighbours[:, :-1] += 1
    num_neighbours[:, 1:] += 1

    return (sim_sum / num_neighbours).reshape(batch, num_patches)


def cls_attention_map(attn_module: nn.Module, attn_input: torch.Tensor,
                      num_registers: int, grid_size: int) -> torch.Tensor:
    """How much [CLS] attends to each patch, averaged over heads, as a (grid x grid) image.

    Recomputes softmax(query . key * scale) from the attention layer's own weights,
    exactly as the layer does internally. Works for DINOv2's attention layer and
    ours (both have .qkv, .num_heads and .scale).

    Args:
        attn_module: the attention layer of the last block
        attn_input:  the tokens that went INTO that layer, (batch, tokens, dim)
    Returns:
        (batch, grid_size, grid_size)
    """
    batch, num_tokens, dim = attn_input.shape
    heads = attn_module.num_heads
    qkv = attn_module.qkv(attn_input).reshape(batch, num_tokens, 3, heads, dim // heads)
    q, k, _ = torch.unbind(qkv, dim=2)
    q, k = q.transpose(1, 2), k.transpose(1, 2)                           # (batch, heads, tokens, head_dim)
    attn = ((q * attn_module.scale) @ k.transpose(-2, -1)).softmax(dim=-1)  # (batch, heads, tokens, tokens)

    # Row 0 = what [CLS] looks at. Skip [CLS] and the registers, keep only the patch columns.
    cls_to_patches = attn[:, :, 0, 1 + num_registers:]
    return cls_to_patches.mean(dim=1).reshape(batch, grid_size, grid_size)


# ----------------------------------------------------------------------------
# Pretrained DINOv2 models
# ----------------------------------------------------------------------------
def load_dinov2(size: str, registers: bool, pretrained: bool = True) -> tuple[nn.Module, str]:
    """Load one of Meta's DINOv2 models from torch.hub (code + weights are downloaded).

    Args:
        size: "S", "B", "L" or "g"
        registers: True for the version trained with 4 registers
        pretrained: False gives random weights (only for testing our code)
    """
    name = f"dinov2_{DINOV2_ARCHS[size]}" + ("_reg" if registers else "")
    model = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained,
                           trust_repo=True, skip_validation=True)
    return model.eval(), name


@torch.no_grad()
def collect_dinov2_stats(model: nn.Module, loader: DataLoader, device: torch.device,
                         num_maps: int) -> dict:
    """Run a DINOv2 model over a set of images and record everything we analyse.

    Token norms are measured on the output of the last transformer block, BEFORE
    the final LayerNorm. The final LayerNorm rescales every token to a similar
    length, which hides the outliers; the paper's high-norm tokens (norms up to
    ~600 for ViT-g, Fig 3) are only visible before it. We still record the norms
    after the LayerNorm, to show this effect.

    Returns a dict of numpy arrays:
        patch_norm           (images, patches)   patch-token norms (main measurement)
        patch_norm_after_ln  (images, patches)   patch-token norms after the final LayerNorm
        cls_norm             (images,)           [CLS] norm
        reg_norm             (images, registers) register norms (empty if the model has none)
        cosine               (images, patches)   similarity to neighbours after patch embedding
        attn_maps            (num_maps, grid, grid)  [CLS] attention maps of the first images
        grid_size            int
    """
    num_registers = model.num_register_tokens
    last_attn = model.blocks[-1].attn

    # A "hook" lets us grab the input of the last attention layer while the model runs.
    captured = {}
    hook = last_attn.register_forward_pre_hook(lambda module, args: captured.update(x=args[0]))

    keys = ["patch_norm", "patch_norm_after_ln", "cls_norm", "reg_norm", "cosine", "attn_maps"]
    parts = {key: [] for key in keys}
    maps_collected = 0
    try:
        for images, _ in loader:
            images = images.to(device)
            grid_size = images.shape[-1] // model.patch_size
            out = model.forward_features(images)
            tokens = out["x_prenorm"]  # (batch, 1 + R + patches, dim): [CLS], registers, patches

            parts["patch_norm"].append(tokens[:, 1 + num_registers:].norm(dim=-1).cpu())
            parts["patch_norm_after_ln"].append(out["x_norm_patchtokens"].norm(dim=-1).cpu())
            parts["cls_norm"].append(tokens[:, 0].norm(dim=-1).cpu())
            parts["reg_norm"].append(tokens[:, 1:1 + num_registers].norm(dim=-1).cpu())
            parts["cosine"].append(neighbour_cosine(model.patch_embed(images), grid_size).cpu())

            if maps_collected < num_maps:
                maps = cls_attention_map(last_attn, captured["x"], num_registers, grid_size)
                parts["attn_maps"].append(maps.cpu())
                maps_collected += maps.shape[0]
    finally:
        hook.remove()  # always detach the hook, even if something fails

    stats = {key: torch.cat(values).numpy() for key, values in parts.items()}
    stats["attn_maps"] = stats["attn_maps"][:num_maps]
    stats["grid_size"] = grid_size
    return stats
