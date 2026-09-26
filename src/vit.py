"""Stage 2 of the pipeline: the model.

A Vision Transformer (ViT) with an option to add register tokens.

How an image flows through the model:
    1. Cut the 64x64 image into 8x8 patches -> 64 patches.
    2. Turn each patch into a vector (a "token").
    3. Build the token sequence:

           [CLS] [REG_1] ... [REG_R] [PATCH_1] ... [PATCH_64]

       - [CLS]      collects a summary of the whole image; used to predict the class.
       - [REG_i]    the paper's idea: extra blank tokens the model can use as scratch space.
       - [PATCH_i]  one token per image patch.
    4. Pass the sequence through a stack of transformer blocks
       (attention lets tokens share information, the MLP processes each token).
    5. Throw the registers away. Predict the class from [CLS].

Register details copied from the official DINOv2 code:
    - Registers get NO position embedding (they don't belong to any location).
      They are added after the position embeddings, between [CLS] and the patches.
    - They start as tiny random values (std 1e-6), like [CLS].

References: Dosovitskiy et al., ICLR 2021 (ViT); Darcet et al., ICLR 2024 (registers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model sizes, using the standard ViT names.
#   embed_dim = length of each token vector, depth = number of blocks, num_heads = attention heads
MODEL_SIZES = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "small": dict(embed_dim=384, depth=12, num_heads=6),
    "base": dict(embed_dim=768, depth=12, num_heads=12),
}


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Cuts the image into patches and turns each patch into a token vector."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.grid_size = img_size // patch_size  # patches per side
        self.num_patches = self.grid_size ** 2
        # A convolution whose kernel and stride both equal the patch size looks at each
        # patch exactly once, so it is the same as "cut into patches + apply a linear layer".
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images)                  # (batch, embed_dim, grid, grid)
        return x.flatten(2).transpose(1, 2)    # (batch, num_patches, embed_dim)


class Attention(nn.Module):
    """Multi-head self-attention: every token looks at every other token and mixes in their information.

    For each token we compute a query (what am I looking for), a key (what do I
    contain) and a value (what do I pass on). Attention weights are
    softmax(query . key / sqrt(head_dim)), and each token's output is the
    weighted sum of the values.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)  # computes query, key and value in one go
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        batch, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (batch, heads, tokens, head_dim)

        if return_attn:
            # Written out step by step so we can return the attention weights (for plotting).
            attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)  # (batch, heads, tokens, tokens)
            out = attn @ v
        else:
            # PyTorch's built-in version of the same maths. Faster, used during training.
            out = F.scaled_dot_product_attention(q, k, v)
            attn = None

        out = out.transpose(1, 2).reshape(batch, num_tokens, dim)  # put the heads back together
        return self.proj(out), attn


class MLP(nn.Module):
    """Processes each token on its own: expand -> GELU -> shrink back."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    """Stochastic depth: during training, randomly skip a block's update for some images.

    This acts like dropout for whole blocks and reduces overfitting. It does
    nothing at evaluation time.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # One keep/skip decision per image in the batch.
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        return x * mask / keep_prob  # rescale so the average output stays the same


class Block(nn.Module):
    """One transformer block.

    x = x + Attention(LayerNorm(x))   tokens share information
    x = x + MLP(LayerNorm(x))         each token is processed on its own

    The "x +" parts (residual connections) add to each token instead of replacing it.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        attn_out, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


# ----------------------------------------------------------------------------
# The full model
# ----------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 200,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_registers: int = 0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_registers = num_registers

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))  # for [CLS] + patches only
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_registers, embed_dim)) if num_registers > 0 else None
        )

        # Transformer blocks. The skip probability grows from 0 (first block) to drop_path_rate (last block).
        drop_probs = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop_path=p) for p in drop_probs]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)  # turns [CLS] into one score per class

        self._init_weights()

    def _init_weights(self) -> None:
        """Starting values for the weights (same scheme as the DINOv2 code)."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def prepare_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Image -> token sequence [CLS, REG_1..REG_R, PATCH_1..PATCH_N]."""
        batch = images.shape[0]
        x = self.patch_embed(images)                                     # (batch, N, D)
        x = torch.cat([self.cls_token.expand(batch, -1, -1), x], dim=1)  # (batch, 1+N, D)
        x = x + self.pos_embed                                           # add position information
        if self.register_tokens is not None:
            # Insert the registers right after [CLS]. Added after pos_embed, so they have no position.
            registers = self.register_tokens.expand(batch, -1, -1)
            x = torch.cat([x[:, :1], registers, x[:, 1:]], dim=1)        # (batch, 1+R+N, D)
        return x

    def forward_features(self, images: torch.Tensor, return_attn: bool = False) -> dict:
        """Run the transformer and return every kind of output token separately.

        Returns a dict with:
            x_prenorm     (batch, 1+R+N, D)  output of the last block, before the final LayerNorm
            x_norm_cls    (batch, D)         the [CLS] token
            x_norm_reg    (batch, R, D)      the registers (not used for prediction; kept for analysis)
            x_norm_patch  (batch, N, D)      the patch tokens
            attn_last     (batch, heads, 1+R+N, 1+R+N) attention weights of the last block,
                          or None if return_attn is False
        """
        x = self.prepare_tokens(images)
        last_attn = None
        last_index = len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            keep_attn = return_attn and i == last_index
            x, attn = block(x, return_attn=keep_attn)
            if keep_attn:
                last_attn = attn

        x_norm = self.norm(x)
        R = self.num_registers
        return {
            "x_prenorm": x,
            "x_norm_cls": x_norm[:, 0],
            "x_norm_reg": x_norm[:, 1:1 + R],
            "x_norm_patch": x_norm[:, 1 + R:],
            "attn_last": last_attn,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Class scores (logits). Only [CLS] is used; the registers are thrown away."""
        return self.head(self.forward_features(images)["x_norm_cls"])


def build_vit(size: str = "tiny", num_registers: int = 0, **kwargs) -> VisionTransformer:
    """Create a ViT of a given size ("tiny", "small", "base") with `num_registers` registers."""
    if size not in MODEL_SIZES:
        raise ValueError(f"size must be one of {list(MODEL_SIZES)}")
    return VisionTransformer(num_registers=num_registers, **MODEL_SIZES[size], **kwargs)
