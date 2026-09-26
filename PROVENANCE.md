# Provenance

Tags:
- **Written**: written for this project
- **Adapted**: follows or changes existing code
- **Reused**: used as-is

| Component | Tag | Source / notes |
|---|---|---|
| `src/vit.py`: patch embedding, attention, MLP, block, ViT | Written | Standard ViT architecture (Dosovitskiy et al., 2021), implemented from scratch in PyTorch. |
| `src/vit.py`: register tokens | Written | Where registers are inserted (after position embeddings, between [CLS] and patches) and how they are initialised (std 1e-6) follow `prepare_tokens_with_masks` and `init_weights` in [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) (`dinov2/models/vision_transformer.py`). No code copied. |
| `src/vit.py`: `DropPath` | Written | Stochastic depth (Huang et al., 2016); behaves like the common timm version. |
| `src/data.py` | Written | Reads the official Tiny-ImageNet zip layout. Uses torchvision's transforms, `FakeData` and `download_and_extract_archive` (library functions, reused). |
| `src/train.py` | Written | Training recipe (AdamW, warmup + cosine schedule, label smoothing, stochastic depth) follows DeiT (Touvron et al., 2021) conventions, scaled down. |
| `src/analysis.py`: norms, neighbour similarity, outlier rule | Written | Measurements designed to mirror the paper's Fig 3, 4c, 5a, 7, 15. |
| `src/analysis.py`: `cls_attention_map` | Adapted | Same maths as `Attention.forward` in DINOv2 (`dinov2/layers/attention.py`), rewritten to return the attention weights. Tested against our own model's attention (`tests/test_analysis.py`). |
| `src/plots.py`, `src/report.py`, `src/utils.py` | Written | |
| `scripts/check_forward_pass.py`, `scripts/run_part_a.py` | Written | |
| `tests/`, `notebooks/milestone2_colab.ipynb` | Written | |
| DINOv2 model code and pretrained weights | Reused | Loaded unchanged with `torch.hub.load("facebookresearch/dinov2", ...)`. Apache-2.0 licence. |
| Tiny-ImageNet dataset | Reused | http://cs231n.stanford.edu/tiny-imagenet-200.zip |
| Imagenette dataset | Reused | Through `torchvision.datasets.Imagenette` (fast.ai). |

**AI assistance:** code in this repository was drafted with the help of an AI assistant (Claude, Anthropic) and reviewed, run and tested by us.

## Results provenance

- Every number and figure in `results/` was **produced by us** by running the code in this repository.
- Numbers quoted from the paper (for example the 150 norm cutoff, 2.37% outliers, 4 registers) were **reported by the original authors** and are marked as such wherever they appear.
