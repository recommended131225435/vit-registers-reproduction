# Reproducing "Vision Transformers Need Registers"

Course project (Machine Learning, IBA Karachi): paper reproduction, experiment and deployment.

**Paper:** T. Darcet, M. Oquab, J. Mairal, P. Bojanowski. *Vision Transformers Need Registers.* ICLR 2024 (oral). [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)

## 1. The paper in short

Large Vision Transformers (DINOv2, CLIP, DeiT-III) show bright spots in their attention maps on empty background areas. The authors show these spots are patch tokens with a very large norm (about 10x normal). These tokens have lost the information about their own patch and instead hold information about the whole image. Their explanation: the model needs extra space for global computation, and without any, it takes over patches it considers useless. The fix is to add a few extra learnable tokens ("registers") to the input that the model can use as scratch space. These tokens are thrown away at the output. With registers, the artifacts disappear and dense tasks improve.

## 2. What this repository does

| Part | What | Status |
|---|---|---|
| **A** | Reproduce the paper's diagnostic findings (Fig 1, 3, 4c, 5a, 7, 15) on the **official pretrained DINOv2 models**, with and without registers. No training needed. | Code ready: `scripts/run_part_a.py` |
| **B** | **Our own ViT implementation with a register option**, a Tiny-ImageNet data pipeline, and a training script. Train small ViTs with 0 vs 4 registers. | Pipeline, model and forward pass done. Full training runs in Week 4. |
| Experiment | *At what model size do high-norm artifacts appear, and do registers change anything below that size?* | Week 4 |
| Deployment | FastAPI app: upload an image, see attention maps without vs with registers | Week 5 |

## 3. Repository structure

The code follows the ML pipeline, one file per stage:

```
src/
├── data.py        1. Data: download Tiny-ImageNet, preprocess, batch
├── vit.py         2. Model: our Vision Transformer with optional registers
├── train.py       3. Training + evaluation (AdamW, warmup + cosine LR, mixed precision, CSV logs)
├── analysis.py    4. Measurements from the paper: token norms, neighbour similarity, attention maps
├── plots.py       5. Figures
├── report.py      6. Writes results into this README
└── utils.py          Shared helpers (random seed, device, parameter count)

scripts/
├── check_forward_pass.py   Week 3 check: data -> model -> loss -> gradients (24 checks)
└── run_part_a.py           Part A: analysis of pretrained DINOv2 models

tests/                      Unit tests for the model and the measurements
notebooks/milestone2_colab.ipynb   Runs everything on Google Colab
results/
├── figures/     all plots
├── logs/        forward pass check report
├── part_a/      Part A numbers (summary.md, summary.json)
└── runs/        training logs (config.json, log.csv, final.json)
requirements.txt        minimum library versions
requirements-lock.txt   exact versions used on Colab (created by the notebook)
PROVENANCE.md           where every piece of code came from
```

## 4. How to run

**Easiest: Google Colab.** Open `notebooks/milestone2_colab.ipynb` in Colab, choose a T4 GPU runtime, and run all cells (about 30–40 min).

**By hand** (always from the repo root):

```bash
pip install -r requirements.txt
python -m pytest                                  # unit tests
python -m scripts.check_forward_pass              # downloads Tiny-ImageNet, checks data + forward pass
python -m scripts.run_part_a --sizes S,B,L,g      # Part A (GPU recommended)
python -m src.train --size tiny --registers 4 --epochs 50 --out results/runs/tiny_reg4   # a full training run
```

Every script has a quick code-test mode that uses random data instead of downloads (`--fake` or `--smoke`).

## 5. Part B: our ViT with registers

**Token sequence inside the transformer:**

```
[CLS] [REG_1] [REG_2] [REG_3] [REG_4] [PATCH_1] ... [PATCH_64]
  │      └──── learnable, no position ────┘  └─ 8x8 patches of a 64x64 image
  │           embedding, discarded at output
  └── used for classification
```

**Design choices:**

| Choice | Value | Why |
|---|---|---|
| Dataset | Tiny-ImageNet (100k train / 10k val, 200 classes, 64x64) | Freely downloadable, small enough for a free T4, 200 classes is harder than CIFAR. The official val set is our test set because test labels are not public. |
| Patch size | 8 → 8x8 = 64 patch tokens | Standard choice for 64x64 images |
| Model size | ViT-Tiny: 192-dim, 12 layers, 3 heads (5.4M params). Small and Base are also available for the Week 4 experiment. | Trains in reasonable time on a free GPU |
| Number of registers | 0 (baseline) and 4 | The paper uses 4 in all its main experiments (Sec 3.2) |
| Register placement | After the position embeddings, between [CLS] and patches, with no position embedding | Same as the official DINOv2 implementation |
| Register init | Normal, std 1e-6 (same as [CLS]) | Same as the official DINOv2 implementation |
| Output | Registers discarded; [CLS] → linear head | As in the paper (Fig 6) |
| Training | AdamW (lr 1e-3, wd 0.05), 5 warmup epochs + cosine decay, label smoothing 0.1, stochastic depth 0.1, random-resized-crop + flip, mixed precision | Standard DeiT-style recipe for training ViTs from scratch on small data |

### Forward pass check (Week 3 milestone)

Output of `python -m scripts.check_forward_pass` on the real dataset:

<!-- FORWARD_CHECK_START -->
*Not run yet. Run `python -m scripts.check_forward_pass` (or the Colab notebook) and this section fills itself in.*
<!-- FORWARD_CHECK_END -->

![Tiny-ImageNet samples](results/figures/tinyimagenet_samples.png)

**Sanity training runs** (10k-image subset, 5 epochs, 0 vs 4 registers). These only show that the training loop learns. They are not results.

![Sanity runs](results/figures/sanity_runs.png)

## 6. Part A: reproducing the paper's findings on pretrained DINOv2

We load Meta's released DINOv2 checkpoints (ViT-S/B/L/g with patch size 14, each with and without 4 registers). We run them on 256 Imagenette validation images, and measure:

| Paper claim | Paper evidence | Our evidence |
|---|---|---|
| Some patch tokens have a much larger norm (two separate humps in the histogram) | Fig 3 | `part_a_norm_histograms.png` |
| The outliers appear only in larger models | Fig 4c | `part_a_outliers_by_size.png` |
| Outliers sit on patches that are very similar to their neighbours | Fig 5a | `part_a_neighbour_cosine.png` |
| Registers remove the outliers | Fig 7 | histograms + table below |
| Registers give clean attention maps | Fig 1, 19 | `part_a_attention_maps.png` |
| With registers, the high norms move into the register tokens | Fig 15 | register-norm table below |

**How we find outliers:** We use the L2 norm (vector length) of each patch token at the output of the last transformer block, **before the model's final LayerNorm**. The final LayerNorm rescales every token to a similar length, which hides the outliers; our first run measured after it and found none (see `part_a_norm_histograms_after_layernorm.png`). A token is an outlier if its norm is more than 3 times the median norm of the same model. We judge each model against its own typical token because different models have very different typical norms. For ViT-g we also apply the paper's own hand-picked cutoff of 150.

**How we get attention maps:** A hook captures the input to the last attention layer. We recompute `softmax(q·kᵀ/√d)` and take the [CLS] row over the patch tokens, averaged over heads. We checked this against the DINOv2 module: recomputing the full attention output this way matches the module's own output to within 1e-7.

<!-- PART_A_START -->
*Not run yet. Run `python -m scripts.run_part_a` (or the Colab notebook) and this section fills itself in.*
<!-- PART_A_END -->

![Norm histograms](results/figures/part_a_norm_histograms.png)
![Outliers by size](results/figures/part_a_outliers_by_size.png)
![Attention maps](results/figures/part_a_attention_maps.png)
![Norm maps](results/figures/part_a_norm_maps.png)
![Neighbour similarity](results/figures/part_a_neighbour_cosine.png)
![Norms after LayerNorm](results/figures/part_a_norm_histograms_after_layernorm.png)

### What Part A shows

- **High-norm tokens (Fig 3): reproduced for ViT-g.** Without registers, ViT-g's patch-token norms form two separate humps: most tokens sit below ~150 and a small group sits between roughly 300 and 560. About 2.6% of tokens are above the paper's cutoff of 150, against 2.37% in the paper.
- **Registers remove them (Fig 7): reproduced.** The ViT-g model with registers has no second hump.
- **Attention maps (Fig 1): reproduced.** Without registers, ViT-B, L and g show bright spots on background patches (for example image corners). With registers, attention sits on the object. ViT-S is clean in both versions, in line with the paper's finding that small models don't have artifacts.
- **Model size (Fig 4c): partly reproduced.** Clear high-norm outliers appear only in ViT-g (ViT-B has a very small tail, S and L none). The paper finds outliers from ViT-L upwards. A likely reason: Meta's released S, B and L models were distilled from ViT-g rather than trained on their own as in the paper's Fig 4c. ViT-L still shows attention-map artifacts, so the behaviour is not fully gone in it.
- **The final LayerNorm hides the outliers.** After it, the largest token is only slightly longer than a typical one in every model. The paper's norms must therefore be measured before this layer.

## 7. Differences from the paper

**Changes from our proposal**

- We use only DINOv2, not CLIP. Meta released DINOv2 models trained both with and without registers, but no CLIP model with registers exists publicly, so a before/after comparison is only possible for DINOv2.
- We load the models from Meta's official repository through PyTorch Hub instead of HuggingFace. These are the same weights from the original source.
- For Part A we use Imagenette validation images (object-centred photos, similar to the paper's figures), since the DINOv2 models expect larger images than Tiny-ImageNet's 64x64.

**Differences from the paper's setup**

- **Model sizes in Part A:** The paper's Fig 4c compares DINOv2 models trained separately at each size. Meta's *released* ViT-S/B/L were distilled from ViT-g, so the size trend in our Part A is measured on distilled models. This may differ from the paper's trend.
- **Images:** 256 Imagenette validation images at 224x224 (16x16 patches), not the paper's image set or resolution.
- **Outlier cutoff:** a fixed relative rule (3 × the model's median norm) instead of a hand-picked value per model. For ViT-g we also report the paper's cutoff of 150.
- **Part B scale:** ViTs with 5–86M parameters trained from scratch on 64x64 images for tens of epochs. The paper trains ViT-B/L on ImageNet-22k or larger datasets for much longer. The paper reports that artifacts only appear in large, long-trained models (Fig 4), so our small models may show no artifacts at all. This is what our Week 4 experiment tests.

## 8. Next steps

- **Week 4:** full 50-epoch runs, ViT-Tiny with 0 vs 4 registers (accuracy, norm histograms, attention maps). Then the scaling experiment: Tiny / Small / Base, with vs without registers.
- **Week 5:** FastAPI deployment, final report and presentation.

## 9. Provenance

See [PROVENANCE.md](PROVENANCE.md) for a file-by-file record of what was written for this project, what was adapted, and what was reused.

## References

- Darcet et al. *Vision Transformers Need Registers.* ICLR 2024.
- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. Code and weights: https://github.com/facebookresearch/dinov2
- Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- Touvron et al. *Training data-efficient image transformers & distillation through attention (DeiT).* ICML 2021.
- Le & Yang. *Tiny ImageNet Visual Recognition Challenge.* CS231N, 2015.
- Imagenette: https://github.com/fastai/imagenette
