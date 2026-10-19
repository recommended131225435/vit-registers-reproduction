Our models: analysed on the same 2,000 Tiny-ImageNet validation images. Norms are taken before the final LayerNorm; an outlier is a patch token whose norm is more than 3x the median norm of the same model (same rule as Part A).

| Model | Parameters | Registers | Peak LR | Val accuracy | Train accuracy | Median norm | Largest / median | % outliers | Peak largest/median over blocks | Patch attention to registers (uniform) |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT-Tiny | 5.4M | 0 | 0.00025 | 39.74% | 73.9% | 7.8 | 2.61x | 0.00% | 3.19x | n/a |
| ViT-Tiny | 5.4M | 4 | 0.00025 | 38.93% | 73.2% | 7.9 | 2.64x | 0.00% | 3.10x | 6.0% (5.8%) |
| ViT-Small | 21.5M | 0 | 0.00025 | 39.69% | 97.2% | 22.2 | 2.04x | 0.00% | 2.61x | n/a |
| ViT-Small | 21.5M | 4 | 0.00025 | 40.13% | 97.3% | 22.6 | 2.17x | 0.00% | 2.69x | 5.7% (5.8%) |
| ViT-Base | 85.4M | 0 | 0.00025 | 40.44% | 99.0% | 85.2 | 2.32x | 0.00% | 2.50x | n/a |
| ViT-Base | 85.4M | 4 | 0.00025 | 39.78% | 99.0% | 96.1 | 2.39x | 0.00% | 2.54x | 6.2% (5.8%) |

Runs not used in the figures (same recipe, other peak learning rate; see the discussion below):

| Model | Parameters | Registers | Peak LR | Val accuracy | Train accuracy | Median norm | Largest / median | % outliers | Peak largest/median over blocks | Patch attention to registers (uniform) |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT-Tiny | 5.4M | 0 | 0.001 | 38.99% | 91.6% | 37.6 | 2.09x | 0.00% | 2.31x | n/a |
| ViT-Tiny | 5.4M | 4 | 0.001 | 37.86% | 91.5% | 40.2 | 1.84x | 0.00% | 2.26x | 4.1% (5.8%) |
| ViT-Small | 21.5M | 0 | 0.001 | 33.35% | 89.8% | 442.2 | 1.67x | 0.00% | 1.85x | n/a |
| ViT-Small | 21.5M | 4 | 0.001 | 30.60% | 66.5% | 646.1 | 1.98x | 0.00% | 2.36x | 1.5% (5.8%) |
| ViT-Base | 85.4M | 0 | 0.001 | 8.59% | 8.9% | 11175.5 | 1.49x | 0.00% | 1.64x | n/a |
| ViT-Base | 85.4M | 4 | 0.001 | 10.27% | 10.8% | 7924.1 | 1.45x | 0.00% | 1.58x | 16.4% (5.8%) |

Meta's DINOv2 models (from Part A), on the same axis:

| Model | Parameters | % outliers (no reg / 4 reg) | Largest / median (no reg / 4 reg) |
|---|---|---|---|
| DINOv2 ViT-S/14 | ≈21M | 0.00% / 0.00% | 1.48x / 1.51x |
| DINOv2 ViT-B/14 | ≈86M | 0.20% / 0.00% | 4.68x / 1.56x |
| DINOv2 ViT-L/14 | ≈300M | 0.00% / 0.00% | 1.71x / 1.65x |
| DINOv2 ViT-g/14 | ≈1100M | 2.62% / 0.00% | 11.30x / 2.66x |

![Scaling](results/figures/experiment_scaling.png)

![Norms through the blocks](results/figures/experiment_layer_norms.png)

![Norm histograms](results/figures/experiment_norm_histograms.png)

![Accuracy and register use](results/figures/experiment_accuracy_registers.png)

![Attention maps](results/figures/experiment_attention_maps.png)

![Learning rate](results/figures/experiment_learning_rate.png)
