Our models: analysed on the same 2,000 Tiny-ImageNet validation images. Norms are taken before the final LayerNorm; an outlier is a patch token whose norm is more than 3x the median norm of the same model (same rule as Part A).

| Model | Parameters | Registers | Val accuracy | Median norm | Largest / median | % outliers | Peak largest/median over blocks | Patch attention to registers (uniform) |
|---|---|---|---|---|---|---|---|---|
| ViT-Tiny | 5.4M | 0 | 38.99% | 37.6 | 2.09x | 0.00% | 2.31x | n/a |
| ViT-Tiny | 5.4M | 4 | 37.86% | 40.2 | 1.84x | 0.00% | 2.26x | 4.1% (5.8%) |
| ViT-Small | 21.5M | 0 | 33.35% | 442.2 | 1.67x | 0.00% | 1.85x | n/a |
| ViT-Small | 21.5M | 4 | 30.60% | 646.1 | 1.98x | 0.00% | 2.36x | 1.5% (5.8%) |

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
