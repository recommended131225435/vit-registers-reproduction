Validation accuracy after the last epoch (we do not pick the best epoch, because the validation set is also our test set).

| Model | Registers | Parameters | Epochs | Val accuracy | Val loss | Training time (min) |
|---|---|---|---|---|---|---|
| ViT-Tiny | 0 | 5,427,080 | 50 | **38.99%** | 3.537 | 75 |
| ViT-Tiny | 4 | 5,427,848 | 50 | **37.86%** | 3.592 | 75 |

Effect of adding registers: **-1.13 percentage points**. Paper (Table 2a, ImageNet top-1): DeiT-III 84.7 → 84.7, OpenCLIP 78.2 → 78.1, DINOv2 84.3 → 84.8, i.e. registers do not hurt accuracy. Our absolute accuracies are not comparable with the paper's (different dataset, much smaller model, shorter training); only the with/without-registers difference is. Each setting was trained once (seed 0), so differences of a few tenths of a point may be random variation.

![Full training runs](results/figures/full_training_runs.png)
