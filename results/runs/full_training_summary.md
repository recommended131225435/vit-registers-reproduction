Validation accuracy after the last epoch (we do not pick the best epoch, because the validation set is also our test set).

| Model | Augmentation | Registers | Train images | Epochs | Val accuracy | Train accuracy | Val loss (final / lowest) | Time (min) |
|---|---|---|---|---|---|---|---|---|
| ViT-Tiny | basic | 0 | 100,000 | 50 | **38.99%** | 91.6% | 3.537 / 3.225 | 75 |
| ViT-Tiny | basic | 4 | 100,000 | 50 | **37.86%** | 91.5% | 3.592 / 3.246 | 75 |

- Effect of adding registers (basic augmentation): **-1.13 points**.
- Paper (Table 2a, ImageNet top-1): DeiT-III 84.7 → 84.7, OpenCLIP 78.2 → 78.1, DINOv2 84.3 → 84.8, i.e. registers do not hurt accuracy. Our absolute accuracies are not comparable with the paper's (different dataset, much smaller model, shorter training); only the with/without-registers differences are. Each setting was trained once (seed 0), so differences of a few tenths of a point may be random variation.
- How to read the overfitting columns: a large gap between train and validation accuracy, and a final validation loss well above its lowest value, both mean the model is memorising the training images.

![Full training runs](results/figures/full_training_runs.png)
