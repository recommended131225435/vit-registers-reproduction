"""Reproduction of "Vision Transformers Need Registers" (Darcet et al., ICLR 2024).

Pipeline overview (one file per stage):

    data.py      1. Load and prepare images (Tiny-ImageNet)
    vit.py       2. The model: a Vision Transformer with optional register tokens
    train.py     3. Train the model and evaluate it on the validation set
    analysis.py  4. Measure what the paper measures: token norms, attention maps
    plots.py     5. Turn those measurements into figures
    report.py    6. Write the results into README.md
    utils.py        Small helpers shared by all stages (seed, device, ...)
"""
