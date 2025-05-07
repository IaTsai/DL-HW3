# Mask R-CNN Cell Instance Segmentation (DL-HW3)

This project implements a customizable Mask R-CNN training pipeline for cell instance segmentation, supporting multiple backbones (e.g., ResNet50, ResNeXt50, EfficientNet-B0/B1) with optional attention modules (SE/CBAM). The final model is trained and evaluated on a COCO-style annotated dataset, and results are saved in standard formats for leaderboard submission.

---

## Project Structure Overview

| File | Description |
| :-- | :-- |
| `datasets.py` | Custom PyTorch `Dataset` class to load images and per-instance cell masks for training/validation. Supports multi-class `.tif` masks. |
| `train.py` | Main training script. Supports AMP, FPN, attention modules, learning rate warmup, early stopping, and COCOEval-based mAP evaluation. |
| `transforms.py` | Defines torchvision transforms for data augmentation during training and resizing/normalization during validation. |
| `utils.py` | Utility functions including mask preprocessing and instance mask separation from class-wise masks. |
| `environment.yml` | Conda environment specification. Use to reproduce the exact dependency setup. |
| `Gen_test-results.py` | Inference script to load the best model checkpoint and generate `test-results.json` for leaderboard submission. |
| `val_gt.json` | Ground truth annotations for validation subset, automatically extracted from `train_gt.json` during training. Used in local COCOEval. |


---

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate dl-hw3
```


---

## Training

### 2. Run Training

```bash
python train.py
```

Key configurations inside `train.py`:

* Backbone selection: `resnet50`, `resnext50_32x4d`, `efficientnet_b0`, etc.
* AMP mixed precision enabled
* Learning rate warmup for first 3 epochs
* Early stopping after 30 epochs of no mAP improvement
* COCO evaluation using `val_gt.json`
* Best model saved to `saved_models/`

---

## Inference and Submission

### 3. Generate Prediction File for Submission

```bash
python Gen_test-results.py
```

This will produce:

```bash
test-results.json
```

> Upload `test-results.json` to the course leaderboard submission system for scoring.

---

## Backbone Variants and Modules

Supported backbone configurations:

* `maskrcnn_resnet50_fpn_v2`
* `resnet50`, `resnet101`
* `resnext50_32x4d`
* `efficientnet_b0`, `efficientnet_b1`

Optional modules:

* `SELayer`: Squeeze-and-Excitation Channel Attention
* `CBAM`: Channel + Spatial Attention
* Both attention modules can be patched into backbone before training.

---

## Tips and Notes

* Ensure the output **test-results.json** follows the COCO result format. The following is an example:

```
[
    {
        'image_id': 1,
        'bbox': [95.22177124023438, 381.214111328125, 24.7103271484375, 25.109375],
        'score': 0.56789,
        'category_id': 1,
        'segmentation': {
            'size': [446, 512],
            'counts': 'PhY1f0W=1000O100O2N1O2N1N3N2M5Jgb_5'
        }
    },
    {
        'image_id': 1,
        'bbox': [304.9966735839844, 241.36700439453125, 45.23297119140625, 41.11309814453125],
        'score': 0.45678,
        'category_id': 1,
        'segmentation': {
            'size': [446, 512],
            'counts': 'cRU4d0Y=3N1O01O01O10O010O10O01O01O0001O0O100O101N1O2M3N2M5JjT^2'
        }
    }
    ...
]
```

* `val_gt.json` is automatically created on first run of `train.py` to enable subset evaluation using COCOEval.
* If your model causes OOM, reduce `BATCH_SIZE` or switch to lightweight backbones like `efficientnet_b0`.

---

## References

See the [`Report`](./report.pdf) for detailed model comparison, architecture choices, and ablation studies.

