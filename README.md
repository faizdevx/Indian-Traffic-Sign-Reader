# Indian Traffic Sign Reader

PyTorch-based Indian traffic sign classification benchmark comparing a CNN trained from scratch against ResNet18 transfer learning on Indian traffic-sign imagery of mixed and partly unverified provenance (see Dataset).

> **Read before citing any number.** The only usable dataset I could obtain
> (`kannanwisen/Indian-Traffic-Sign-Classification`) has an empty dataset card, an anonymous
> uploader, a `cc-by-4.0` tag that is contradicted by watermarked stock-photo images in it, and
> a mix of tiny crops, web images, illustrations and street scenes. These results measure
> classification of *this* mixed set. They are **not** evidence of performance on phone
> photographs taken on Indian roads. Details: [DATASET.md](DATASET.md).

## Problem

Recognise a traffic sign from an image as single-label classification into 85 Indian sign classes.

## Dataset

* **Used:** `kannanwisen/Indian-Traffic-Sign-Classification`, Hugging Face revision `89ba50c9...`, retrieved 2026-10-07, public download, licence tag `cc-by-4.0` (**not corroborated**).
* **Published:** 5,726 JPEGs, 85 classes (folder names as published), official train 4,438 / test 1,288.
* **Cleaning (computed, not assumed):** 49% of files were exact duplicates, and 534 duplicate sets spanned the official train and test splits (test contamination); 22 duplicate sets had conflicting labels. After removing conflicting and duplicate images (keeping test copies) and 472 train/val images that near-duplicate a same-label test image: **3,705 images, train 2,055 / val 458 / test 1,192**. The official test split was never altered beyond removing its duplicates.
* **Imbalance:** train class counts range from 1 to 143 (median 15); 29 classes have under 10 training images; 5 classes have no validation images.
* Full provenance, findings and limitations: [DATASET.md](DATASET.md). Data is not committed.

## Models

* **CNN from scratch** (`src/model_cnn.py`): 4 x [Conv3x3-BN-ReLU-MaxPool], 32-64-128-256 channels, global average pooling, dropout, linear layer producing logits (410,741 parameters with 85 classes).
* **ResNet18** (`src/model_resnet.py`): torchvision ResNet18 with ImageNet-1k weights (`ResNet18_Weights.IMAGENET1K_V1`) and a new 85-way `fc`. *frozen*: only `fc` trained, backbone and BatchNorm statistics frozen; *finetune*: all layers trained (11,220,117 parameters).

Why: the pair isolates what pretraining buys on a small, imbalanced, domain-mixed dataset.

## Data pipeline

```
Image file -> decode (corrupt files dropped) -> RGB -> resize 96x96 -> [train only: affine,
colour jitter, blur] -> tensor + ImageNet normalisation -> DataLoader (shuffle in train)
```

No horizontal flip (direction-bearing signs), no hue jitter (colour carries class). Splits keep the official test set and use near-duplicate-grouped, class-stratified folds for validation.

## Training (exactly what was run)

AdamW, cosine schedule over 30 epochs, weight decay 1e-4, batch 64, early stopping on validation loss (patience 7), best-validation checkpoint, seed 42, CPU (4 cores), torch 2.14.1. LR 1e-3 (CNN) and 3e-4 (both ResNet18 runs). Loss: cross-entropy with inverse-frequency class weights (`--class-weights auto`, enabled because the observed train max/min ratio was 143, above the threshold of 3). Hyperparameters were not tuned; single run per model.

| Run | Epochs run | Best epoch | Train acc / Val acc at best epoch |
|---|---:|---:|---|
| cnn | 30 (cap) | 29 | 0.410 / 0.389 |
| resnet18_frozen | 30 (cap) | 30 | 0.631 / 0.563 |
| resnet18_finetune | 22 (early stop) | 15 | 0.991 / 0.924 |

## Results

Test split (1,192 images, evaluated once after training; temperature scaling fitted on validation only).

<!-- RESULTS:BEGIN -->
| Model | Params | Test Accuracy | Macro F1 | Top-3 Accuracy | CPU Inference (ms, median) | GPU Inference |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 410,741 | 0.3674 | 0.2791 | 0.5982 | 2.12 | not measured |
| resnet18_finetune | 11,220,117 | 0.8784 | 0.8111 | 0.9497 | 5.96 | not measured |
| resnet18_frozen | 11,220,117 | 0.4815 | 0.3738 | 0.7131 | 5.37 | not measured |

Weakest classes (test, by F1):
- cnn: lowest-F1 classes: COMPULSARY_SOUND_HORN (F1 0.00, n=12), COMPULSARY_TURN_LEFT (F1 0.00, n=12), CATTLE (F1 0.00, n=11), CYCLE_CROSSING (F1 0.00, n=11), OVERTAKING_PROHIBITED (F1 0.00, n=9)
- resnet18_finetune: lowest-F1 classes: SPEED_LIMIT_20 (F1 0.00, n=3), HANDCART_PROHIBITED (F1 0.00, n=2), FERRY (F1 0.00, n=1), COMPULSARY_AHEAD_OR_TURN_RIGHT (F1 0.40, n=11), RIGHT_HAIR_PIN_BEND (F1 0.46, n=5)
- resnet18_frozen: lowest-F1 classes: COMPULSARY_TURN_LEFT (F1 0.00, n=12), SPEED_LIMIT_15 (F1 0.00, n=10), HEIGHT_LIMIT (F1 0.00, n=8), PASS_EITHER_SIDE (F1 0.00, n=8), STAGGERED_INTERSECTION (F1 0.00, n=6)

Calibration (T fitted on validation):
- cnn: ECE raw 0.0912 -> temperature-scaled 0.0343 (T=0.780)
- resnet18_finetune: ECE raw 0.0312 -> temperature-scaled 0.0302 (T=1.064)
- resnet18_frozen: ECE raw 0.0929 -> temperature-scaled 0.0794 (T=0.700)
<!-- RESULTS:END -->

Other test metrics (from `reports/metrics/*_metrics.json`):

| Model | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| cnn | 0.3176 | 0.3197 | 0.2791 | 0.3604 |
| resnet18_frozen | 0.4075 | 0.4007 | 0.3738 | 0.4707 |
| resnet18_finetune | 0.8385 | 0.8213 | 0.8111 | 0.8758 |

Latency: batch size 1, 96x96 input, 10 warm-up + 50 timed forward passes, model compute only, 4-core CPU. GPU: not measured (no GPU available).

How to read this honestly:
* Fine-tuned ResNet18 is far better than the CNN and than the frozen backbone on this split. That ordering is large enough to be believable from a single seed; the exact gaps are not.
* The CNN and the frozen ResNet18 both ran to the 30-epoch cap with validation still improving (CNN train acc only 0.41), so they are **under-trained**; the comparison with fine-tuning is confounded by the epoch budget and untuned learning rates. A longer run could narrow the gap; I did not run one.
* Fine-tuned ResNet18 overfits (train acc 0.991, train loss 0.054 vs val acc 0.924, val loss 0.809 at the best epoch).
* 0.878 test accuracy is likely optimistic for real roads: the test split is mostly stock/web/tiny-crop images, and near-duplicates beyond my hash threshold may remain. Metrics for rare classes rest on 1 to 3 test images.

## Error analysis (fine-tuned ResNet18, 145 of 1,192 test images wrong)

* **Weakest classes by F1:** SPEED_LIMIT_20 (0.00, n=3), HANDCART_PROHIBITED (0.00, n=2), FERRY (0.00, n=1), COMPULSARY_AHEAD_OR_TURN_RIGHT (0.40, n=11), RIGHT_HAIR_PIN_BEND (0.46, n=5). The first three are the rarest classes in training (1 to 3 images).
* **Most frequent confusions (true -> predicted):** COMPULSARY_AHEAD_OR_TURN_RIGHT -> COMPULSARY_AHEAD (8), CROSS_ROAD -> Y_INTERSECTION (8), HUMP_OR_ROUGH_ROAD -> MEN_AT_WORK (7), and a cluster among SPEED_LIMIT_30/50/70/80 (4-5 each): similar-looking signs that differ in small details at low resolution.
* **Confidence:** mean confidence is 0.933 on correct and 0.607 on wrong predictions, but 23 wrong predictions have confidence >= 0.9. High-confidence errors in `reports/figures/error_examples_resnet18_finetune.png` include a dark image whose sign is covered by a watermark overlay and TONGA_PROHIBITED -> BULLOCK_PROHIBITED (visually near-identical animal-cart signs), so some are data/label-quality problems rather than model failures.
* **By image type (test):** tiny crops (<= 64 px) 0.896 (833 images), mid-size 0.831 (320), very large photos 0.897 (only 39, noisy).
* **Calibration** (ECE on test, temperature fitted on validation): cnn 0.0912 -> 0.0343 (T=0.780); resnet18_finetune 0.0312 -> 0.0302 (T=1.064); resnet18_frozen 0.0929 -> 0.0794 (T=0.700). For the best model temperature scaling changes almost nothing. Figures in `reports/figures/`: confusion matrices, reliability diagrams, training curves, class distribution. The `error_examples_*.png` grids are generated by `evaluate_models.py` but deliberately not committed, because they embed dataset images (some visibly watermarked) whose licence is unverified.

## Demo

* `GET /` web UI: upload an image, choose CNN or ResNet18, see prediction, top-3, confidence, latency.
* `GET /health`: service status and which checkpoints exist.
* `POST /predict` (multipart: `file`, `model` = `cnn|resnet18`, optional `top_k`): JSON with `prediction`, `confidence` (raw softmax probability), optional `calibrated_confidence`, `top_k`. Errors: 400 invalid/corrupt image or model, 413 too large, 415 unsupported type, 422 malformed request, 503 checkpoint missing, 500 checkpoint without class mapping.

`resnet18` in the API/UI serves `models/resnet18_finetune_best.pt`. Checkpoints are not committed (regenerate with the commands below).

## Installation

Verified on Python 3.13.16 / Linux (CPU).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                   # unit tests; integration tests need prepared data
```

Dataset setup (the zip is ~1.2 GB). These are the commands that were run:

```bash
curl -L -o ds.zip https://huggingface.co/datasets/kannanwisen/Indian-Traffic-Sign-Classification/resolve/main/Indian-Traffic-Sign-Classification.zip
python3 -I -c "import zipfile; zipfile.ZipFile('ds.zip').extractall('extracted')"
mkdir -p data/raw/kannanwisen_indian_traffic_sign && cp -r extracted/Dataset/. data/raw/kannanwisen_indian_traffic_sign/
python scripts/inspect_dataset.py --source data/raw/kannanwisen_indian_traffic_sign
python scripts/prepare_dataset.py --source data/raw/kannanwisen_indian_traffic_sign --name "kannanwisen/Indian-Traffic-Sign-Classification"
```

Read the licence caveat in DATASET.md before using this data.

## Training (exactly what was run)

AdamW, cosine schedule over 30 epochs, weight decay 1e-4, batch 64, early stopping on validation loss (patience 7), best-validation checkpoint, seed 42, CPU (4 cores), torch 2.14.1. LR 1e-3 (CNN) and 3e-4 (both ResNet18 runs). Loss: cross-entropy with inverse-frequency class weights (`--class-weights auto`, enabled because the observed train max/min ratio was 143, above the threshold of 3). Hyperparameters were not tuned; single run per model.

| Run | Epochs run | Best epoch | Train acc / Val acc at best epoch |
|---|---:|---:|---|
| cnn | 30 (cap) | 29 | 0.410 / 0.389 |
| resnet18_frozen | 30 (cap) | 30 | 0.631 / 0.563 |
| resnet18_finetune | 22 (early stop) | 15 | 0.991 / 0.924 |

## Results

Test split (1,192 images, evaluated once after training; temperature scaling fitted on validation only).

<!-- RESULTS:BEGIN -->
| Model | Params | Test Accuracy | Macro F1 | Top-3 Accuracy | CPU Inference (ms, median) | GPU Inference |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 410,741 | 0.3674 | 0.2791 | 0.5982 | 2.12 | not measured |
| resnet18_finetune | 11,220,117 | 0.8784 | 0.8111 | 0.9497 | 5.96 | not measured |
| resnet18_frozen | 11,220,117 | 0.4815 | 0.3738 | 0.7131 | 5.37 | not measured |

Weakest classes (test, by F1):
- cnn: lowest-F1 classes: COMPULSARY_SOUND_HORN (F1 0.00, n=12), COMPULSARY_TURN_LEFT (F1 0.00, n=12), CATTLE (F1 0.00, n=11), CYCLE_CROSSING (F1 0.00, n=11), OVERTAKING_PROHIBITED (F1 0.00, n=9)
- resnet18_finetune: lowest-F1 classes: SPEED_LIMIT_20 (F1 0.00, n=3), HANDCART_PROHIBITED (F1 0.00, n=2), FERRY (F1 0.00, n=1), COMPULSARY_AHEAD_OR_TURN_RIGHT (F1 0.40, n=11), RIGHT_HAIR_PIN_BEND (F1 0.46, n=5)
- resnet18_frozen: lowest-F1 classes: COMPULSARY_TURN_LEFT (F1 0.00, n=12), SPEED_LIMIT_15 (F1 0.00, n=10), HEIGHT_LIMIT (F1 0.00, n=8), PASS_EITHER_SIDE (F1 0.00, n=8), STAGGERED_INTERSECTION (F1 0.00, n=6)

Calibration (T fitted on validation):
- cnn: ECE raw 0.0912 -> temperature-scaled 0.0343 (T=0.780)
- resnet18_finetune: ECE raw 0.0312 -> temperature-scaled 0.0302 (T=1.064)
- resnet18_frozen: ECE raw 0.0929 -> temperature-scaled 0.0794 (T=0.700)
<!-- RESULTS:END -->

Other test metrics (from `reports/metrics/*_metrics.json`):

| Model | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| cnn | 0.3176 | 0.3197 | 0.2791 | 0.3604 |
| resnet18_frozen | 0.4075 | 0.4007 | 0.3738 | 0.4707 |
| resnet18_finetune | 0.8385 | 0.8213 | 0.8111 | 0.8758 |

Latency: batch size 1, 96x96 input, 10 warm-up + 50 timed forward passes, model compute only, 4-core CPU. GPU: not measured (no GPU available).

How to read this honestly:
* Fine-tuned ResNet18 is far better than the CNN and than the frozen backbone on this split. That ordering is large enough to be believable from a single seed; the exact gaps are not.
* The CNN and the frozen ResNet18 both ran to the 30-epoch cap with validation still improving (CNN train acc only 0.41), so they are **under-trained**; the comparison with fine-tuning is confounded by the epoch budget and untuned learning rates. A longer run could narrow the gap; I did not run one.
* Fine-tuned ResNet18 overfits (train acc 0.991, train loss 0.054 vs val acc 0.924, val loss 0.809 at the best epoch).
* 0.878 test accuracy is likely optimistic for real roads: the test split is mostly stock/web/tiny-crop images, and near-duplicates beyond my hash threshold may remain. Metrics for rare classes rest on 1 to 3 test images.

## Error analysis (fine-tuned ResNet18, 145 of 1,192 test images wrong)

* **Weakest classes by F1:** SPEED_LIMIT_20 (0.00, n=3), HANDCART_PROHIBITED (0.00, n=2), FERRY (0.00, n=1), COMPULSARY_AHEAD_OR_TURN_RIGHT (0.40, n=11), RIGHT_HAIR_PIN_BEND (0.46, n=5). The first three are the rarest classes in training (1 to 3 images).
* **Most frequent confusions (true -> predicted):** COMPULSARY_AHEAD_OR_TURN_RIGHT -> COMPULSARY_AHEAD (8), CROSS_ROAD -> Y_INTERSECTION (8), HUMP_OR_ROUGH_ROAD -> MEN_AT_WORK (7), and a cluster among SPEED_LIMIT_30/50/70/80 (4-5 each): similar-looking signs that differ in small details at low resolution.
* **Confidence:** mean confidence is 0.933 on correct and 0.607 on wrong predictions, but 23 wrong predictions have confidence >= 0.9. High-confidence errors in `reports/figures/error_examples_resnet18_finetune.png` include a dark image whose sign is covered by a watermark overlay and TONGA_PROHIBITED -> BULLOCK_PROHIBITED (visually near-identical animal-cart signs), so some are data/label-quality problems rather than model failures.
* **By image type (test):** tiny crops (<= 64 px) 0.896 (833 images), mid-size 0.831 (320), very large photos 0.897 (only 39, noisy).
* **Calibration** (ECE on test, temperature fitted on validation): cnn 0.0912 -> 0.0343 (T=0.780); resnet18_finetune 0.0312 -> 0.0302 (T=1.064); resnet18_frozen 0.0929 -> 0.0794 (T=0.700). For the best model temperature scaling changes almost nothing. Figures in `reports/figures/`: confusion matrices, reliability diagrams, training curves, class distribution. The `error_examples_*.png` grids are generated by `evaluate_models.py` but deliberately not committed, because they embed dataset images (some visibly watermarked) whose licence is unverified.

## Demo

* `GET /` web UI: upload an image, choose CNN or ResNet18, see prediction, top-3, confidence, latency.
* `GET /health`: service status and which checkpoints exist.
* `POST /predict` (multipart: `file`, `model` = `cnn|resnet18`, optional `top_k`): JSON with `prediction`, `confidence` (raw softmax probability), optional `calibrated_confidence`, `top_k`. Errors: 400 invalid/corrupt image or model, 413 too large, 415 unsupported type, 422 malformed request, 503 checkpoint missing, 500 checkpoint without class mapping.

`resnet18` in the API/UI serves `models/resnet18_finetune_best.pt`. Checkpoints are not committed (regenerate with the commands below).

## Installation

Verified on Python 3.13.16 / Linux (CPU).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                   # unit tests; integration tests need prepared data
```

Dataset setup (the zip is ~1.2 GB; extract so that `Dataset/train` and `Dataset/test` exist, then point the scripts at it):

```bash
curl -L -o ds.zip https://huggingface.co/datasets/kannanwisen/Indian-Traffic-Sign-Classification/resolve/main/Indian-Traffic-Sign-Classification.zip
mkdir -p data/raw/kannanwisen_indian_traffic_sign && unzip ds.zip 'Dataset/*' -d /tmp/itsr && cp -r /tmp/itsr/Dataset/. data/raw/kannanwisen_indian_traffic_sign/
python scripts/inspect_dataset.py --source data/raw/kannanwisen_indian_traffic_sign
python scripts/prepare_dataset.py --source data/raw/kannanwisen_indian_traffic_sign --name "kannanwisen/Indian-Traffic-Sign-Classification"
```

(The download and extraction were performed with equivalent commands; the `unzip`/`cp` lines above are the same operation written for the shell.) Please read the licence caveat in DATASET.md before using this data.

## Training

```bash
python scripts/train_cnn.py --class-weights auto --num-workers 3
python scripts/train_resnet.py --mode frozen   --class-weights auto --num-workers 3
python scripts/train_resnet.py --mode finetune --class-weights auto --num-workers 3
```

`--smoke-test` runs 2 epochs on a tiny slice of the real data. ResNet18 downloads ImageNet weights on first use; if that fails the script stops (`--no-pretrained` trains from random init and is recorded as not being transfer learning).

## Evaluation

```bash
python scripts/evaluate_models.py   # reads the test split once; rewrites the results block above
```

## API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Command-line inference: `python -m src.inference path/to/image.jpg --model resnet18`.

## Docker

A `Dockerfile` is included but has **not been built or tested**; no instructions are given.

## Limitations

Class imbalance (1 to 143 training images per class), difficult lighting, occlusion, domain shift (stock/web images versus phone photos), camera differences, and unseen sign classes (a softmax classifier always outputs one of its 85 known classes). The near-duplicate threshold (dHash Hamming <= 4) is a heuristic, some of the weakest-class conclusions rest on 1 to 3 test images, results are single-seed without confidence intervals, and images are resized to a square (most source images are 50x50, so many are upsampled).

## Responsible use

This is a research/portfolio classifier. It must not be treated as a safety-critical autonomous-driving component without extensive validation.
