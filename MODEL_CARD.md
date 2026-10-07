# Model card: Indian Traffic Sign Reader

Three PyTorch classifiers trained and evaluated in this repo: `cnn` (from scratch),
`resnet18_frozen` and `resnet18_finetune` (ImageNet-pretrained). The API/UI serve `cnn` and
`resnet18_finetune`. Checkpoints are not committed; regenerate with the README commands.

* **Purpose:** compare training from scratch with transfer learning on Indian traffic-sign classification (research/portfolio benchmark).
* **Intended use:** studying the comparison and the pipeline; demonstration.
* **Not intended for:** driver assistance, autonomous driving, enforcement, or any safety-relevant decision.
* **Training data:** `kannanwisen/Indian-Traffic-Sign-Classification` after cleaning: 2,055 train and 458 validation images, 85 classes (see DATASET.md). Provenance and licence are **unverified**; the set mixes watermarked web/stock images, illustrations, tiny crops and street scenes.
* **Evaluation data:** 1,192 images from the official test split (duplicates of training images removed, test images otherwise untouched). Read once, after training.
* **Metrics (test, single run, seed 42):**

| Model | Accuracy | Macro F1 | Top-3 | ECE raw -> temp-scaled |
|---|---:|---:|---:|---|
| cnn | 0.3674 | 0.2791 | 0.5982 | 0.0912 -> 0.0343 |
| resnet18_frozen | 0.4815 | 0.3738 | 0.7131 | 0.0929 -> 0.0794 |
| resnet18_finetune | 0.8784 | 0.8111 | 0.9497 | 0.0312 -> 0.0302 |

  The CNN and frozen runs hit the 30-epoch cap while still improving (under-trained); the fine-tuned model overfits (train acc 0.991 vs val 0.924). See the README for the full table and analysis.
* **Known failure modes (observed):** very rare classes (1 to 3 training images; SPEED_LIMIT_20, HANDCART_PROHIBITED and FERRY scored F1 0.00 for the fine-tuned model); look-alike signs (speed limits 30/50/70/80; CROSS_ROAD vs Y_INTERSECTION; COMPULSARY_AHEAD_OR_TURN_RIGHT vs COMPULSARY_AHEAD; HUMP_OR_ROUGH_ROAD vs MEN_AT_WORK); signs obscured by watermark overlays. 23 of the fine-tuned model's 145 test errors had confidence >= 0.9.
* **Geographic / environmental limits:** the dataset does not document where, when or how images were captured. Performance on phone photos from Indian roads, at night, in rain or glare, or with occlusion is **not measured**.
* **Confidence interpretation:** reported confidence is the softmax probability, not the probability of being correct; for the fine-tuned model mean confidence was 0.933 on correct and 0.607 on wrong predictions, yet some confident predictions were wrong. Temperature scaling (fitted on validation) barely changes its calibration (ECE 0.0312 -> 0.0302). The model cannot answer "not a sign" or "unseen class".
