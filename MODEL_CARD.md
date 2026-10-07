# Model card: Indian Traffic Sign Reader (CNN-from-scratch and ResNet18 classifiers)

**Status: no model has been trained.** This card describes the intended design; every
data-dependent field is "not established" until a real run exists. Do not cite this repo for
accuracy.

* **Purpose:** compare a small CNN trained from scratch with an ImageNet-pretrained ResNet18 on
  Indian traffic-sign classification (research/portfolio benchmark).
* **Intended use:** studying the CNN-vs-transfer-learning trade-off, reproducible-pipeline demo.
* **Not intended for:** driver assistance, autonomous driving, enforcement, or any
  safety-relevant decision.
* **Training / evaluation data:** not established (see DATASET.md). Splits will be class-stratified
  and near-duplicate-grouped; the test split is used once, for the final comparison.
* **Metrics:** none. The evaluation script will report accuracy, macro/weighted F1, macro
  precision/recall, top-1/top-3, per-class scores, confusion matrices and ECE.
* **Limitations / failure modes to check once trained:** rare classes, night/glare/rain,
  occlusion and damaged signs, unusual viewpoints, resize distortion of non-square crops, visually
  similar signs (e.g. different speed limits), and inputs that are not signs.
* **Geographic limits:** coverage is whatever the chosen dataset covers; generalisation to other
  regions, sign designs or cameras is unknown.
* **Environmental limits:** performance in conditions absent from the training data is unknown.
* **Confidence interpretation:** the reported confidence is the softmax probability. It is not
  the probability of being correct; neural networks are often over-confident. The pipeline
  offers temperature scaling fitted on the validation split and reports ECE before and after, but
  calibration on one dataset does not guarantee calibration on new images. The model cannot say
  "not a sign" or "unseen class".
