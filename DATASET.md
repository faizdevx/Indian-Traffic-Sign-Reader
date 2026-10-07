# Dataset provenance

**Dataset used:** `kannanwisen/Indian-Traffic-Sign-Classification` (Hugging Face). Retrieved 2026-10-07.

> **Read this first.** This dataset was chosen because it was the only candidate that was
> both reachable and usable for classification. It does **not** meet a strong provenance
> bar, and the project must not be described as using "real-world" or "licensed" data
> without these caveats:
> * The dataset card is empty (only a licence header). There is no documented publisher,
>   collection method, annotation process or contributor information.
> * The metadata tag says `cc-by-4.0`, but a manual look at a 48-image sample found
>   images carrying visible stock-photo watermarks ("shutterstock", "alamy stock photo")
>   and clean vector-style sign illustrations. Such images are unlikely to be
>   CC-BY-4.0 licensable by an anonymous uploader, so **the licence tag is not supported
>   by evidence**. Treat usage as unverified; do not redistribute the images.
> * It is a mix of image types (see 4), not a uniform set of phone photographs.
> * The test split itself contains watermarked stock images (a `dreamstime.com` mark is visible
>   in the evaluation error-example figure), so reported test accuracy partly measures
>   stock-image classification.

## 1. Verified facts

| Field | Value | How verified |
|---|---|---|
| Name | Indian-Traffic-Sign-Classification | HF API |
| Publisher | user `kannanwisen` on Hugging Face (individual; no affiliation stated) | HF API (`author`) |
| Source URL | https://huggingface.co/datasets/kannanwisen/Indian-Traffic-Sign-Classification | |
| Revision | `89ba50c970706109f500449354c142cd28da6a6b` (last modified 2023-02-09) | HF API (`sha`) |
| Access | Public, ungated, direct download of one zip | HF API (`gated: false`) |
| Licence | Tag `cc-by-4.0` only; no card text, no attribution requirement spelled out; **claim not corroborated** | HF API / README |
| File | `Indian-Traffic-Sign-Classification.zip`, 1,211,757,927 bytes, sha256 `8754fcd5d55380d75c3d361e5210813a98a4bcd4593591c04e251357cd5e9a6e` | downloaded, hashed |
| Contents | 5,726 JPEG files, `Dataset/{train,test}/<CLASS>/<file>.jpg`; no annotation or metadata files | listed from the zip |
| Official split | train 4,438 / test 1,288 (as published) | counted |
| Classes | 85 (folder names as published, e.g. `SPEED_LIMIT_40`, `NO_ENTRY`); never renamed | counted |
| Annotation format | ImageFolder; image-level labels only; no boxes, no contributor/session IDs | listed |
| Geographic coverage / capture conditions | **not documented** by the publisher | no source |

Everything else below is computed from the files by `scripts/inspect_dataset.py` and
`scripts/prepare_dataset.py` (`reports/dataset_report.json`, `data/processed/split_info.json`).

## 2. Why it was considered, and what else was ruled out

* Datacluster Labs (the brief's candidate): public repo has only a README and 17 unlabeled
  sample images; the full set requires contacting the publisher. Its official Hugging Face
  sample (`Dataclusterlabspvtltd/Indian_Traffic_Sign_Image_Dataset`) is under 1,000 images,
  Pascal-VOC detection XML, and states "evaluation purposes only; commercial use and
  redistribution not permitted". Not used.
* IEEE DataPort IRTSD-Datasetv1 (5,141 phone-captured images, 37 classes, detection): needs an
  IEEE login; not accessible here. Not used.
* Zenodo/Mendeley/GitHub searches found no downloadable Indian traffic-sign classification set.
* Whether the chosen set derives from another publisher's data is **unknown**.

## 3. Findings from inspection (computed)

| Check | Result |
|---|---|
| Image files / corrupted | 5,726 / 0 |
| Exact duplicate files (SHA-256) | 2,811 files in 1,284 sets (49% of all files) |
| Duplicate sets spanning official train **and** test | 534 sets: the official test set is contaminated with training images |
| Duplicate sets with conflicting labels | 22 sets (label noise) |
| Near-duplicates (dHash Hamming <= 4) | groups of >1 image cover 4,046 images; in the cleaned data 65 groups mix classes (likely hash false positives, so the threshold is a heuristic) |
| Class counts (raw, 85 classes) | min 3, max 323, max/min ratio about 108 |
| Image size | median 50x50 px; max 3456x4608 |

## 4. Heterogeneity (observed)

Of the 3,705 images kept after cleaning, 2,001 are at most 64x64 px (tiny sign crops),
489 are over 1000 px on a side (wide street photographs where the sign is a small part of
the frame, so the whole-image label is a weak target), and the rest are mid-sized images
(web images, some watermarked, some illustrations). This 48-image visual check was not an exhaustive audit.

## 5. Cleaning and split (exact, as implemented and run)

1. Removed 44 images belonging to duplicate sets whose copies carry different labels.
2. Removed 1,505 exact duplicates, keeping one copy; if a set spans splits, the **test copy
   is kept** and train copies are dropped (the official test set is never altered).
3. Grouped near-duplicates (dHash, Hamming <= 4).
4. Official train/test split preserved. A validation set was carved from official train with
   5-fold `StratifiedGroupKFold` (one fold) grouped by near-duplicate clusters.
5. Removed 472 train/val images that share a near-duplicate group **and the same label**
   with a test image (213 groups spanned train/test). Cross-label matches were left in place.

Resulting data: **3,705 images, 85 classes: train 2,055 / val 458 / test 1,192** (test is
32% of the data because the official test split was kept intact while training data was
de-duplicated).

## 6. Known limitations

* **Severe imbalance after cleaning:** train class counts range from 1 to 143 (median 15);
  29 classes have fewer than 10 training images. This justifies the class-weighted loss used
  in training, and makes macro metrics noisy for rare classes.
* Five classes (`COMPULSARY_AHEAD`, `SPEED_LIMIT_50`, `SPEED_LIMIT_70`, `TRAFFIC_SIGNAL`,
  `T_INTERSECTION`) have **no validation images**; 26 classes have fewer than 5 test images.
* Residual near-duplicate leakage may remain beyond the dHash threshold; train/test
  independence is improved, not guaranteed.
* Source mix (watermarked web images, illustrations, tiny crops, wide scenes) means results
  do not measure performance on phone photographs taken on Indian roads.
* No contributor/session IDs, so grouping relies on image similarity only.
* Licence and provenance are unverified (see top).

## 7. Preprocessing at load time

Decode, convert to RGB (JPEGs larger than the model input are decoded at reduced scale via
PIL draft mode), resize directly to 96x96 (aspect ratio not preserved; most images are
50x50, so many are *up*sampled), scale to [0,1], normalise with ImageNet mean/std. Training
adds small affine (<=10 deg, 5% shift, 0.9-1.1 scale), brightness/contrast/saturation jitter,
and mild blur (p=0.2). No horizontal flip (direction-bearing signs), no hue shift, no
random-resized-crop.
