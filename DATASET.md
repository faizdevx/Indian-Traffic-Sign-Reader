# Dataset provenance

**Status (retrieved/checked 2026-10-07): no dataset has been selected, downloaded or used.**
No model in this repository has been trained, and no metric exists. This file records what
was verified, what could not be verified, and why. Nothing below is an estimate.

## 1. Candidate named in the project brief: Datacluster Labs

| Field | Verified value | Source |
|---|---|---|
| Name | Indian Traffic Sign Image Dataset | repo README |
| Publisher | Datacluster Labs | repo README |
| Repository | https://github.com/datacluster-labs/Indian-Traffic-Sign-Image-Dataset | cloned 2026-10-07 |
| Stated size | "Approx. 6000 unique images" | README |
| Stated contributors / coverage | "2000+ unique users", "20+ cities across India" | README |
| Capture | mobile phones; day, evening, night; varied distances and viewpoints | README |
| Annotations | classification and detection; COCO, PASCAL VOC, YOLO formats | README |
| Access | **Full dataset requires contacting sales@datacluster.ai** (not freely downloadable) | README |
| License | **None stated** in the README; the cloned snapshot contains no LICENSE file | cloned file listing |
| What the public repo actually contains | `README.md` + 17 sample JPEGs (`sample_datasets/traffic_images/`), **no labels, no annotations** | cloned file listing |

Consequence: 17 unlabeled images cannot train or evaluate a classifier, and access to the
full data is gated by the publisher. I did not try to bypass that, scrape, or substitute data.
A Kaggle page by the same publisher was surfaced by web search with different headline
numbers ("2000+ images", "400+ contributors") than this README ("~6000", "2000+ users");
I could not open it (see 3), so the discrepancy is unresolved. Its README figures are the
publisher's own claims, not something I counted.

## 2. Alternatives found by search (NOT verified)

Only the search-result titles/snippets were seen. The hosts below were blocked by the
sandbox's network policy (HTTP 403 on CONNECT), so licence, counts, labels and provenance
could **not** be confirmed and none of these qualifies yet.

| Candidate | What the search snippet claimed | Open questions |
|---|---|---|
| Hugging Face `kannanwisen/Indian-Traffic-Sign-Classification` | CC-BY-4.0, 5,726 images, 85 classes | Who collected the images? Is it derived from the Datacluster data (which would conflict with a CC-BY claim)? |
| IEEE DataPort IRTSD-Datasetv1 | 5,141 images, 37 classes, 90+ cities, phone-captured; detection-oriented | Licence/access terms; whether classification labels are usable |
| Kaggle ITSRD / "Traffic Signs Dataset (Indian Roads)" | folder-per-class / 5 coarse categories, MIT | Original collection method; coarse categories are not sign classes |
| Zenodo "Indian Traffic VQA" | 1,085 images, CC-BY-4.0 | Visual question answering, not class labels |

GitHub-hosted projects that were reachable and inspected by cloning, both **without any
images**: `Shashank1130/Indian-Traffic-Sign-Detection` (notebook + MIT licence; points to
Kaggle) and `srikanthkb/Real-Time-Traffic-sign-detection-recognition` (code + a `.pt` file).

## 3. What is needed to proceed

Either allow the relevant host in the environment's network policy (then re-run the
search and verification), or place an authorized copy locally. Before using any
candidate, fill in the template below from the dataset's own metadata and from
`reports/dataset_report.json`, and confirm the licence permits this use.

```
python scripts/inspect_dataset.py --source data/raw/<dataset>   # writes reports/dataset_report.json
python scripts/prepare_dataset.py --source data/raw/<dataset> --name "<dataset name>"
```

## 4. Template to complete once a dataset is in place (all currently unknown)

| Field | Value |
|---|---|
| Dataset name / publisher / source URL / access method / licence / retrieval date | not established |
| Image count, class count, class distribution, imbalance ratio | not established; computed by `inspect_dataset.py` |
| Annotation format, geography, capture conditions | not established |
| Contributor/session IDs available? | not established |
| Known limitations | not established |

## 5. Split methodology implemented (code exists, not yet run on real data)

`scripts/prepare_dataset.py`:

1. Decodes every image; corrupted files are removed and counted.
2. Exact duplicates (SHA-256): one copy kept (in an official layout the test copy wins, so
   the test set is never altered to remove a train/test overlap — the train copy is dropped).
   Duplicate sets with conflicting labels are removed entirely.
3. Near-duplicates (64-bit dHash, Hamming distance <= 4) are clustered into groups, because
   no contributor/session IDs exist in a plain folder-per-class layout. The distance
   threshold is a heuristic and has not been validated on real images.
4. If the dataset ships `train/` and `test/` folders, they are preserved; validation is
   carved from train by class-stratified grouped folds. Otherwise a 7-fold
   `StratifiedGroupKFold` gives 1 test / 1 val / 5 train (~70/15/15), with an assertion
   that no group spans two splits.
5. Reports (does not fix) near-duplicate groups spanning an official train/test boundary.

## 6. Preprocessing at load time

Decode, convert to RGB, resize directly to 96x96 (aspect ratio not preserved), scale to
[0,1], normalise with ImageNet mean/std. Training adds small affine (<=10 deg, 5% shift,
0.9-1.1 scale), brightness/contrast/saturation jitter, and mild blur (p=0.2). No horizontal
flip (direction-bearing signs), no hue shift (colour carries class), no random-resized-crop.
