# Dataset expansion and evaluation

The November 11, 2026 update expands CT training with additional publisher images and symptom training with explicitly synthetic examples. These are different kinds of expansion; neither is new clinical validation. The fixed protocol and former metrics are recorded in `data/expansion_protocol.json`, before training the new candidates.

## Symptom training

The original table remains unchanged: 4,920 rows, 304 unique patterns, and the same seed-20 split of 182/61/61. The [SymbiPredict release by Jay Tucker](https://data.mendeley.com/datasets/dv5z3v2xyd/1) declares CC BY 4.0 and contains 4,961 rows. After the same column normalization, all 304 binary patterns already exist in this project. Two labels are outside the current catalog. It is not merged, and it is not independent evaluation data.

Instead, training generates **913 unique synthetic variants** by deleting one positive symptom from original training rows, leaving at least three positives. This produces **1,095 training examples**. Candidates matching any original pattern, including validation/test, are excluded. Conflicting synthetic labels are discarded. Synthetic rows collectively receive at most the original rows' weight within each category; categories without eligible variants remain original-only. The actual total synthetic weight fraction is approximately 46.5%.

The unfamiliar-input reference and all serving thresholds retain their original definitions. Checkpoint selection uses original validation loss. No submitted symptoms or medical files enter training. The held-out test and missing-one-symptom stress check both remain at 100% accuracy and macro F1 1.000; the former model already achieved these scores. No accuracy improvement or new patient cohort is claimed. The existing benchmark has been reported before and is not a fresh independent test.

```bash
python train.py
# Reproduce an original-pattern baseline separately:
python train.py --no-augmentation --output artifacts-experiment
# Audit the separately downloaded SymbiPredict CSV without merging it:
python tools/audit_symptom_candidate.py --data data/external/symbipredict_2022.csv
```

The candidate's publisher download URL and SHA-256 are in the protocol record. The audit compares binary vectors, not medical equivalence or patient identities. The symptom training report records every accepted synthetic pattern hash, exclusion counts, weights, and zero added patient records.

## Larger chest CT release

The [Soares–Angelov Harvard Dataverse release](https://doi.org/10.7910/DVN/SZDUQX) is the same research family's larger release, with a declared CC0 1.0 license. It replaces the smaller Kaggle archive; the two releases are not concatenated or claimed as independent cohorts. Its checksummed ZIP contains 4,173 entries: **4,171 actual images**, a shortcut, and a text report. Training ignores the two non-image files without opening them. Two identical-pixel duplicates within patient folders are removed, leaving **4,169 images**, up **68.2%** from 2,479.

The publisher provides 210 patient folders, numbered separately within Covid/Healthy/Others. Group keys include the category before hashing. A fixed seed-20, category-stratified 60/20/20 patient-folder split gives:

| Partition | Patient folders | Images |
|---|---:|---:|
| Training | 126 | 2,472 |
| Validation | 42 | 864 |
| Test | 42 | 833 |

No exact pixel duplicate spans patient folders or partitions. The loader refuses conflicting labels or cross-folder identical images rather than silently allowing leakage. Folder identities have not been independently verified; this does not guarantee that all near-duplicates, repeated visits, or undocumented identity errors are absent. Both hospitals in the source release can occur in multiple partitions; this is not site separation.

The same frozen ImageNet MobileNetV3 Small encoder and 64-unit head are trained from a fresh initialization, without loading the old project checkpoint. Normalization uses training features only. The fixed optimizer, seed, patience, and validation-loss selection choose **epoch 2**, stopping after **27 epochs**. No hyperparameter changes were made in response to the test score.

The model correctly classifies **697/833 test images (83.67%)**, with macro F1 **0.8367** and ROC AUC **0.9184**. These are image-weighted metrics on held-out publisher patient folders, not patient-level diagnostic accuracy. The previous **96.77%** came from an image-random split of the smaller archive, where related slices could cross partitions. Different data and grouping make the two percentages unsuitable as an improvement/regression comparison.

The browser shows the new score, 833 test images, and 42 held-out folders. It retains the fixed 70% uncalibrated score threshold. Test match coverage is approximately 79.5%, with 90.0% accuracy among accepted dataset matches. Neither threshold nor score is clinically validated. Model size remains about 3.9 MB, and inference still runs in the visitor's browser.

## Reproduction and checks

See [REPORT_ANALYSIS.md](REPORT_ANALYSIS.md) for the pinned archive download and training command. Separate CPU repeats reproduced every checkpoint tensor, learning history, selected epoch, partition membership, augmentation membership, and test metric. The CT ONNX SHA-256 was reproduced exactly. The native ONNX/PyTorch maximum logit difference is below 0.00001; browser preprocessing/inference is checked against saved PyTorch example logits.

The no-augmentation baseline also reproduced the former checkpoint tensors and 152-epoch history exactly.

Tests verify training-only augmentation, ambiguous-label rejection, unchanged original partition membership, rejection of synthetic training variants as independent evaluation data, disjoint CT patient folders, image counts, real browser OCR, model integrity, and local inference. Raw archives and extracted datasets remain ignored and are excluded from deployment. No paid inference service or additional web-runtime dependency is introduced.
