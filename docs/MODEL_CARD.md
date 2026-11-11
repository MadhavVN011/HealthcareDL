# SymptomNetwork model card

## Intended use

Educational study of multiclass classification on the supplied symptom table. The web interface explores dataset matches and links to public health information. The model is not intended for diagnosis, screening, emergency triage, prescribing, or individual treatment decisions.

## Architecture and training

A CPU PyTorch MLP maps 131 binary features to 41 category logits through 128-unit and 64-unit hidden layers, each followed by ReLU and dropout at 0.15. Training uses AdamW (learning rate 0.003, weight decay 0.001), cross-entropy loss, batch size 32, seed 20, at most 400 epochs, and validation-loss early stopping with patience 40 and minimum improvement 0.0001.

Exact duplicates are removed before a stratified per-category split. There are 182 training, 61 validation, and 61 held-out test patterns; no identical pattern is shared between partitions. The selected checkpoint is epoch 112; training stopped at epoch 152. Test labels are not used in gradient updates or checkpoint selection. The serving reference contains only training patterns.

## Results from the included run

| Evaluation | Accuracy | Macro F1 |
|---|---:|---:|
| Validation, 61 patterns | 1.000 | 1.000 |
| Held-out test, 61 patterns | 1.000 | 1.000 |
| Synthetic test with one positive symptom removed | 1.000 | 1.000 |
| Linear SVC comparison on the same held-out test | 1.000 | 1.000 |

The network does not demonstrate an improvement over the SVC baseline on these examples. The neural architecture fulfills the project's deep learning implementation, not a claim that deeper models are inherently better for this dataset.

Held-out neural log loss is approximately 0.00165 and multiclass Brier score approximately 0.000128. These scores on this small, highly separable table do not establish probability calibration in a clinical population. The JSON report includes full per-category precision, recall, support, confusion matrices, split hashes, and learning history.

With the fixed serving policy, 46 of 61 test patterns return an educational match and 15 return an urgent-care message. Match coverage is 75.4%, with 100% accuracy among those 46 returned matches against the dataset labels. These urgent messages are not evaluated or validated as clinical triage outcomes. Classifier metrics and serving-policy coverage measure different behavior and must be reported separately.

## Inference policy

The app normalizes and validates input names. Missing and unrecognized inputs are errors. Repeated names count once. Certain symptoms return an urgent-care message before inference; this list is not comprehensive. Examples include chest pain, breathlessness, sudden one-sided weakness, and slurred speech. See [MedlinePlus heart attack information](https://medlineplus.gov/heartattack.html) and [stroke information](https://medlineplus.gov/ency/article/000726.htm). Symptom severity or timing is not modeled.

Other inputs abstain when they have fewer than 3 or more than 17 features, a maximum Jaccard similarity to training patterns below 0.5, a top softmax score below 0.7, or a top-two score margin below 0.2. Predicted heart attack or brain hemorrhage also produces an urgent-care message. These are fixed engineering heuristics; no threshold calibration, clinical validation, or comprehensive out-of-distribution protection is claimed. Unsupported or uncertain results do not rule out illness.

## Limitations and remaining evidence gaps

- The original table has been matched to a public publisher release and its declared license recorded. Original clinical collection, consent, patient identity, and clinical verification remain unknown; see `data/SOURCES.md`.
- Only 304 patterns are unique, with 5–10 per category. Each held-out category has only one or two examples.
- Related templates may cross partitions. Patient-, time-, and site-separated evaluation is impossible with the supplied metadata.
- Inputs combine reported symptoms and some history/risk-factor fields. Unreported features are encoded as zero, which does not establish confirmed absence.
- No healthy or unknown-condition class exists. Categories are incomplete and mix diseases, manifestations, and stages.
- Performance across demographics, sites, illness severity, comorbidities, or real-world prevalence has not been measured.
- The synthetic missing-symptom check is not independent clinical validation.
- Educational descriptions are source-linked but have not undergone qualified clinical review for this application. Original treatment advice is quarantined and not served.

These gaps cannot be resolved by changing the neural architecture or adding more copies of the same patterns. They require documented data rights, qualified clinical review, and independent evaluation data.

## Artifact handling

The bundle contains `model.pt`, `inference.npz`, `metadata.json`, `training_patterns.npz`, and `report.json`. PyTorch training selects an evaluation-mode checkpoint and exports every dense-layer weight and bias. The lightweight runtime verifies schema, class IDs, checksums, dimensions, and finite values and executes the identical network with NumPy, without dropout. Tests compare logits to the canonical PyTorch model on all 304 unique patterns. The web function does not need PyTorch installed; research tools load the canonical checkpoint with `weights_only=True`. Hashes detect accidental inconsistency, not malicious replacement of an entire bundle. Only load artifacts from a trusted source.

Retraining regenerates the metrics; this document records the included seed-20 run and should be updated if the checked-in bundle changes. The old SVC pickle and inconsistent recommendation files are retained only in `data/legacy/` for traceability.
