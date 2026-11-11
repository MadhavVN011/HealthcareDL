# Data sources and attribution

Investigation date: November 11, 2026. `provenance.json` records publisher references, release versions, checksums, and comparisons. Re-run `python tools/audit_provenance.py` to repeat the checks against the publisher's then-current release; a changed upstream release may require a fresh review.

## Training table

- **Publisher:** KAUSHIL268 (Kaggle account `kaushil268`).
- **Title:** Disease Prediction Using Machine Learning.
- **Source:** <https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning>, version 1 as reported by the publisher API at the investigation date.
- **Declared license:** “Database: Open Database, Contents: Database Contents” in the publisher metadata. Relevant license texts are [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) and [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
- **Comparison:** All 4,920 rows and all named columns match exactly after removing one empty trailing column from the publisher file. The repository's `Training.csv` is unchanged from the initial checkout. Runtime preprocessing normalizes feature/label spellings, merges the duplicate fluid-overload column, and removes repeated patterns.

The publisher's `Testing.csv` contains 42 rows; 41 also occur in the original training table. It has not been presented as an independent patient benchmark or added as a new validation cohort. The publisher metadata is evidence of a declaration, not proof of original medical collection or a warranty of rights in every underlying element.

## Archived severity table

- **Publisher:** Pranay Patil (Kaggle account `itachi9604`).
- **Title:** Disease Symptom Prediction.
- **Source:** <https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset>, version 2 as reported by the publisher API.
- **Declared license:** [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
- **Comparison:** `data/legacy/Symptom-severity.csv` is an exact byte-for-byte match with the publisher's file. It is archived unchanged and not used in training or serving.

No matching claim or permission grant is made here for the remaining original treatment, description, symptom, precaution, diet, exercise, logo, or pickle files. Their provenance still needs to be established independently. The original logo is now archived and replaced in the interface by a newly authored SVG mark.

## Educational descriptions

`conditions.json` contains short source-linked educational summaries with stable IDs and explicit aliases for the original labels. Each entry links to a MedlinePlus health topic or encyclopedia page. The text is not a medical endorsement and has not been clinically approved for this application. See `reference_audit.json` for URL availability checks and `docs/CLINICAL_REVIEW.md` for the pending reviewer record.

## Code, artifacts, and distribution

Dataset licenses do not automatically license the repository's software, original branding, model artifacts, or unrelated files. No blanket software license has been invented for inherited code. Keep this attribution and the upstream license references with the relevant data, record adaptations, and complete the remaining ownership/permission review before distributing the original collection or claiming unrestricted commercial use.
