# Dataset notes

## What is known

`Training.csv` was supplied with this repository. It contains 4,920 rows, 132 raw symptom columns, and one categorical `prognosis` column. There are 41 equally frequent original labels, each appearing 120 times. Only 304 complete symptom patterns are unique, with 5–10 unique patterns per category. No missing cells were found in the raw training table. Symptom values are binary; each pattern contains 3–17 positive features.

## Cleaning policy

The CSV contains two columns named `fluid_overload`, at zero-based positions 45 and 117. The first is always zero; the second has 114 positive raw rows. The loader reads the original header directly and merges same-named normalized columns using logical OR. The resulting schema has 131 features. It does not discard the active signal or expose pandas' duplicate-column suffix as a symptom name.

Names are normalized through `healthdl.data`, shared with inference. This includes whitespace and punctuation inconsistencies in names such as `spotting_ urination`, `foul_smell_of urine`, and `dischromic _patches`. A small explicit alias table handles known spellings. Unrecognized user inputs are rejected rather than approximately mapped to an unrelated symptom.

Disease labels use stable catalog IDs. Explicit aliases cover legacy misspellings, trailing whitespace, and doubled whitespace, including peptic ulcer disease, diabetes, hypertension, positional vertigo, osteoarthritis, and hemorrhoids. The raw file is unchanged. Duplicate normalized symptom patterns are retained once. Conflicting labels for the same pattern are an error.

Each category's unique patterns are shuffled with a seeded generator and partitioned approximately 60/20/20. With seed 20 this yields 182 training, 61 validation, and 61 test patterns. The generated report records hashes for every pattern in each partition. The model's unfamiliar-input reference contains training patterns only.

## Training augmentation

The 182 original training patterns generate 913 unique single-positive-symptom deletion variants, for 1,095 training examples. Candidates matching any original pattern (including validation/test) are excluded, as are synthetic patterns with conflicting labels; at least three positive symptoms remain. Variants are downweighted so that collectively they have at most the original examples' weight within each category. The unfamiliar-input reference retains only original training patterns.

These are synthetic examples, not new patients or an enlarged clinical cohort. The candidate SymbiPredict table has 4,961 rows but the same 304 binary patterns; it adds no new patterns and is not merged. See [the expansion protocol](DATA_EXPANSION.md).

## What is unknown

A public file-level source has now been identified and publisher license declarations recorded in `data/SOURCES.md` and `data/provenance.json`. The repository still does not establish original clinical collection, consent, geography, time period, patient identifiers, or clinical verification. The data must not be described as a representative patient cohort or a clinically validated dataset. Repetition and separability are observable; they do not establish how the data was generated.

Labels mix conditions, manifestations, and disease stages. Binary inputs do not distinguish an unreported symptom from a confirmed absence. Duration, severity, examination, laboratory findings, medication history, and demographics are missing. There are no healthy controls or general unknown-condition examples. The 41 categories do not cover the range of possible causes of symptoms.

## Evaluation boundary

Exact duplicate removal prevents identical-pattern overlap, but related templates may remain in different partitions. Pattern-level separation cannot replace patient-, site-, or time-separated evaluation. The held-out set contains only 61 patterns, so per-category results may depend on one or two examples. A perfect score is not evidence that this application can diagnose patients.

Model selection uses validation loss only. The report includes a synthetic test in which one positive symptom is removed from each test pattern; it is a stress check, not an external validation set. The serving policy uses fixed, unvalidated abstention rules and conservative urgent-care messages. Neither these rules nor a high neural score establish clinical safety.

## Health content and original assets

`data/conditions.json` contains short educational descriptions and links to corresponding MedlinePlus topics. Reference URLs were checked on 2026-11-11. No qualified clinical review or endorsement of the project is claimed. Educational references do not validate the training labels.

Original treatment, diet, exercise, precaution, severity, and symptom lookup files are retained under `data/legacy/` and excluded from runtime. The archived severity table is a byte-for-byte match with a publisher release declaring CC BY-SA 4.0. Other inherited lookup tables and the original logo remain unverified. See `data/SOURCES.md` for file-level evidence and publisher license declarations. The original logo is no longer served. No blanket license is asserted for unrelated inherited materials.

## Publisher test file audit

The identified publisher training file has one extra trailing empty column; all named headers and all 4,920 rows otherwise match. Its associated `Testing.csv` has 42 rows, of which 41 already occur in the original training table. This file does not provide independent clinical validation. `tools/audit_provenance.py` reproduces the comparison using public publisher archives.
