# Independent evaluation plan

The included held-out test is a pattern-level split of one repetitive table. A publisher test file was found during provenance investigation: **41 of its 42 rows already occur in this project's original training table**. It is unsuitable as independent patient validation, and the external evaluation command rejects overlapping data.

## Prepared evaluation workflow

`evaluate.py` accepts an external binary symptom CSV with the model's complete feature schema and a `prognosis` label column. Known source spelling aliases and duplicate feature columns are normalized with the same rules as training. Optional patient IDs can be supplied using `--group-column`. The command checks schema and labels, reports overlap with every original train/validation/test pattern and every synthetic training variant, and refuses to calculate independent metrics if any overlap exists. Patient IDs are counted, never written to the report.

No external patient records are bundled in this repository. Put approved data under ignored `data/external/` and retain the source's access and license records separately. Unknown columns and missing features are rejected; they are not silently imputed or mapped using guessed medical equivalences.

```bash
python -m pip install -r requirements-training.txt
python evaluate.py --data data/external/approved.csv \
  --group-column patient_id --output data/external/evaluation.json
```

A report remains a research result. Supplied patient IDs cannot establish independence from the original data because that table has no patient identifiers. Identical-pattern exclusion is an engineering contamination check, not a clinical cohort design.

## Candidate sources investigated

| Source | What it provides | Appropriate use and limitation |
|---|---|---|
| [DDXPlus, original authors](https://github.com/mila-iqia/ddxplus) | Synthesized patients, symptoms/antecedents, differential diagnoses, and predefined train/validation/test releases; authors declare CC-BY | A better documented research benchmark. Its category and evidence schema differs, so it needs a separate adapter and experimental protocol. Synthesized cases do not establish clinical patient performance. |
| [MIMIC access documentation](https://mimic.mit.edu/docs/faq/how-to-get-access.html) | Credentialed clinical datasets accessed through PhysioNet | Requires the user's own credentialing, required training, and dataset-specific data-use agreements. No restricted records were accessed or downloaded. These records are not a drop-in symptom checklist benchmark. |

## Protocol requiring qualified collaborators

1. Specify the intended research population, inclusion criteria, outcomes, and reference diagnoses before inspecting final test results.
2. Establish access rights, applicable consent/ethics requirements, and an appropriate environment for the source data.
3. Agree a clinically defensible feature mapping. Unmeasured findings must not automatically become confirmed negatives.
4. Separate patient groups and, where possible, collection times and sites. Freeze the model and serving policy before evaluation.
5. Report per-category precision/recall with support, macro metrics, uncertainty, abstention coverage, and relevant subgroup results. Address class prevalence and small-sample uncertainty.
6. Have qualified reviewers assess failures and false reassurance. Keep the project educational until evidence supports any broader use.

Independent patient data and reviewer sign-off are **not completed** by the preparation of this workflow.
