# Clinical review record

Status: **pending qualified reviewer**. No clinician has reviewed or approved this application in this work session. No clinical certification, safety validation, or institutional endorsement is claimed.

## Completed engineering preparation

- Removed the original medication, precaution, diet, and exercise advice from serving.
- Mapped every dataset category to a source-linked educational description.
- Added an explicit research scope, uncertainty handling, and conservative urgent-care messages.
- Recorded reference availability separately in `data/reference_audit.json`.
- Added held-out pattern evaluation, an external-data evaluation command, and contamination rejection.

Reference availability checks only show whether a link can be reached. They do not verify that a source supports every claim or that a model prediction is medically meaningful.

## Materials for the reviewer

Review `data/conditions.json` (all 41 entries), `healthdl/inference.py` (messages and serving rules), all user-facing templates, `docs/DATASET.md`, `docs/MODEL_CARD.md`, and the included evaluation report. Assess individual summaries against their linked medical sources. Dataset labels and associations need their own review; medical references do not validate those associations.

| Review area | Required assessment | Current status |
|---|---|---|
| Condition terminology and summaries | Accuracy, scope, source support, and non-stigmatizing language | Pending |
| Symptom vocabulary | Ambiguous names, exposures versus symptoms, and missing context | Pending |
| Dataset labels and associations | Clinical plausibility and origin of labels | Pending |
| Urgent-care messages | Coverage, wording, severity/timing assumptions, false reassurance | Pending |
| Abstention behavior | Sparse inputs, overlapping illness patterns, comorbidities, and out-of-distribution inputs | Pending |
| Educational scope | Whether users could reasonably mistake results for diagnosis or treatment | Pending |
| Independent evaluation protocol | Population, endpoints, subgroup analysis, and appropriate reference standard | Pending |

## Review sign-off template

- Reviewer name and relevant qualifications:
- Organization and contact channel:
- Review date:
- Git revision or artifact hashes reviewed:
- Sources and reference dates checked:
- Findings, severity, and requested changes:
- Changes verified after remediation:
- Approved scope (if any):
- Outstanding limitations and conditions:

Leave this record pending until a qualified reviewer actually completes it. Do not substitute automated checks or generated prose for professional review.
