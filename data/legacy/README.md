# Archived original artifacts — not for recommendation use

These files preserve the original repository inputs and model for traceability. Neither the application nor the neural training pipeline imports or loads them. The active dataset remains `../../Training.csv`; the active condition catalog is `../conditions.json`.

The original `medications.csv` contains incorrect disease-to-treatment associations, including clot-related medicines for brain hemorrhage and varicose-vein treatments for heart attack. Other tables contain unverified health advice, inconsistent disease names, missing precautions, dietary advice labeled as exercise, and list-shaped strings instead of structured values. Fixing spelling does not make these recommendations medically valid.

The original descriptions have been superseded by the source-linked condition catalog. `symtoms_df.csv` and `Symptom-severity.csv` are unused legacy datasets with undocumented provenance. `svc.pkl` is the original scikit-learn 1.6.1 SVC, retained only for history; do not load an untrusted pickle. The reproducible baseline is retrained in memory by `train.py` and is never deployed.

Do not restore these tables to the serving path without establishing provenance and performing qualified clinical review. Original content is preserved unchanged, including its defects. See the Git history for the former file locations and application.
