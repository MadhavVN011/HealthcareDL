# Blood reports and chest CT research

The `/reports` workspace has two separate pipelines. Neither feeds uploaded data into the symptom classifier, diagnoses a patient, or generates treatment advice. Its results do not inherit the symptom model's test accuracy.

## Blood reports

PDF.js 6.5.438 extracts text from PDFs. Pages without supported readable test rows are rendered and passed to Tesseract.js 7.0.0 with its English LSTM OCR model. PNG/JPEG images use the same OCR engine; UTF-8 text files and pasted text bypass OCR. All dependencies and language data are served from this site, without a third-party inference API.

The parser recognizes common test names, numeric readings, measurement units, and a single printed numeric reference interval or one-sided bound. It reports **within**, **above**, or **below the printed range**. Comparisons are arithmetic, not a trained blood-disease classifier. No universal reference ranges or medical cutoffs are filled in.

Missing ranges/units, unsupported decimal formats, inconsistent units, multiple readings, group-specific ranges, and ambiguous inequalities are marked **Needs review**. Unsupported tests are not interpreted. The workflow is not a complete medical report reader: page layout, OCR errors, qualitative results, methods, sample dates, age/sex context, and unfamiliar tests can prevent interpretation. Verify the extracted source text and every reading.

The [MedlinePlus explanation of lab results](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/) explains why ranges vary and why being inside or outside a range does not establish health or disease. Fictional sample PDF, image, and text files in `public/static/examples/` contain no patient data; their numbers are walkthrough inputs, not prescribed reference ranges.

Limits: 10 MB per file, at most five PDF pages, English-language recognition, at most 16 million source-image pixels, and 200,000 extracted text characters. Longer PDFs are rejected rather than partially interpreted. Encrypted, malformed, or unsupported documents produce an error, not a medical result.

## Chest CT model

The classifier accepts a single **rendered axial chest CT slice in PNG/JPEG format**. It does not read DICOM studies or interpret other body regions, X-rays, photos, lung tumors, emboli, or all possible lung conditions. A user confirms the supported input type before running an uploaded image. There is no validated automatic domain detector.

The neural network uses a frozen ImageNet-pretrained MobileNetV3 Small encoder, 576 normalized features, and a trained 64-unit ReLU/dropout classification head. The feature mean and scale come only from the training partition. CPU training uses seed 20, AdamW, cross-entropy, and validation-loss checkpoint selection. The selected run completed 43 epochs and selected epoch 18. Training does not use the test partition for updates or checkpoint selection.

The source is the [SARS-CoV-2 CT-scan dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset). Three pixel-identical duplicates were removed from 2,482 images before a stratified **1,487 / 496 / 496 image split**. The downloaded archive does not provide the patient identifiers needed for a patient-separated split. Related slices from the same person can occur in different partitions; image-level results can therefore be optimistic. This is not independent patient validation or clinical performance.

| Held-out image metric | Measured value |
|---|---:|
| Accuracy | 96.77% (480 / 496) |
| Balanced accuracy | 96.77% |
| Macro F1 | 0.9677 |
| ROC AUC | 0.9956 |

The classes are **COVID-associated dataset pattern** and **comparator dataset pattern**. Comparator does not mean healthy, COVID-negative, or free from other diseases. Softmax scores are uncalibrated; the 70% abstention threshold is a fixed research heuristic. A low score returns no selected class. The displayed evaluation accuracy describes the dataset, not the uploaded image's correctness.

The complete selected checkpoint is `artifacts/chest_ct/model.pt`; the full report records split hashes, history, versions, and metrics. The 3.9 MB ONNX export and a compact report are under `public/static/models/chest-ct/`. ONNX Runtime Web 1.31.1 executes the network in the visitor's browser using single-threaded CPU WebAssembly. A SHA-256 check detects a mismatched model/report bundle; it is not a signature or an authenticity guarantee. Browser preprocessing and logits are checked against recorded PyTorch outputs. The initial two example images come from validation, are reencoded as PNG, and are not independent evaluation cases.

### Reproduce CT training

Download the public publisher archive into the ignored `data/external/chest_ct/` directory:

```bash
mkdir -p data/external/chest_ct
curl -L https://www.kaggle.com/api/v1/datasets/download/plameneduardo/sarscov2-ctscan-dataset \
  -o data/external/chest_ct/sarscov2-ctscan-dataset.zip
python -m pip install -r requirements-imaging-training.txt
python tools/train_chest_ct.py
```

The script verifies the archive and pretrained encoder checksums before training. Feature caches live beside the ignored dataset; remove the cache when changing preprocessing. The web server does not install or import this imaging training stack. Exact reproducibility is only established for the recorded CPU environment.

A second CPU run reproduced the selected checkpoint tensors, learning history, split hashes, test metrics, and ONNX export exactly. A custom `--output` destination keeps its browser export in that destination's `web/` subdirectory, leaving the checked-in model untouched.

## Privacy and deployment

Files, pixel arrays, report text, and results stay in the current browser tab. No selected file or medical content is posted to Flask or an external AI service, and report data is not written to local storage, IndexedDB, or a database by the application. The theme preference remains the only application preference saved in local storage. The OCR engine's language cache is disabled. Clear buttons remove the selected file, extracted text, and displayed results from the workspace. Browser memory, extensions, screenshots, and the operating system remain outside the application's control.

Vercel serves static model/worker assets and the page. Its logs can record ordinary page and asset requests. Heavy inference runs on the visitor's device, so slow devices can take longer; the first CT run downloads about 19 MB of model/runtime assets. The report page allows local preview URLs, same-origin workers, and WebAssembly in its content security policy. Other application pages retain their previous script restrictions.

Source updates belong on `development`; review and merge into `main` using a pull request. Third-party assets and data attribution are recorded in [THIRD_PARTY.md](THIRD_PARTY.md) and `data/chest_ct_sources.json`.
