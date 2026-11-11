# Medical workspace dependencies and attribution

The browser workspace bundles these dependencies so uploaded documents are not sent to third-party servers. Versions, file paths, licenses, and SHA-256 checksums are recorded in `public/static/vendor/manifest.json`.

| Material | Version / source | License |
|---|---|---|
| [Tesseract.js](https://github.com/naptha/tesseract.js) | 7.0.0 | Apache-2.0; bundled `vendor/tesseract/LICENSE.txt` |
| [Tesseract.js Core](https://github.com/naptha/tesseract.js-core) | 7.0.0, including SIMD variants | Apache-2.0; bundled `vendor/tesseract/CORE-LICENSE.txt` |
| [English trained language data](https://github.com/naptha/tessdata) | `4.0.0_fast/eng.traineddata.gz`, pinned by checksum | Apache-2.0 |
| [PDF.js](https://github.com/mozilla/pdf.js) | 6.5.438 | Apache-2.0; bundled `vendor/pdfjs/LICENSE.txt` |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) | 1.31.1 | MIT; bundled `vendor/onnx/LICENSE.txt` and upstream third-party notices |
| [Torchvision MobileNetV3 source](https://github.com/pytorch/vision) | 0.29.2 CPU build | BSD-3-Clause; see `artifacts/chest_ct/TORCHVISION-LICENSE.txt` |
| ImageNet-pretrained encoder weights | `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, source/checksum recorded in the CT report | Follow the upstream pretrained-weight and dataset terms; the software license does not establish rights in the training images |

## CT dataset and example images

Credit: **Eduardo Soares, Plamen Angelov, Sarah Biaso, Michele Higa Froes, and Daniel Kanda Abe**, *SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification* (2020), [doi:10.1101/2020.04.24.20078584](https://doi.org/10.1101/2020.04.24.20078584).

The [Kaggle publisher release](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) declares **CC BY-NC-SA 4.0**. The archive's checksum, declared license, source, authors, and download size are recorded in `data/chest_ct_sources.json`. The preprint has its own terms; its text/images are not bundled here.

The two CT walkthrough images are reencoded validation-set images from that release. Image resizing/normalization is used for model inference. Keep attribution and the CC BY-NC-SA terms with these examples and the CT research artifacts; the project does not expand the publisher's permissions or establish permission for clinical/commercial use. Other inherited assets retain their separate provenance and terms.

The original dataset archive and complete image collection are excluded from Git and web deployment. Fictional blood-report examples were authored for this project and contain no real patient information.
