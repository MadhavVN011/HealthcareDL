"""Research-only chest CT transfer learning; never imported by the web runtime."""

import hashlib
import json
import random
import time
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .data import ROOT, file_hash

CLASSES = ["comparator", "covid_associated"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ARCHIVE_SHA256 = "1df9c4555fbae535fb2ca97132a94eefa88e7092f24a94d5f254ab97c0846c2a"
ENCODER_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


def preprocess(image):
    """Explicit half-pixel bilinear resize, mirrored by medical-imaging.js."""
    pixels = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    xs = (np.arange(224, dtype=np.float64) + 0.5) * width / 224 - 0.5
    ys = (np.arange(224, dtype=np.float64) + 0.5) * height / 224 - 0.5
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    wx, wy = (xs - x0).astype(np.float32), (ys - y0).astype(np.float32)
    x1, y1 = np.clip(x0 + 1, 0, width - 1), np.clip(y0 + 1, 0, height - 1)
    x0, y0 = np.clip(x0, 0, width - 1), np.clip(y0, 0, height - 1)
    top = pixels[y0[:, None], x0[None, :]] * (1 - wx[None, :, None]) + pixels[y0[:, None], x1[None, :]] * wx[None, :, None]
    bottom = pixels[y1[:, None], x0[None, :]] * (1 - wx[None, :, None]) + pixels[y1[:, None], x1[None, :]] * wx[None, :, None]
    resized = top * (1 - wy[:, None, None]) + bottom * wy[:, None, None]
    return ((resized / 255 - MEAN) / STD).transpose(2, 0, 1).copy()


class ChestNetwork(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(base.features, base.avgpool, nn.Flatten(1))
        self.register_buffer("feature_mean", torch.zeros(576))
        self.register_buffer("feature_scale", torch.ones(576))
        self.head = nn.Sequential(nn.Linear(576, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, pixels):
        return self.head((self.encoder(pixels) - self.feature_mean) / self.feature_scale)


def train_chest(archive_path, output, seed=20, epochs=180, patience=25):
    archive_path, output = Path(archive_path), Path(output)
    if file_hash(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Unexpected dataset archive; verify the source before training")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    names, labels, hashes, seen = [], [], [], {}
    with ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            group = Path(name).parts[0]
            if group not in {"COVID", "non-COVID"}:
                continue
            label = int(group == "COVID")
            with Image.open(BytesIO(archive.read(name))) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError("Conflicting labels on identical CT images")
                continue
            seen[digest] = label
            names.append(name)
            labels.append(label)
            hashes.append(digest)
    labels = np.array(labels, dtype=np.int64)
    ids = np.arange(len(names))
    train, remainder = train_test_split(ids, test_size=0.4, random_state=seed, stratify=labels)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed, stratify=labels[remainder])
    model = ChestNetwork(pretrained=True).eval()
    if file_hash(Path(torch.hub.get_dir()) / "checkpoints/mobilenet_v3_small-047dcff4.pth") != ENCODER_SHA256:
        raise ValueError("The pretrained ImageNet encoder checksum does not match")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    cache = archive_path.parent / "mobilenet_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if data["hashes"].tolist() != hashes:
                raise ValueError("Feature cache does not match the dataset")
            features = torch.from_numpy(data["features"].copy())
    else:
        encoded = []
        start = time.monotonic()
        with ZipFile(archive_path) as archive, torch.inference_mode():
            for offset in range(0, len(names), 32):
                batch = []
                for name in names[offset:offset + 32]:
                    with Image.open(BytesIO(archive.read(name))) as image:
                        batch.append(preprocess(image))
                encoded.append(model.encoder(torch.from_numpy(np.stack(batch))).clone())
                if offset % 320 == 0:
                    print(f"Encoded {offset + len(batch)}/{len(names)} CT slices ({time.monotonic() - start:.1f}s CPU)", flush=True)
        features = torch.cat(encoded)
        np.savez_compressed(cache, features=features.numpy(), hashes=np.array(hashes))
    # Feature normalization uses the training partition only.
    model.feature_mean.copy_(features[train].mean(0))
    model.feature_scale.copy_(features[train].std(0).clamp_min(0.01))
    standardized = (features - model.feature_mean) / model.feature_scale
    target = torch.from_numpy(labels)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    history, best_loss, best_epoch, best_state, stale = [], float("inf"), 0, None, 0
    for epoch in range(1, epochs + 1):
        model.head.train()
        losses = []
        for indexes in torch.from_numpy(train)[torch.randperm(len(train))].split(64):
            optimizer.zero_grad()
            loss = criterion(model.head(standardized[indexes]), target[indexes])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.head.eval()
        with torch.inference_mode():
            val_loss = float(criterion(model.head(standardized[validation]), target[validation]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        if val_loss < best_loss - 0.0001:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f"CT epoch {epoch}: validation loss {val_loss:.4f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()

    def metrics(indexes):
        with torch.inference_mode():
            probabilities = model.head(standardized[indexes]).softmax(1).numpy()
        predicted = probabilities.argmax(1)
        truth = labels[indexes]
        accepted = probabilities.max(1) >= 0.7
        return {"accuracy": accuracy_score(truth, predicted), "balanced_accuracy": balanced_accuracy_score(truth, predicted),
                "macro_f1": f1_score(truth, predicted, average="macro"), "roc_auc": roc_auc_score(truth, probabilities[:, 1]),
                "confusion_matrix": confusion_matrix(truth, predicted).tolist(), "count": len(indexes),
                "match_coverage_at_0_7": float(accepted.mean()),
                "accuracy_among_matches": float((predicted[accepted] == truth[accepted]).mean()) if accepted.any() else None}

    report = {
        "schema_version": 1, "architecture": "mobilenet_v3_small_frozen_mlp64", "seed": seed,
        "classes": CLASSES, "input_shape": [1, 3, 224, 224], "device": "cpu", "preprocessing": "half_pixel_bilinear_rgb_imagenet_normalization",
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__},
        "encoder": {"source": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth", "sha256": ENCODER_SHA256, "weights": "IMAGENET1K_V1", "frozen": True},
        "dataset": {"source": "https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset",
                    "license": "CC BY-NC-SA 4.0", "archive_sha256": ARCHIVE_SHA256, "unique_images": len(names),
                    "split_unit": "image", "patient_disjoint": False},
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_hashes": {key: [hashes[int(i)] for i in indexes] for key, indexes in [("train", train), ("validation", validation), ("test", test)]},
        "best_epoch": best_epoch, "epochs_run": len(history), "history": history,
        "validation": metrics(validation), "test": metrics(test), "min_score": 0.7,
        "limitations": [
            "Research classifier for rendered axial chest CT slices; not a diagnosis or a complete CT interpretation.",
            "Comparator does not mean healthy, negative for COVID, or free of other disease.",
            "Patient identifiers are unavailable; related slices from the same patient may cross image partitions. Scores may be optimistic.",
            "Softmax outputs and abstention thresholds are uncalibrated. No clinical or external patient evaluation has been performed.",
            "Not trained for X-rays, photos, head CT, DICOM studies, tumors, emboli, or general lung findings.",
        ],
    }
    torch.save(model.state_dict(), output / "model.pt")
    # The browser receives only the compact inference export and its own report.
    public = (ROOT / "public/static/models/chest-ct" if output.resolve() == (ROOT / "artifacts/chest_ct").resolve()
              else output / "web")
    public.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        sample = torch.zeros(1, 3, 224, 224)
        torch.onnx.export(model, sample, public / "model.onnx", input_names=["pixels"], output_names=["logits"],
                          opset_version=17, dynamo=False)
    import onnxruntime as ort
    session = ort.InferenceSession(str(public / "model.onnx"), providers=["CPUExecutionProvider"])
    examples = []
    errors = []
    with ZipFile(archive_path) as archive:
        for label in range(2):
            index = next(int(i) for i in validation if labels[i] == label)
            payload = archive.read(names[index])
            with Image.open(BytesIO(payload)) as image:
                image.convert("RGB").save(public / f"example-{label}.png")
                array = preprocess(image)[None]
            with torch.inference_mode():
                expected = model(torch.from_numpy(array)).numpy()
            actual = session.run(None, {"pixels": array})[0]
            errors.append(float(np.abs(expected - actual).max()))
            np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=0.0001)
            examples.append({"file": f"example-{label}.png", "class_id": CLASSES[label], "split": "validation", "logits": expected[0].tolist()})
    report["onnx_max_abs_error"] = max(errors)
    report["model_sha256"] = file_hash(public / "model.onnx")
    report["examples"] = examples
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key not in {"split_hashes", "history"}}
    summary_text = json.dumps(summary, indent=2) + "\n"
    (public / "report.json").write_text(summary_text)
    runtime_report = (ROOT / "artifacts/chest_ct_runtime.json"
                      if output.resolve() == (ROOT / "artifacts/chest_ct").resolve()
                      else output / "runtime_report.json")
    runtime_report.write_text(summary_text)
    print(json.dumps({"epochs_run": len(history), "best_epoch": best_epoch, "test": report["test"],
                      "model_bytes": (public / "model.onnx").stat().st_size, "onnx_max_abs_error": max(errors)}, indent=2), flush=True)
    return report
