import { interpretLabText, DEMO_LAB_TEXT } from "./medical-core.mjs";
import { classifyCtPixels } from "./medical-imaging.mjs";

const base = new URL("./", import.meta.url);
const local = (path) => new URL(path, base).href;
const byId = (id) => document.getElementById(id);
const maxBytes = 10 * 1024 * 1024;
let bloodJob = 0,
  ctJob = 0,
  activeWorker = null,
  activePdf = null,
  previewUrl = null;

function errorText(error) {
  return error instanceof Error
    ? error.message
    : "The file could not be processed. Try a supported format.";
}

function setError(prefix, message) {
  byId(`${prefix}-error`).textContent = message;
  byId(`${prefix}-error`).hidden = !message;
}

async function fileKind(file, allowDocuments) {
  if (!file || !file.size) throw new Error("Choose a non-empty file first.");
  if (file.size > maxBytes)
    throw new Error("Choose a file smaller than 10 MB.");
  const header = new Uint8Array(await file.slice(0, 16).arrayBuffer());
  const name = file.name?.toLowerCase() || "";
  if (
    header[0] === 137 &&
    header[1] === 80 &&
    header[2] === 78 &&
    header[3] === 71 &&
    (!name || name.endsWith(".png"))
  )
    return "image";
  if (
    header[0] === 255 &&
    header[1] === 216 &&
    header[2] === 255 &&
    (!name || /\.jpe?g$/.test(name))
  )
    return "image";
  if (
    allowDocuments &&
    name.endsWith(".pdf") &&
    String.fromCharCode(...header.slice(0, 5)) === "%PDF-"
  )
    return "pdf";
  if (allowDocuments && name.endsWith(".txt") && !header.includes(0))
    return "text";
  throw new Error(
    allowDocuments
      ? "Choose a readable PDF, PNG, JPEG, or UTF-8 text file. The contents must match the extension."
      : "Choose a PNG or JPEG chest CT slice. DICOM files, PDFs, and other image types are not supported.",
  );
}

async function decodeImage(blob) {
  const image = await createImageBitmap(blob, {
    imageOrientation: "from-image",
  });
  try {
    if (image.width * image.height > 16000000)
      throw new Error("Choose an image with at most 16 million pixels.");
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0);
    return canvas;
  } finally {
    image.close();
  }
}

function renderLab(text, method) {
  const result = interpretLabText(text);
  const body = byId("blood-rows");
  body.replaceChildren();
  for (const row of result.rows) {
    const tr = document.createElement("tr");
    const test = document.createElement("td");
    const name = document.createElement("strong");
    name.textContent = row.label;
    const reading = document.createElement("span");
    reading.className = "lab-reading";
    reading.textContent = `${row.result} ${row.unit}`;
    test.append(name, reading);
    const reference = document.createElement("td");
    reference.textContent = row.reference;
    const interpretation = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `lab-badge lab-${row.status}`;
    badge.textContent = {
      within: "Within printed range",
      above: "Above printed range",
      below: "Below printed range",
      review: "Needs review",
    }[row.status];
    const reason = document.createElement("p");
    reason.className = "lab-reason";
    reason.textContent = row.reason;
    interpretation.append(badge, reason);
    tr.append(test, reference, interpretation);
    body.append(tr);
  }
  byId("blood-source").textContent = text;
  byId("blood-method").textContent = method;
  byId("blood-summary").textContent = result.rows.length
    ? `${result.rows.length} supported readings found. ${result.compared} compared with printed ranges; ${result.outside} outside those ranges; ${result.review} need review. This is not an overall health assessment.`
    : "No supported test rows could be read reliably. Review the extracted text or paste one test per line with its result, unit, and reference range. No medical conclusion has been generated.";
  byId("blood-results").hidden = false;
  byId("blood-clear").hidden = false;
}

async function recognize(canvas, job) {
  if (!window.Tesseract)
    throw new Error(
      "The OCR engine did not load. Reload this page and try again.",
    );
  let worker;
  try {
    worker = await window.Tesseract.createWorker("eng", 1, {
      workerPath: local("vendor/tesseract/worker.min.js"),
      corePath: local("vendor/tesseract/"),
      langPath: local("vendor/tesseract/"),
      workerBlobURL: false,
      cacheMethod: "none",
      logger(message) {
        if (job === bloodJob)
          byId("blood-status").textContent =
            `Reading report on this device: ${message.status}${Number.isFinite(message.progress) ? ` ${Math.round(message.progress * 100)}%` : ""}…`;
      },
    });
    if (job !== bloodJob) return "";
    activeWorker = worker;
    const { data } = await worker.recognize(canvas);
    return data.text;
  } finally {
    if (activeWorker === worker) activeWorker = null;
    if (worker) await worker.terminate();
    canvas.width = canvas.height = 0;
  }
}

function pdfLines(items) {
  const lines = [];
  for (const item of items) {
    if (!item.str?.trim()) continue;
    const y = item.transform[5];
    let line = lines.find((row) => Math.abs(row.y - y) < 2.5);
    if (!line) {
      line = { y, items: [] };
      lines.push(line);
    }
    line.items.push({ x: item.transform[4], text: item.str });
  }
  return lines
    .sort((a, b) => b.y - a.y)
    .map((row) =>
      row.items
        .sort((a, b) => a.x - b.x)
        .map((item) => item.text)
        .join(" "),
    )
    .join("\n");
}

async function readPdf(file, job) {
  const pdfjs = await import("./vendor/pdfjs/pdf.min.mjs");
  pdfjs.GlobalWorkerOptions.workerSrc = local(
    "vendor/pdfjs/pdf.worker.min.mjs",
  );
  const loading = pdfjs.getDocument({
    data: new Uint8Array(await file.arrayBuffer()),
    isEvalSupported: false,
    useWasm: false,
    useSystemFonts: true,
    disableFontFace: true,
    maxImageSize: 16000000,
    stopAtErrors: true,
  });
  activePdf = loading;
  let usedOcr = false;
  try {
    const pdf = await loading.promise;
    if (pdf.numPages > 5)
      throw new Error(
        "Choose a PDF with at most 5 pages. This limit prevents incomplete analysis of longer reports.",
      );
    const pages = [];
    for (let index = 1; index <= pdf.numPages; index++) {
      if (job !== bloodJob) return null;
      byId("blood-status").textContent =
        `Reading PDF page ${index} of ${pdf.numPages} on this device…`;
      const page = await pdf.getPage(index);
      let text = pdfLines((await page.getTextContent()).items);
      const parsed = interpretLabText(text);
      if (!parsed.rows.some((row) => row.result !== "Unreadable")) {
        const original = page.getViewport({ scale: 1 });
        const viewport = page.getViewport({
          scale: Math.min(
            2,
            1800 / original.width,
            Math.sqrt(8000000 / (original.width * original.height)),
          ),
        });
        const canvas = document.createElement("canvas");
        canvas.width = Math.ceil(viewport.width);
        canvas.height = Math.ceil(viewport.height);
        await page.render({ canvasContext: canvas.getContext("2d"), viewport })
          .promise;
        text = await recognize(canvas, job);
        usedOcr = true;
      }
      pages.push(text);
      page.cleanup();
    }
    return {
      text: pages.join("\n"),
      method: usedOcr ? "PDF text + local OCR" : "PDF text extraction",
    };
  } finally {
    if (activePdf === loading) activePdf = null;
    await loading.destroy();
  }
}

byId("blood-controls").disabled = false;
byId("blood-file").addEventListener("change", () => {
  byId("blood-text").value = "";
  byId("blood-source").textContent = "";
  byId("blood-rows").replaceChildren();
  byId("blood-results").hidden = true;
  setError("blood", "");
  byId("blood-status").textContent =
    "New file selected. Analyze it to extract its readings.";
});
byId("blood-text").addEventListener("input", () => {
  byId("blood-source").textContent = "";
  byId("blood-rows").replaceChildren();
  byId("blood-results").hidden = true;
});
byId("blood-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const job = ++bloodJob;
  setError("blood", "");
  byId("blood-results").hidden = true;
  byId("blood-clear").hidden = false;
  byId("blood-controls").disabled = true;
  byId("blood-status").textContent = "Reading report on this device…";
  try {
    let extracted;
    const pasted = byId("blood-text").value.trim();
    if (pasted) extracted = { text: pasted, method: "Pasted text" };
    else {
      const file = byId("blood-file").files[0];
      const kind = await fileKind(file, true);
      if (kind === "text")
        extracted = {
          text: new TextDecoder("utf-8", { fatal: true }).decode(
            await file.arrayBuffer(),
          ),
          method: "Text report",
        };
      else if (kind === "image")
        extracted = {
          text: await recognize(await decodeImage(file), job),
          method: "Local image OCR",
        };
      else extracted = await readPdf(file, job);
    }
    if (job !== bloodJob || !extracted) return;
    renderLab(extracted.text, extracted.method);
    byId("blood-status").textContent =
      "Analysis finished. Verify the extracted readings against the original report.";
  } catch (error) {
    if (job === bloodJob) {
      setError("blood", errorText(error));
      byId("blood-status").textContent =
        "No report interpretation was produced.";
    }
  } finally {
    if (job === bloodJob) byId("blood-controls").disabled = false;
  }
});
byId("blood-example").addEventListener("click", () => {
  setError("blood", "");
  byId("blood-file").value = "";
  byId("blood-text").value = DEMO_LAB_TEXT;
  renderLab(DEMO_LAB_TEXT, "Fictional example");
  byId("blood-status").textContent =
    "Fictional example analyzed using the same comparison rules as uploaded reports.";
});
byId("blood-clear").addEventListener("click", () => {
  bloodJob++;
  activeWorker?.terminate();
  activePdf?.destroy();
  activeWorker = activePdf = null;
  byId("blood-file").value = byId("blood-text").value = "";
  byId("blood-source").textContent = "";
  byId("blood-rows").replaceChildren();
  byId("blood-results").hidden = byId("blood-clear").hidden = true;
  byId("blood-controls").disabled = false;
  setError("blood", "");
  byId("blood-status").textContent =
    "Report and results cleared from this workspace.";
});

async function analyzeCt(file, example = null) {
  const job = ++ctJob;
  byId("ct-controls").disabled = true;
  byId("ct-clear").hidden = false;
  byId("ct-results").hidden = true;
  setError("ct", "");
  byId("ct-status").textContent =
    "Preparing the local CT model. The first run downloads about 19 MB of model and runtime assets…";
  try {
    await fileKind(file, false);
    const canvas = await decodeImage(file);
    let prediction;
    try {
      prediction = classifyCtPixels(
        canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height)
          .data,
        canvas.width,
        canvas.height,
        () => {
          if (job === ctJob)
            byId("ct-status").textContent =
              "Running the convolutional neural network on this device…";
        },
      );
    } finally {
      canvas.width = canvas.height = 0;
    }
    const { scores, metadata } = await prediction;
    if (job !== ctJob) return;
    const index = scores[1] > scores[0] ? 1 : 0,
      score = scores[index],
      uncertain = score < metadata.min_score;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    byId("ct-preview").src = previewUrl;
    byId("ct-preview").hidden = false;
    byId("ct-result-badge").textContent = uncertain
      ? "Uncertain model output"
      : "Research dataset pattern";
    byId("ct-result-label").textContent = uncertain
      ? "No clear CT dataset pattern"
      : index === 1
        ? "COVID-associated dataset pattern"
        : "Comparator dataset pattern";
    byId("ct-result-description").textContent =
      `${example === null ? "Selected image." : "Published validation example."} ${uncertain ? "The model score is below the demonstration's 70% threshold. No class has been selected." : "This is a slice-level research classification, not a diagnosis or a complete scan interpretation."} The model cannot determine whether an image belongs to its supported domain.`;
    byId("ct-result-score").textContent = `${(score * 100).toFixed(1)}%`;
    byId("ct-result-accuracy").textContent =
      `${(metadata.test.accuracy * 100).toFixed(1)}%`;
    byId("ct-result-support").textContent =
      `${metadata.test.count} held-out dataset images; not a patient-separated evaluation.`;
    byId("ct-results").hidden = false;
    byId("ct-status").textContent =
      "Local classification finished. Your image was not uploaded.";
  } catch (error) {
    if (job === ctJob) {
      setError("ct", errorText(error));
      byId("ct-status").textContent = "No CT classification was produced.";
    }
  } finally {
    if (job === ctJob) byId("ct-controls").disabled = false;
  }
}

if (!byId("ct-controls").dataset.unavailable)
  byId("ct-controls").disabled = false;
byId("ct-file").addEventListener("change", () => {
  byId("ct-results").hidden = true;
  byId("ct-confirm").checked = false;
  byId("ct-preview").removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  setError("ct", "");
  byId("ct-status").textContent =
    "New image selected. Confirm the image type before running the model.";
});
byId("ct-form").addEventListener("submit", (event) => {
  event.preventDefault();
  analyzeCt(byId("ct-file").files[0]);
});
for (const button of document.querySelectorAll("[data-ct-example]")) {
  button.addEventListener("click", async () => {
    const index = Number(button.dataset.ctExample);
    const job = ++ctJob;
    byId("ct-controls").disabled = true;
    byId("ct-clear").hidden = false;
    byId("ct-status").textContent = "Loading the published dataset example…";
    try {
      const response = await fetch(
        local(`models/chest-ct/example-${index}.png`),
      );
      if (!response.ok) throw new Error("The CT example is unavailable.");
      const blob = await response.blob();
      if (job !== ctJob) return;
      byId("ct-file").value = "";
      byId("ct-confirm").checked = true;
      await analyzeCt(blob, index);
    } catch (error) {
      if (job === ctJob) {
        setError("ct", errorText(error));
        byId("ct-controls").disabled = false;
      }
    }
  });
}
byId("ct-clear").addEventListener("click", () => {
  ctJob++;
  byId("ct-file").value = "";
  byId("ct-confirm").checked = false;
  byId("ct-controls").disabled = Boolean(
    byId("ct-controls").dataset.unavailable,
  );
  byId("ct-results").hidden = byId("ct-clear").hidden = true;
  byId("ct-preview").removeAttribute("src");
  byId("ct-preview").hidden = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  for (const id of [
    "ct-result-label",
    "ct-result-description",
    "ct-result-score",
    "ct-result-accuracy",
    "ct-result-support",
  ])
    byId(id).textContent = "";
  setError("ct", "");
  byId("ct-status").textContent =
    "Image and results cleared from this workspace.";
});
window.addEventListener("pagehide", () => {
  bloodJob++;
  ctJob++;
  activeWorker?.terminate();
  activePdf?.destroy();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    byId("blood-clear").click();
    byId("ct-clear").click();
  }
});
