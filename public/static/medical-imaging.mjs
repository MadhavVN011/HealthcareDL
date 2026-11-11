import { prepareCtPixels, ctScores } from "./medical-core.mjs";

const base = new URL("./", import.meta.url);
const local = (path) => new URL(path, base).href;
let sessionPromise = null;

async function loadSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      if (!window.ort || !crypto.subtle)
        throw new Error(
          "This browser cannot start the CT engine. Use an up-to-date browser over HTTPS or localhost.",
        );
      window.ort.env.wasm.wasmPaths = local("vendor/onnx/");
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.wasm.proxy = false;
      const [metadataResponse, modelResponse] = await Promise.all([
        fetch(local("models/chest-ct/report.json")),
        fetch(local("models/chest-ct/model.onnx")),
      ]);
      if (!metadataResponse.ok || !modelResponse.ok)
        throw new Error("The CT model bundle is unavailable. Try again later.");
      const metadata = await metadataResponse.json(),
        bytes = await modelResponse.arrayBuffer();
      const hash = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
        (value) => value.toString(16).padStart(2, "0"),
      ).join("");
      if (
        metadata.schema_version !== 1 ||
        !Array.isArray(metadata.classes) ||
        !Number.isFinite(metadata.test?.accuracy) ||
        metadata.test.accuracy < 0 ||
        metadata.test.accuracy > 1 ||
        !Number.isInteger(metadata.test.count) ||
        metadata.test.count < 1 ||
        !Number.isFinite(metadata.min_score) ||
        metadata.min_score < 0 ||
        metadata.min_score > 1 ||
        hash !== metadata.model_sha256 ||
        metadata.architecture !== "mobilenet_v3_small_frozen_mlp64" ||
        metadata.classes.join(",") !== "comparator,covid_associated"
      )
        throw new Error("The CT model bundle failed its integrity check.");
      const session = await window.ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
      });
      return { session, metadata };
    })().catch((error) => {
      sessionPromise = null;
      throw error;
    });
  }
  return sessionPromise;
}

export async function classifyCtPixels(rgba, width, height, ready = () => {}) {
  const pixels = prepareCtPixels(rgba, width, height);
  const { session, metadata } = await loadSession();
  ready();
  const output = await session.run({
    pixels: new window.ort.Tensor("float32", pixels, [1, 3, 224, 224]),
  });
  const logits = Array.from(output.logits.data);
  return { logits, scores: ctScores(logits), metadata };
}
