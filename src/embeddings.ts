/**
 * Local-first embedding pipeline for localmind.
 *
 * Uses @xenova/transformers with all-MiniLM-L6-v2 (a 384-dim
 * sentence-transformers model). The model lives on local disk — bundled under
 * models/ in the published package — so the runtime makes ZERO network calls:
 * no API keys, no Hugging Face fetches. allowRemoteModels is hard-disabled
 * whenever the local model is present, so the embedding path can never
 * silently phone home.
 */
import { env, pipeline } from "@xenova/transformers";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
export const EMBEDDING_DIM = 384;

/** Directory that contains the all-MiniLM-L6-v2 model folder. */
function getModelDir(): string {
  const override = process.env.LOCALMIND_MODELS;
  if (override) return path.resolve(override);
  // dist/embeddings.js -> package root/models
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "models");
}

const modelDir = getModelDir();
if (fs.existsSync(path.join(modelDir, MODEL_ID))) {
  // Fully offline mode: load the bundled model, refuse any network fallback.
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = modelDir;
} else {
  // Development fallback (model not bundled, e.g. a shallow clone): persist
  // the one-time download under ~/.localmind so later runs are still local.
  env.allowLocalModels = true;
  env.cacheDir = path.join(modelDir, ".cache");
  console.error(
    `[localmind] model not found at ${path.join(modelDir, MODEL_ID)}; downloading once to ${env.cacheDir}`
  );
}

type Extractor = Awaited<ReturnType<typeof createExtractor>>;

function createExtractor() {
  return pipeline("feature-extraction", MODEL_ID, { quantized: true });
}

let extractorPromise: Promise<Extractor> | null = null;

/** Lazily create (and memoize) the feature-extraction pipeline. */
function getExtractor(): Promise<Extractor> {
  if (!extractorPromise) {
    extractorPromise = createExtractor();
  }
  return extractorPromise;
}

/** Convert a tensor's generic data value to a plain Float32Array. */
function toFloat32Array(data: unknown): Float32Array {
  if (data instanceof Float32Array) return data;
  if (Array.isArray(data)) return new Float32Array(data as number[]);
  throw new TypeError(`Unexpected tensor data type: ${typeof data}`);
}

/**
 * Embed a single text into a normalized 384-dim vector.
 *
 * `pooling: "mean"` + `normalize: true` is the canonical recipe for
 * sentence-transformers models: masked mean-pooling over token embeddings,
 * then L2 normalization (so cosine similarity is a plain dot product).
 */
export async function embed(text: string): Promise<Float32Array> {
  const extractor = await getExtractor();
  const output = await extractor(text, {
    pooling: "mean",
    normalize: true,
  });
  const dims = output.dims as number[];
  const data = toFloat32Array(output.data);
  const hidden = dims[dims.length - 1];
  // Batch size is 1; return a copy of the single vector.
  return data.slice(0, hidden);
}

/** Cosine similarity between two L2-normalized vectors (dot product). */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`embedding dimension mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}
