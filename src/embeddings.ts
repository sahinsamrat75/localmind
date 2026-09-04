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

/**
 * Maximum number of characters handed to the model.
 *
 * MiniLM truncates at 512 tokens anyway, and inference cost is dominated by
 * sequence length — measured on this repo: ~139 texts/s at 117 chars vs ~7
 * texts/s at 1647 chars. Truncating the *embedded* text (while still storing
 * the full chunk) keeps ingestion tractable at codebase scale. The head of a
 * function is also the most retrieval-useful part (signature, doc comment,
 * opening logic).
 */
export const EMBED_TEXT_MAX_CHARS = 1200;

/** Default number of texts per model call when batching. */
export const DEFAULT_BATCH_SIZE = 32;

/** Clip text to the length we are willing to spend model time on. */
export function clipForEmbedding(text: string): string {
  if (text.length <= EMBED_TEXT_MAX_CHARS) return text;
  return text.slice(0, EMBED_TEXT_MAX_CHARS);
}

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

/** Options for a multi-text (batched) extractor call. */
type ExtractorOptions = Parameters<Extractor>[1];

/**
 * Batched-call options. `padding` is valid at runtime (transformers.js pads the
 * batch to its longest member) but missing from the published type, so it is
 * asserted rather than added.
 */
const BATCH_OPTIONS = {
  pooling: "mean",
  normalize: true,
  padding: true,
} as ExtractorOptions;

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

/** Warm the model so the first real call is not charged for load time. */
export async function warmModel(): Promise<void> {
  const extractor = await getExtractor();
  await extractor("warmup", { pooling: "mean", normalize: true });
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
  const output = await extractor(clipForEmbedding(text), {
    pooling: "mean",
    normalize: true,
  });
  const dims = output.dims as number[];
  const data = toFloat32Array(output.data);
  const hidden = dims[dims.length - 1];
  // Batch size is 1; return a copy of the single vector.
  return data.slice(0, hidden);
}

/**
 * Embed many texts in one model call per batch.
 *
 * Batching is a large win at codebase scale, but transformers.js pads every
 * sequence in a batch to the batch's longest member, and that padding
 * measurably perturbs mean-pooled output (cosine drift up to ~0.008 for a
 * short text batched next to a long one; exactly 0 when lengths match). So we
 * sort by length and batch within the sorted order: padding is minimal, which
 * keeps batched vectors numerically consistent with single-text vectors *and*
 * makes each call cheaper. Results come back in the caller's original order.
 */
export async function embedBatch(
  texts: string[],
  opts: { batchSize?: number; onBatch?: (done: number, total: number) => void } = {}
): Promise<Float32Array[]> {
  if (texts.length === 0) return [];
  const batchSize = Math.max(1, opts.batchSize ?? DEFAULT_BATCH_SIZE);
  const extractor = await getExtractor();

  const clipped = texts.map(clipForEmbedding);
  const order = clipped.map((_, i) => i).sort((a, b) => clipped[a].length - clipped[b].length);

  const out: Float32Array[] = new Array(texts.length);
  let done = 0;

  for (let start = 0; start < order.length; start += batchSize) {
    const idxs = order.slice(start, start + batchSize);
    const output = await extractor(idxs.map((i) => clipped[i]), BATCH_OPTIONS);
    const data = toFloat32Array(output.data);
    const dims = output.dims as number[];
    const hidden = dims[dims.length - 1] || EMBEDDING_DIM;
    for (let k = 0; k < idxs.length; k++) {
      out[idxs[k]] = data.slice(k * hidden, (k + 1) * hidden);
    }
    done += idxs.length;
    opts.onBatch?.(done, texts.length);
  }

  return out;
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
