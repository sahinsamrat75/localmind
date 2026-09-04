import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { chunkFile, detectLanguage, type Chunk } from "./chunker.js";
import { walkDirectory, looksBinary } from "./walk.js";
import { embedBatch, DEFAULT_BATCH_SIZE } from "./embeddings.js";
import { KIND_CODE, type MemoryStore } from "./store.js";
import type { VectorIndex } from "./vectorIndex.js";

/** SHA-256 of the exact chunk text — the unit of incremental re-ingestion. */
export function contentHash(text: string): string {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

export interface IngestOptions {
  project: string;
  /** Directory to ingest. */
  root: string;
  /** Only ingest files whose relative path matches one of these substrings/globs. */
  include?: string[];
  /** Skip files whose relative path matches one of these substrings/globs. */
  exclude?: string[];
  respectGitignore?: boolean;
  /** Files larger than this are skipped rather than chunked (default 512 KiB). */
  maxFileBytes?: number;
  maxFiles?: number;
  chunkLines?: number;
  chunkOverlap?: number;
  maxChunkLines?: number;
  maxChunkChars?: number;
  /** Texts per embedding call. */
  batchSize?: number;
  /** New chunks buffered before a batched embed + write. */
  waveSize?: number;
  /** Compute and report the plan without touching the database or index. */
  dryRun?: boolean;
  log?: (msg: string) => void;
}

export interface IngestSummary {
  root: string;
  project: string;
  dryRun: boolean;
  filesScanned: number;
  filesSkipped: number;
  filesChanged: number;
  filesDeleted: number;
  chunksCreated: number;
  /** Chunks whose text was unchanged but whose line range moved. */
  chunksUpdated: number;
  chunksUnchanged: number;
  chunksDeleted: number;
  /** Chunks actually sent through the embedding model. */
  chunksEmbedded: number;
  totalChunks: number;
  elapsedMs: number;
  embedMs: number;
  byLanguage: Record<string, number>;
}

/** A chunk awaiting embedding. */
interface Pending {
  relPath: string;
  chunk: Chunk;
  hash: string;
}

/**
 * Prefix handed to the model alongside the code.
 *
 * The file path and unit name carry a lot of signal for queries like "the
 * upload handler in the payments service", so they are included in the
 * *embedded* text while the stored `text` stays the clean source slice.
 */
function embedTextFor(relPath: string, chunk: Chunk): string {
  const unit = chunk.unit ? ` ${chunk.unitKind} ${chunk.unit}` : ` ${chunk.unitKind}`;
  return `${relPath}:${chunk.startLine}-${chunk.endLine}${unit}\n${chunk.content}`;
}

/** Substring match, or glob match when the pattern contains a wildcard. */
function matchesAny(relPath: string, patterns: string[]): boolean {
  return patterns.some((p) => {
    if (p.includes("*") || p.includes("?")) {
      const re = new RegExp(
        "^" +
          p
            .replace(/[.+^${}()|[\]\\]/g, "\\$&")
            .replace(/\*\*/g, "\u0000")
            .replace(/[*]/g, "[^/]*")
            .replace(/\?/g, "[^/]")
            .replace(/\u0000/g, ".*") +
          "$"
      );
      return re.test(relPath);
    }
    return relPath.includes(p);
  });
}

/**
 * Embed a chunk wave and write it through to SQLite + the ANN index.
 * Returns the number of chunks that went through the model.
 */
async function flushWave(
  pending: Pending[],
  opts: IngestOptions,
  store: MemoryStore,
  index: VectorIndex,
  timing: { embedMs: number }
): Promise<number> {
  if (pending.length === 0) return 0;
  const texts = pending.map((p) => embedTextFor(p.relPath, p.chunk));
  const t0 = Date.now();
  const vectors = await embedBatch(texts, { batchSize: opts.batchSize ?? DEFAULT_BATCH_SIZE });
  timing.embedMs += Date.now() - t0;

  const rows = pending.map((p, i) => ({
    project: opts.project,
    kind: KIND_CODE,
    text: p.chunk.content,
    tags: [p.chunk.unitKind, ...(p.chunk.unit ? [p.chunk.unit] : [])],
    embedding: vectors[i],
    filePath: p.relPath,
    lineStart: p.chunk.startLine,
    lineEnd: p.chunk.endLine,
    contentHash: p.hash,
  }));
  const ids = store.insertMany(rows);
  index.addMany(
    opts.project,
    ids.map((id, i) => ({ id, embedding: vectors[i] }))
  );
  return ids.length;
}

/**
 * Ingest a codebase into a project namespace.
 *
 * Incremental by construction: each chunk is keyed by the SHA-256 of its text,
 * so re-running against a large repo only pays for the functions that actually
 * changed. Unchanged chunks keep their existing row, embedding and index slot;
 * chunks whose text is identical but which moved in the file get a cheap
 * metadata-only line-range update; chunks that disappeared are tombstoned out
 * of the ANN index. Files that vanish entirely have all their chunks removed.
 */
export async function ingestCodebase(
  opts: IngestOptions,
  store: MemoryStore,
  index: VectorIndex
): Promise<IngestSummary> {
  const log = opts.log ?? (() => {});
  const started = Date.now();
  const root = path.resolve(opts.root);
  if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
    throw new Error(`not a directory: ${root}`);
  }

  const maxFileBytes = opts.maxFileBytes ?? 512 * 1024;
  const waveSize = opts.waveSize ?? 2000;
  const dryRun = opts.dryRun ?? false;

  const files = walkDirectory(root, {
    respectGitignore: opts.respectGitignore,
    maxFiles: opts.maxFiles,
    filter: (rel) => {
      if (opts.include?.length && !matchesAny(rel, opts.include)) return false;
      if (opts.exclude?.length && matchesAny(rel, opts.exclude)) return false;
      return true;
    },
  });

  const summary: IngestSummary = {
    root,
    project: opts.project,
    dryRun,
    filesScanned: 0,
    filesSkipped: 0,
    filesChanged: 0,
    filesDeleted: 0,
    chunksCreated: 0,
    chunksUpdated: 0,
    chunksUnchanged: 0,
    chunksDeleted: 0,
    chunksEmbedded: 0,
    totalChunks: 0,
    elapsedMs: 0,
    embedMs: 0,
    byLanguage: {},
  };

  const timing = { embedMs: 0 };
  const pending: Pending[] = [];
  const seenFiles = new Set<string>();
  let processed = 0;

  for (const file of files) {
    processed++;
    if (file.size > maxFileBytes || looksBinary(file.absPath)) {
      summary.filesSkipped++;
      continue;
    }

    let source: string;
    try {
      source = fs.readFileSync(file.absPath, "utf8");
    } catch {
      summary.filesSkipped++;
      continue;
    }

    const chunks = chunkFile(file.relPath, source, {
      fallbackLines: opts.chunkLines,
      fallbackOverlapLines: opts.chunkOverlap,
      maxChunkLines: opts.maxChunkLines,
      maxChunkChars: opts.maxChunkChars,
    });
    if (chunks.length === 0) {
      summary.filesSkipped++;
      continue;
    }
    summary.filesScanned++;

    seenFiles.add(file.relPath);
    const lang = detectLanguage(file.relPath);
    summary.byLanguage[lang] = (summary.byLanguage[lang] ?? 0) + chunks.length;
    summary.totalChunks += chunks.length;

    // Index this file's existing chunks by content hash, so an identical chunk
    // is still recognised as unchanged when it moved elsewhere in the file.
    const existing = store.chunksForFile(opts.project, file.relPath);
    const byHash = new Map<string, { id: number; line_start: number; line_end: number }>();
    for (const row of existing) {
      if (!byHash.has(row.content_hash)) byHash.set(row.content_hash, row);
    }

    let fileTouched = false;
    const matchedHashes = new Set<string>();

    for (const chunk of chunks) {
      const hash = contentHash(chunk.content);
      const prior = byHash.get(hash);
      if (prior && !matchedHashes.has(hash)) {
        matchedHashes.add(hash);
        if (prior.line_start !== chunk.startLine || prior.line_end !== chunk.endLine) {
          // Same text, new position: metadata only, no re-embedding.
          if (!dryRun) {
            store.updateChunkRange(prior.id, opts.project, chunk.startLine, chunk.endLine);
          }
          summary.chunksUpdated++;
          fileTouched = true;
        } else {
          summary.chunksUnchanged++;
        }
        continue;
      }
      matchedHashes.add(hash);
      summary.chunksCreated++;
      fileTouched = true;
      if (!dryRun) pending.push({ relPath: file.relPath, chunk, hash });

      if (pending.length >= waveSize) {
        summary.chunksEmbedded += await flushWave(pending, opts, store, index, timing);
        pending.length = 0;
        log(
          `ingest ${opts.project}: ${processed}/${files.length} files, ` +
            `${summary.chunksEmbedded} chunks embedded so far`
        );
      }
    }

    // Anything this file used to have that we did not match is gone.
    const stale = existing
      .filter((row) => !matchedHashes.has(row.content_hash))
      .map((r) => r.id);
    if (stale.length > 0) {
      summary.chunksDeleted += stale.length;
      if (!dryRun) {
        store.removeMany(stale, opts.project);
        index.removeMany(opts.project, stale);
      }
    }
    if (fileTouched || stale.length > 0) summary.filesChanged++;
  }

  // Files ingested before but no longer present: drop all of their chunks.
  for (const rel of store.ingestedFiles(opts.project)) {
    if (seenFiles.has(rel)) continue;
    const stale = store.chunksForFile(opts.project, rel).map((r) => r.id);
    if (stale.length === 0) continue;
    summary.chunksDeleted += stale.length;
    summary.filesDeleted++;
    if (!dryRun) {
      store.removeMany(stale, opts.project);
      index.removeMany(opts.project, stale);
    }
  }

  if (pending.length > 0) {
    summary.chunksEmbedded += await flushWave(pending, opts, store, index, timing);
    pending.length = 0;
  } else if (dryRun) {
    log(`dry run: ${summary.chunksCreated} chunks would be embedded`);
  }

  if (!dryRun) index.flush(opts.project);

  summary.embedMs = timing.embedMs;
  summary.elapsedMs = Date.now() - started;
  log(
    `ingest ${opts.project} ${dryRun ? "(dry run) " : ""}done: ${summary.filesScanned} files scanned, ` +
      `${summary.chunksCreated} new, ${summary.chunksUpdated} moved, ` +
      `${summary.chunksUnchanged} unchanged, ${summary.chunksDeleted} deleted, ` +
      `${summary.chunksEmbedded} embedded in ${(summary.embedMs / 1000).toFixed(1)}s`
  );
  return summary;
}

