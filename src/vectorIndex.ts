import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { indexPath } from "./paths.js";
import { EMBEDDING_DIM } from "./embeddings.js";

// hnswlib-node is a native CommonJS addon loaded through bindings; createRequire
// keeps the ESM build working under NodeNext resolution without relying on CJS
// named-export interop heuristics.
const require = createRequire(import.meta.url);
const HNSW = require("hnswlib-node") as {
  HierarchicalNSW: new (space: string, dim: number) => HNSWIndex;
};

/** Minimal surface of the native index that we depend on. */
interface HNSWIndex {
  initIndex(opts: {
    maxElements: number;
    m?: number;
    efConstruction?: number;
    randomSeed?: number;
    allowReplaceDeleted?: boolean;
  }): void;
  readIndexSync(file: string, allowReplaceDeleted?: boolean): void;
  writeIndexSync(file: string): void;
  resizeIndex(newMaxElements: number): void;
  addPoint(point: number[], label: number, replaceDeleted?: boolean): void;
  markDelete(label: number): void;
  searchKnn(
    queryPoint: number[],
    numNeighbors: number,
    filter?: (label: number) => boolean
  ): { distances: number[]; neighbors: number[] };
  getIdsList(): number[];
  getPoint(label: number): number[];
  getMaxElements(): number;
  getCurrentCount(): number;
  setEf(ef: number): void;
}

/** A nearest-neighbour hit: a memory id and its cosine similarity. */
export interface VectorHit {
  id: number;
  score: number;
}

export interface VectorIndexOptions {
  indexDir: string;
  dim?: number;
  /** HNSW graph degree. */
  m?: number;
  /** Candidate-set size used while building the graph. */
  efConstruction?: number;
  /** Source of truth: the live memory ids SQLite holds for a project. */
  liveIds: (project: string) => number[];
  /** Source of truth for vectors, used when an index has to be rebuilt. */
  vectors: (project: string) => Iterable<{ id: number; embedding: Float32Array }>;
  log?: (msg: string) => void;
}

interface Entry {
  index: HNSWIndex;
  file: string;
  metaFile: string;
  /** Live labels, mirrored in memory so adds/deletes stay idempotent. */
  live: Set<number>;
  dirty: boolean;
  timer: NodeJS.Timeout | null;
}

const FLUSH_DEBOUNCE_MS = 1000;

/**
 * Per-project approximate-nearest-neighbour index over the stored embeddings.
 *
 * Design notes:
 *
 * - **One index file per project.** Project isolation is therefore structural
 *   rather than a filter predicate: a recall in project A physically cannot
 *   reach project B's vectors, and each graph stays small (faster search, less
 *   RAM) as the corpus grows.
 * - **Labels are SQLite memory ids.** No translation table, and deletes map
 *   straight onto hnswlib tombstones (`markDelete`), so `forget` is O(log n)
 *   instead of a rebuild.
 * - **SQLite is the source of truth.** The persisted index is a cache. On open
 *   we fingerprint the live id set and compare it with SQLite; on mismatch
 *   (crash between a DB write and an index flush, manual DB edit, parameter
 *   skew) we rebuild the graph from the stored embedding BLOBs.
 * - **Writes are debounced.** Rewriting a 30 MB index after every single
 *   `remember` would dominate runtime, so mutations mark the index dirty and a
 *   short timer coalesces them into one atomic write (tmp file + rename).
 *   Anything that must be durable now — end of an ingestion, shutdown — calls
 *   `flush()` explicitly.
 */
export class VectorIndex {
  private readonly indexDir: string;
  private readonly dim: number;
  private readonly m: number;
  private readonly efConstruction: number;
  private readonly liveIds: (project: string) => number[];
  private readonly vectors: (project: string) => Iterable<{ id: number; embedding: Float32Array }>;
  private readonly logger: ((msg: string) => void) | undefined;
  private readonly entries = new Map<string, Entry>();
  private closed = false;

  constructor(options: VectorIndexOptions) {
    this.indexDir = options.indexDir;
    this.dim = options.dim ?? EMBEDDING_DIM;
    this.m = options.m ?? 16;
    this.efConstruction = options.efConstruction ?? 200;
    this.liveIds = options.liveIds;
    this.vectors = options.vectors;
    this.logger = options.log;
    fs.mkdirSync(this.indexDir, { recursive: true });
  }

  private log(msg: string): void {
    this.logger?.(msg);
  }

  /** Fingerprint a sorted id list; used to detect a stale persisted index. */
  static digest(ids: number[]): string {
    const h = crypto.createHash("sha256");
    for (const id of ids) h.update(String(id));
    return h.digest("hex").slice(0, 16);
  }

  /** Open (loading or building as needed) the index for a project. */
  private entry(project: string): Entry {
    const existing = this.entries.get(project);
    if (existing) return existing;
    if (this.closed) throw new Error("vector index is closed");

    const file = indexPath(this.indexDir, project);
    const metaFile = `${file}.meta.json`;
    const expected = this.liveIds(project).slice().sort((a, b) => a - b);

    let entry: Entry | null = null;
    if (fs.existsSync(file) && fs.existsSync(metaFile)) {
      entry = this.tryLoad(project, file, metaFile, expected);
    }
    if (!entry) entry = this.build(project, file, metaFile, expected);

    this.entries.set(project, entry);
    return entry;
  }

  /** Load a persisted index and verify it still agrees with SQLite. */
  private tryLoad(project: string, file: string, metaFile: string, expected: number[]): Entry | null {
    try {
      const meta = JSON.parse(fs.readFileSync(metaFile, "utf8")) as {
        dim: number;
        space: string;
        m: number;
        efConstruction: number;
        idsDigest: string;
      };
      if (
        meta.dim !== this.dim ||
        meta.space !== "cosine" ||
        meta.m !== this.m ||
        meta.efConstruction !== this.efConstruction
      ) {
        this.log(`index parameters changed for '${project}'; rebuilding`);
        return null;
      }
      const index = new HNSW.HierarchicalNSW("cosine", this.dim);
      index.readIndexSync(file, true);
      const loaded = (index.getIdsList() as number[]).slice().sort((a, b) => a - b);
      if (VectorIndex.digest(loaded) !== VectorIndex.digest(expected)) {
        this.log(
          `index for '${project}' disagrees with sqlite (${loaded.length} vs ${expected.length} vectors); rebuilding`
        );
        return null;
      }
      return { index, file, metaFile, live: new Set(loaded), dirty: false, timer: null };
    } catch (err) {
      this.log(`could not read index for '${project}' (${(err as Error).message}); rebuilding`);
      return null;
    }
  }

  /** Build a fresh index from the embeddings stored in SQLite. */
  private build(project: string, file: string, metaFile: string, expected: number[]): Entry {
    const index = new HNSW.HierarchicalNSW("cosine", this.dim);
    index.initIndex({
      maxElements: capacityFor(expected.length),
      m: this.m,
      efConstruction: this.efConstruction,
      allowReplaceDeleted: true,
    });
    const live = new Set<number>();
    if (expected.length > 0) {
      const expectedSet = new Set(expected);
      for (const { id, embedding } of this.vectors(project)) {
        if (!expectedSet.has(id)) continue;
        index.addPoint(Array.from(embedding), id, true);
        live.add(id);
      }
    }
    const entry: Entry = { index, file, metaFile, live, dirty: true, timer: null };
    this.persist(entry, project);
    return entry;
  }

  /** Grow the underlying graph before it runs out of slots. */
  private ensureCapacity(entry: Entry, incoming: number): void {
    let max = entry.index.getMaxElements();
    while (entry.index.getCurrentCount() + incoming > max) {
      max = Math.ceil(max * 1.6) + 1024;
      entry.index.resizeIndex(max);
    }
  }

  /** Nearest neighbours of a query vector within one project. */
  search(project: string, query: Float32Array, k: number): VectorHit[] {
    const entry = this.entry(project);
    const n = entry.live.size;
    if (n === 0 || k <= 0) return [];
    const want = Math.min(k, n);
    entry.index.setEf(efFor(want));
    const res = entry.index.searchKnn(Array.from(query), want);
    const out: VectorHit[] = [];
    for (let i = 0; i < res.neighbors.length; i++) {
      // hnswlib's cosine space reports distance = 1 - cosine similarity.
      out.push({ id: res.neighbors[i], score: 1 - res.distances[i] });
    }
    return out;
  }

  /** Add one vector under a memory id. Idempotent per label. */
  add(project: string, id: number, embedding: Float32Array): void {
    this.addMany(project, [{ id, embedding }]);
  }

  /** Add many vectors with a single capacity check. */
  addMany(project: string, items: Iterable<{ id: number; embedding: Float32Array }>): void {
    const entry = this.entry(project);
    const pending: Array<{ id: number; vec: number[] }> = [];
    for (const item of items) {
      if (entry.live.has(item.id)) continue;
      pending.push({ id: item.id, vec: Array.from(item.embedding) });
    }
    if (pending.length === 0) return;
    this.ensureCapacity(entry, pending.length);
    for (const p of pending) {
      entry.index.addPoint(p.vec, p.id, true);
      entry.live.add(p.id);
    }
    this.markDirty(project, entry);
  }

  /** Tombstone a label. Safe to call for ids that are not indexed. */
  remove(project: string, id: number): void {
    this.removeMany(project, [id]);
  }

  removeMany(project: string, ids: Iterable<number>): void {
    const entry = this.entry(project);
    let changed = 0;
    for (const id of ids) {
      if (!entry.live.has(id)) continue;
      entry.index.markDelete(id);
      entry.live.delete(id);
      changed++;
    }
    if (changed > 0) this.markDirty(project, entry);
  }

  /** Number of live vectors indexed for a project. */
  size(project: string): number {
    return this.entry(project).live.size;
  }

  /** Raw graph stats for diagnostics and the benchmark. */
  stats(project: string): {
    live: number;
    slots: number;
    capacity: number;
    file: string;
    bytes: number;
  } {
    const entry = this.entry(project);
    let bytes = 0;
    try {
      bytes = fs.statSync(entry.file).size;
    } catch {
      bytes = 0;
    }
    return {
      live: entry.live.size,
      slots: entry.index.getCurrentCount(),
      capacity: entry.index.getMaxElements(),
      file: entry.file,
      bytes,
    };
  }

  /** Delete a project's index files entirely (project dropped / re-ingested). */
  drop(project: string): void {
    const entry = this.entries.get(project);
    if (entry?.timer) clearTimeout(entry.timer);
    this.entries.delete(project);
    const file = indexPath(this.indexDir, project);
    fs.rmSync(file, { force: true });
    fs.rmSync(`${file}.meta.json`, { force: true });
    fs.rmSync(`${file}.tmp`, { force: true });
  }

  private markDirty(project: string, entry: Entry): void {
    entry.dirty = true;
    if (entry.timer) return;
    entry.timer = setTimeout(() => {
      entry.timer = null;
      if (entry.dirty && !this.closed) {
        try {
          this.persist(entry, project);
        } catch (err) {
          this.log(`index flush failed for '${project}': ${(err as Error).message}`);
        }
      }
    }, FLUSH_DEBOUNCE_MS);
    // Never hold the event loop open just for a pending index write.
    entry.timer.unref?.();
  }

  /** Write one project's index to disk now. Omit the argument to flush all. */
  flush(project?: string): void {
    if (project === undefined) {
      this.flushAll();
      return;
    }
    const entry = this.entries.get(project);
    if (!entry) return;
    if (entry.timer) {
      clearTimeout(entry.timer);
      entry.timer = null;
    }
    if (entry.dirty) this.persist(entry, project);
  }

  /** Write every loaded, dirty index to disk. */
  flushAll(): void {
    for (const [project, entry] of this.entries) {
      if (entry.timer) {
        clearTimeout(entry.timer);
        entry.timer = null;
      }
      if (entry.dirty) {
        try {
          this.persist(entry, project);
        } catch (err) {
          this.log(`index flush failed for '${project}': ${(err as Error).message}`);
        }
      }
    }
  }

  private persist(entry: Entry, project: string): void {
    // hnswlib round-trips awkwardly for a never-populated index, and an empty
    // project has nothing worth caching: skip it and let the next touch build a
    // fresh one (which costs nothing when there are no vectors to add).
    if (entry.live.size === 0 && entry.index.getCurrentCount() === 0) {
      entry.dirty = false;
      return;
    }
    const tmp = `${entry.file}.tmp`;
    entry.index.writeIndexSync(tmp);
    fs.renameSync(tmp, entry.file);
    const meta = {
      dim: this.dim,
      space: "cosine",
      m: this.m,
      efConstruction: this.efConstruction,
      count: entry.live.size,
      idsDigest: VectorIndex.digest([...entry.live].sort((a, b) => a - b)),
      project,
      writtenAt: new Date().toISOString(),
    };
    fs.writeFileSync(entry.metaFile, JSON.stringify(meta));
    entry.dirty = false;
  }

  /** Flush everything pending and release the native indexes. */
  close(): void {
    if (this.closed) return;
    this.flushAll();
    for (const entry of this.entries.values()) {
      if (entry.timer) clearTimeout(entry.timer);
    }
    this.entries.clear();
    this.closed = true;
  }
}

/** Initial slot count, with headroom so small inserts avoid a resize. */
function capacityFor(expected: number): number {
  return Math.max(1024, Math.ceil(expected * 1.25) + 256);
}

/**
 * `ef` is the search-time candidate queue size: higher is more accurate and
 * slower. Scale it with how many neighbours were asked for, bounded so a wide
 * recall stays cheap.
 */
function efFor(k: number): number {
  return Math.max(64, Math.min(512, k * 8));
}

/** Path helpers used by tests and the benchmark script. */
export function indexFileFor(indexDir: string, project: string): string {
  return indexPath(indexDir, project);
}

export function indexDirFiles(indexDir: string): string[] {
  try {
    return fs.readdirSync(indexDir).map((f) => path.join(indexDir, f));
  } catch {
    return [];
  }
}

