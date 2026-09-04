import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import { migrate } from "./migrate.js";

/** Namespace used when a caller does not name a project (v1 compatibility). */
export const DEFAULT_PROJECT = "default";
/** Hand-written memory (the original `remember` tool). */
export const KIND_MANUAL = "manual";
/** Memory produced by codebase ingestion. */
export const KIND_CODE = "code";

/** A memory as returned by queries (embedding excluded from user-facing shape). */
export interface Memory {
  id: number;
  project: string;
  kind: string;
  text: string;
  tags: string[];
  file_path: string | null;
  line_start: number | null;
  line_end: number | null;
  content_hash: string | null;
  created_at: string;
  updated_at: string;
}

/** Internal row shape including the raw embedding blob. */
interface MemoryRow {
  id: number;
  project: string;
  kind: string;
  text: string;
  tags: string;
  embedding: Buffer;
  file_path: string | null;
  line_start: number | null;
  line_end: number | null;
  content_hash: string | null;
  created_at: string;
  updated_at: string | null;
}

/** Fields accepted by {@link MemoryStore.insert}. */
export interface InsertInput {
  text: string;
  tags?: string[];
  embedding: Float32Array;
  project?: string;
  kind?: string;
  filePath?: string | null;
  lineStart?: number | null;
  lineEnd?: number | null;
  contentHash?: string | null;
}

export interface InsertResult {
  id: number;
  created_at: string;
}

/** Lightweight chunk descriptor used by incremental ingestion. */
export interface ChunkRow {
  id: number;
  content_hash: string;
  line_start: number;
  line_end: number;
}

/** A recall hit: memory plus its semantic similarity score. */
export interface ScoredMemory extends Memory {
  score: number;
}

/**
 * SQLite-backed storage for memories and their embeddings.
 *
 * SQLite owns the metadata (text, tags, project, file path, line range,
 * content hash, timestamps) and is also the durable copy of every embedding,
 * which lets the HNSW index be rebuilt from scratch if its persisted file is
 * lost or stale. Similarity search itself is delegated to the ANN index in
 * `vectorIndex.ts` — this class never scans all rows at query time.
 *
 * Every row belongs to exactly one `project` namespace; all read paths here
 * require one so cross-project leakage is structurally impossible.
 */
export class MemoryStore {
  private db: Database.Database;

  constructor(dbPath: string) {
    // Ensure the containing directory exists (e.g. ~/.localmind).
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });

    this.db = new Database(dbPath);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("synchronous = NORMAL");

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
    `);

    migrate(this.db);
  }

  /** Store a memory with its (normalized) embedding. Returns the new row id. */
  insert(input: InsertInput): InsertResult {
    const buf = float32ToBuffer(input.embedding);
    const info = this.db
      .prepare(
        `INSERT INTO memories
           (project, kind, text, tags, embedding, file_path, line_start, line_end, content_hash)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        input.project ?? DEFAULT_PROJECT,
        input.kind ?? KIND_MANUAL,
        input.text,
        JSON.stringify(input.tags ?? []),
        buf,
        input.filePath ?? null,
        input.lineStart ?? null,
        input.lineEnd ?? null,
        input.contentHash ?? null
      );
    const row = this.db
      .prepare("SELECT created_at FROM memories WHERE id = ?")
      .get(info.lastInsertRowid) as { created_at: string };
    return { id: Number(info.lastInsertRowid), created_at: row.created_at };
  }

  /** Insert many memories inside a single transaction (used by ingestion). */
  insertMany(inputs: InsertInput[]): number[] {
    const stmt = this.db.prepare(
      `INSERT INTO memories
         (project, kind, text, tags, embedding, file_path, line_start, line_end, content_hash)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
    );
    const ids: number[] = [];
    const tx = this.db.transaction((rows: InsertInput[]) => {
      for (const r of rows) {
        const info = stmt.run(
          r.project ?? DEFAULT_PROJECT,
          r.kind ?? KIND_CODE,
          r.text,
          JSON.stringify(r.tags ?? []),
          float32ToBuffer(r.embedding),
          r.filePath ?? null,
          r.lineStart ?? null,
          r.lineEnd ?? null,
          r.contentHash ?? null
        );
        ids.push(Number(info.lastInsertRowid));
      }
    });
    tx(inputs);
    return ids;
  }

  /** All memories in a project, newest first, optionally filtered by tag/kind. */
  list(project: string, limit: number, tag?: string, kind?: string): Memory[] {
    const rows = this.db
      .prepare(
        `SELECT id, project, kind, text, tags, file_path, line_start, line_end,
                content_hash, created_at, updated_at
           FROM memories WHERE project = ? ORDER BY created_at DESC, id DESC`
      )
      .all(project) as MemoryRow[];
    const out: Memory[] = [];
    for (const row of rows) {
      const tags: string[] = parseTagsJson(row.tags);
      if (tag !== undefined && !tags.includes(tag)) continue;
      if (kind !== undefined && row.kind !== kind) continue;
      out.push(rowToMemory(row));
      if (out.length >= limit) break;
    }
    return out;
  }

  /** Get a single memory by id within a project, or undefined. */
  get(id: number, project: string): Memory | undefined {
    const row = this.db
      .prepare(
        `SELECT id, project, kind, text, tags, file_path, line_start, line_end,
                content_hash, created_at, updated_at
           FROM memories WHERE id = ? AND project = ?`
      )
      .get(id, project) as MemoryRow | undefined;
    return row ? rowToMemory(row) : undefined;
  }

  /** Fetch several memories by id, returned in the order the ids were given. */
  getByIds(ids: number[], project: string): Memory[] {
    if (ids.length === 0) return [];
    const placeholders = ids.map(() => "?").join(",");
    const rows = this.db
      .prepare(
        `SELECT id, project, kind, text, tags, file_path, line_start, line_end,
                content_hash, created_at, updated_at
           FROM memories WHERE project = ? AND id IN (${placeholders})`
      )
      .all(project, ...ids) as MemoryRow[];
    const byId = new Map(rows.map((r) => [r.id, rowToMemory(r)]));
    return ids.map((id) => byId.get(id)).filter((m): m is Memory => m !== undefined);
  }

  /**
   * Delete a memory by id, but only if it belongs to `project`. Returns the
   * deleted memory (so the caller can evict it from the ANN index) or undefined
   * when no such memory exists in that namespace.
   */
  remove(id: number, project: string): Memory | undefined {
    const existing = this.get(id, project);
    if (!existing) return undefined;
    this.db.prepare("DELETE FROM memories WHERE id = ? AND project = ?").run(id, project);
    return existing;
  }

  /** Delete a batch of ids within a project. Returns how many rows went away. */
  removeMany(ids: number[], project: string): number {
    if (ids.length === 0) return 0;
    const stmt = this.db.prepare("DELETE FROM memories WHERE id = ? AND project = ?");
    const tx = this.db.transaction((list: number[]) => {
      let n = 0;
      for (const id of list) n += stmt.run(id, project).changes;
      return n;
    });
    return tx(ids);
  }

  count(project: string): number {
    return (
      this.db.prepare("SELECT COUNT(*) AS n FROM memories WHERE project = ?").get(project) as {
        n: number;
      }
    ).n;
  }

  countAll(): number {
    return (this.db.prepare("SELECT COUNT(*) AS n FROM memories").get() as { n: number }).n;
  }

  /** Distinct projects that currently hold at least one memory. */
  projects(): string[] {
    const rows = this.db
      .prepare("SELECT DISTINCT project FROM memories ORDER BY project")
      .all() as Array<{ project: string }>;
    return rows.map((r) => r.project);
  }

  /** Per-project memory counts. */
  projectCounts(): Array<{ project: string; count: number }> {
    return this.db
      .prepare(
        `SELECT project, COUNT(*) AS count FROM memories GROUP BY project ORDER BY project`
      )
      .all() as Array<{ project: string; count: number }>;
  }

  /** Live ids in a project, ascending — used to fingerprint the ANN index. */
  idsForProject(project: string): number[] {
    const rows = this.db
      .prepare("SELECT id FROM memories WHERE project = ? ORDER BY id")
      .all(project) as Array<{ id: number }>;
    return rows.map((r) => r.id);
  }

  /**
   * Stream every (id, embedding) pair in a project. Used to rebuild the ANN
   * index from SQLite when the persisted index is missing or stale.
   */
  *embeddingsForProject(project: string): Generator<{ id: number; embedding: Float32Array }> {
    const stmt = this.db.prepare(
      "SELECT id, embedding FROM memories WHERE project = ? ORDER BY id"
    );
    const rows = stmt.iterate(project) as IterableIterator<{ id: number; embedding: Buffer }>;
    for (const row of rows) {
      yield { id: row.id, embedding: bufferToFloat32(row.embedding) };
    }
  }
  /** Existing chunks for one file in one project (incremental ingestion). */
  chunksForFile(project: string, filePath: string): ChunkRow[] {
    return this.db
      .prepare(
        `SELECT id, content_hash, line_start, line_end FROM memories
          WHERE project = ? AND file_path = ? AND kind = ?`
      )
      .all(project, filePath, KIND_CODE) as ChunkRow[];
  }

  /** Distinct ingested file paths for a project. */
  ingestedFiles(project: string): string[] {
    const rows = this.db
      .prepare(
        `SELECT DISTINCT file_path FROM memories
          WHERE project = ? AND kind = ? AND file_path IS NOT NULL`
      )
      .all(project, KIND_CODE) as Array<{ file_path: string }>;
    return rows.map((r) => r.file_path);
  }

  /** Refresh a chunk's line range without touching its text or embedding. */
  updateChunkRange(id: number, project: string, lineStart: number, lineEnd: number): void {
    this.db
      .prepare(
        `UPDATE memories SET line_start = ?, line_end = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
           WHERE id = ? AND project = ?`
      )
      .run(lineStart, lineEnd, id, project);
  }

  /** Record an arbitrary key/value in the schema meta table. */
  setMeta(key: string, value: string): void {
    this.db
      .prepare(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
      )
      .run(key, value);
  }

  getMeta(key: string): string | undefined {
    const row = this.db.prepare("SELECT value FROM meta WHERE key = ?").get(key) as
      | { value: string }
      | undefined;
    return row?.value;
  }

  /** Drop every memory in a project. Returns the deleted ids. */
  clearProject(project: string): number[] {
    const ids = this.idsForProject(project);
    this.db.prepare("DELETE FROM memories WHERE project = ?").run(project);
    return ids;
  }

  close(): void {
    this.db.close();
  }
}

/** Materialise a user-facing memory from a raw row. */
function rowToMemory(row: MemoryRow): Memory {
  return {
    id: row.id,
    project: row.project,
    kind: row.kind,
    text: row.text,
    tags: parseTagsJson(row.tags),
    file_path: row.file_path,
    line_start: row.line_start,
    line_end: row.line_end,
    content_hash: row.content_hash,
    created_at: row.created_at,
    updated_at: row.updated_at ?? row.created_at,
  };
}

/** Tags are stored as JSON; tolerate a corrupt value rather than throwing. */
function parseTagsJson(raw: string): string[] {
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((t): t is string => typeof t === "string") : [];
  } catch {
    return [];
  }
}




/** Encode a Float32Array as a little-endian byte buffer for SQLite storage. */
function float32ToBuffer(arr: Float32Array): Buffer {
  const buf = Buffer.allocUnsafe(arr.length * 4);
  for (let i = 0; i < arr.length; i++) {
    buf.writeFloatLE(arr[i], i * 4);
  }
  return buf;
}

/** Decode a stored BLOB back into a Float32Array. */
function bufferToFloat32(buf: Buffer): Float32Array {
  const out = new Float32Array(buf.length / 4);
  for (let i = 0; i < out.length; i++) {
    out[i] = buf.readFloatLE(i * 4);
  }
  return out;
}
