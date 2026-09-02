import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";

/** A memory as returned by queries (embedding excluded from user-facing shape). */
export interface Memory {
  id: number;
  text: string;
  tags: string[];
  created_at: string;
}

/** Internal row shape including the raw embedding blob. */
interface MemoryRow {
  id: number;
  text: string;
  tags: string;
  embedding: Buffer;
  created_at: string;
}

export interface InsertResult {
  id: number;
  created_at: string;
}

/** A recall hit: memory plus its semantic similarity score. */
export interface ScoredMemory extends Memory {
  score: number;
}

/**
 * SQLite-backed storage for memories and their embeddings.
 *
 * Embeddings are stored as little-endian float32 BLOBs. Semantic search is done
 * by scanning all rows and computing cosine similarity in-process, which is
 * fast and dependency-free for the local-first scale this server targets
 * (tens of thousands of memories).
 */
export class MemoryStore {
  private db: Database.Database;

  private insertStmt: Database.Statement;
  private getAllStmt: Database.Statement;
  private getStmt: Database.Statement;
  private deleteStmt: Database.Statement;
  private countStmt: Database.Statement;

  constructor(dbPath: string) {
    // Ensure the containing directory exists (e.g. ~/.localmind).
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });

    this.db = new Database(dbPath);
    this.db.pragma("journal_mode = WAL");

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memories (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        text       TEXT    NOT NULL,
        tags       TEXT    NOT NULL DEFAULT '[]',
        embedding  BLOB    NOT NULL,
        created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
      );
      CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
    `);

    this.insertStmt = this.db.prepare(
      "INSERT INTO memories (text, tags, embedding) VALUES (?, ?, ?)"
    );
    this.getAllStmt = this.db.prepare(
      "SELECT id, text, tags, embedding, created_at FROM memories ORDER BY created_at DESC, id DESC"
    );
    this.getStmt = this.db.prepare(
      "SELECT id, text, tags, embedding, created_at FROM memories WHERE id = ?"
    );
    this.deleteStmt = this.db.prepare("DELETE FROM memories WHERE id = ?");
    this.countStmt = this.db.prepare("SELECT COUNT(*) AS n FROM memories");
  }

  /** Store a memory with its (normalized) embedding. Returns the new row id. */
  insert(text: string, tags: string[], embedding: Float32Array): InsertResult {
    const buf = float32ToBuffer(embedding);
    const info = this.insertStmt.run(text, JSON.stringify(tags), buf);
    const row = this.getStmt.get(info.lastInsertRowid) as MemoryRow;
    return { id: row.id, created_at: row.created_at };
  }

  /** Fetch every memory with its embedding (for in-process similarity search). */
  getAllWithEmbeddings(): Array<{ memory: Memory; embedding: Float32Array }> {
    const rows = this.getAllStmt.all() as MemoryRow[];
    return rows.map((row) => ({
      memory: { id: row.id, text: row.text, tags: JSON.parse(row.tags), created_at: row.created_at },
      embedding: bufferToFloat32(row.embedding),
    }));
  }

  /** List memories, optionally filtered by tag, newest first, up to `limit`. */
  list(limit: number, tag?: string): Memory[] {
    const rows = this.getAllStmt.all() as MemoryRow[];
    const out: Memory[] = [];
    for (const row of rows) {
      const tags: string[] = JSON.parse(row.tags);
      if (tag !== undefined && !tags.includes(tag)) continue;
      out.push({ id: row.id, text: row.text, tags, created_at: row.created_at });
      if (out.length >= limit) break;
    }
    return out;
  }

  /** Get a single memory by id (without its embedding), or undefined. */
  get(id: number): Memory | undefined {
    const row = this.getStmt.get(id) as MemoryRow | undefined;
    if (!row) return undefined;
    return { id: row.id, text: row.text, tags: JSON.parse(row.tags), created_at: row.created_at };
  }

  /** Delete a memory by id. Returns true if a row was removed. */
  remove(id: number): boolean {
    const info = this.deleteStmt.run(id);
    return info.changes > 0;
  }

  count(): number {
    return (this.countStmt.get() as { n: number }).n;
  }

  close(): void {
    this.db.close();
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
