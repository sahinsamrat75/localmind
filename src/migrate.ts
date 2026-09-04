import type Database from "better-sqlite3";

/** Current on-disk schema version, tracked in the `meta` table. */
export const SCHEMA_VERSION = 2;

/** Canonical v2 shape of the memories table. */
export const MEMORIES_DDL = `
      CREATE TABLE memories (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        project      TEXT    NOT NULL DEFAULT 'default',
        kind         TEXT    NOT NULL DEFAULT 'manual',
        text         TEXT    NOT NULL,
        tags         TEXT    NOT NULL DEFAULT '[]',
        embedding    BLOB    NOT NULL,
        file_path    TEXT,
        line_start   INTEGER,
        line_end     INTEGER,
        content_hash TEXT,
        created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
      );
`;

/**
 * Indexes backing the project-scoped access patterns. `project` leads every
 * composite index because every read in the server is namespace-scoped.
 */
export function ensureIndexes(db: Database.Database): void {
  db.exec(`
      CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
      CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project, id);
      CREATE INDEX IF NOT EXISTS idx_memories_project_file ON memories(project, file_path);
      CREATE INDEX IF NOT EXISTS idx_memories_project_hash ON memories(project, content_hash);
  `);
}

/** Read the stored schema version, treating a v1 database (no meta table) as 1. */
function readVersion(db: Database.Database): number {
  const hasMeta = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'meta'")
    .get();
  if (!hasMeta) return 1;
  const row = db.prepare("SELECT value FROM meta WHERE key = 'schema_version'").get() as
    | { value: string }
    | undefined;
  if (!row) return 1;
  const n = Number(row.value);
  return Number.isFinite(n) ? n : 1;
}

function writeVersion(db: Database.Database, version: number): void {
  db.prepare(
    "INSERT INTO meta (key, value) VALUES ('schema_version', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
  ).run(String(version));
}

const NOW = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')";


/**
 * Schema migration for the memories table.
 *
 * v1 (localmind 0.1.x) had a flat single-pool table:
 *   memories(id, text, tags, embedding, created_at)
 *
 * v2 adds project namespacing, a manual/code `kind`, and the code-chunk
 * columns (file_path, line_start, line_end, content_hash) plus updated_at.
 *
 * The migration is non-destructive: ids are preserved verbatim (they double as
 * HNSW labels and appear in user-visible output), and every v1 row is moved
 * into the `default` project as a `manual` memory, so an existing v1 database
 * keeps working with exactly the semantics it had before.
 *
 * Rather than issuing a series of ALTER TABLEs (which cannot add columns with a
 * non-constant default such as strftime(...), and which would leave the table
 * shape depending on which v1 revision a user happened to have), we build the
 * canonical v2 table, copy the rows across, and swap. The whole thing runs in
 * one transaction, so a crash mid-migration leaves the v1 data untouched.
 */
export function migrate(db: Database.Database): void {
  const version = readVersion(db);
  if (version >= SCHEMA_VERSION) {
    ensureIndexes(db);
    return;
  }

  const exists = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memories'")
    .get() as { name: string } | undefined;

  if (!exists) {
    // Fresh database: just create the current schema.
    db.exec(MEMORIES_DDL);
    ensureIndexes(db);
    writeVersion(db, SCHEMA_VERSION);
    return;
  }

  const cols = new Set(
    (db.prepare("PRAGMA table_info(memories)").all() as Array<{ name: string }>).map((c) => c.name)
  );

  // A table that already carries the v2 columns but has an outdated version
  // stamp (e.g. a downgrade/upgrade round-trip) only needs the stamp fixed.
  const isV2Shape = ["project", "kind", "content_hash", "updated_at"].every((c) => cols.has(c));
  if (isV2Shape) {
    ensureIndexes(db);
    writeVersion(db, SCHEMA_VERSION);
    return;
  }

  if (!cols.has("text") || !cols.has("embedding")) {
    throw new Error(
      "memories table exists but is missing v1 columns (text/embedding); refusing to migrate"
    );
  }

  const v1Count = (db.prepare("SELECT COUNT(*) AS n FROM memories").get() as { n: number }).n;

  db.transaction(() => {
    db.exec("DROP TABLE IF EXISTS memories_v2;");
    db.exec(MEMORIES_DDL.replace("CREATE TABLE memories", "CREATE TABLE memories_v2"));

    // Copy across only the columns v1 actually has, defaulting the new ones.
    const hasTags = cols.has("tags");
    const hasCreatedAt = cols.has("created_at");
    const createdAtExpr = hasCreatedAt ? `COALESCE(created_at, ${NOW})` : NOW;
    db.exec(
      `INSERT INTO memories_v2
         (id, project, kind, text, tags, embedding, created_at, updated_at)
       SELECT id,
              'default',
              'manual',
              text,
              ${hasTags ? "COALESCE(tags, '[]')" : "'[]'"},
              embedding,
              ${createdAtExpr},
              ${createdAtExpr}
         FROM memories;`
    );

    const copied = (db.prepare("SELECT COUNT(*) AS n FROM memories_v2").get() as { n: number }).n;
    if (copied !== v1Count) {
      throw new Error(`migration row-count mismatch: copied ${copied} of ${v1Count}`);
    }

    db.exec("DROP TABLE memories;");
    db.exec("ALTER TABLE memories_v2 RENAME TO memories;");

    // RENAME can leave the AUTOINCREMENT counter registered under the old
    // name, which would let new inserts reuse existing ids. Re-point it
    // explicitly. sqlite_sequence only exists once an AUTOINCREMENT table has
    // actually held a row, so guard for the empty-v1-database case.
    const maxId = (db.prepare("SELECT COALESCE(MAX(id), 0) AS m FROM memories").get() as {
      m: number;
    }).m;
    const hasSeq = db
      .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'")
      .get();
    if (hasSeq) {
      db.prepare("DELETE FROM sqlite_sequence WHERE name IN ('memories', 'memories_v2')").run();
      db.prepare("INSERT INTO sqlite_sequence (name, seq) VALUES ('memories', ?)").run(maxId);
    }

    ensureIndexes(db);
    db.prepare(
      "INSERT INTO meta (key, value) VALUES ('migrated_from_version', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
    ).run(String(version));
    db.prepare(
      "INSERT INTO meta (key, value) VALUES ('migrated_at', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value"
    ).run(new Date().toISOString());
    writeVersion(db, SCHEMA_VERSION);
  })();

  console.error(`[localmind] migrated v${version} schema to v${SCHEMA_VERSION} (${v1Count} memories -> project 'default')`);
}
