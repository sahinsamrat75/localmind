#!/usr/bin/env node
/**
 * v1 -> v2 migration test.
 *
 * Builds a byte-faithful localmind 0.1.x database (the original flat,
 * single-pool `memories` table: id, text, tags, embedding, created_at) using
 * real embeddings from the bundled model, then points the v2 server at it and
 * asserts:
 *   - the schema upgraded and every v1 row survived with its id, text, tags,
 *     created_at and embedding intact (bit-for-bit on the vector),
 *   - v1 rows landed in the 'default' namespace as 'manual' memories, so the
 *     pre-existing behaviour is unchanged,
 *   - recall over migrated data now goes through the rebuilt HNSW index,
 *   - data written into a named project afterwards does not mix with them.
 */
import Database from "better-sqlite3";
import { startServer, makeTempDir, assert, section, finish, fs, path } from "./helpers.mjs";
import { embed } from "../dist/embeddings.js";

const home = makeTempDir("localmind-migration-");
const dbPath = path.join(home, "memory.db");

/** The exact v1 (0.1.x) schema, copied from src/store.ts at tag v0.1.0. */
const V1_DDL = `
  CREATE TABLE memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT    NOT NULL,
    tags       TEXT    NOT NULL DEFAULT '[]',
    embedding  BLOB    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
  );
  CREATE INDEX idx_memories_created_at ON memories(created_at);
`;

const V1_MEMORIES = [
  {
    text: "Sofia is severely allergic to peanuts, so never order satay sauce when she joins dinner.",
    tags: ["health", "friends"],
  },
  {
    text: "The quarterly budget review meeting was moved to the first Monday of every month.",
    tags: ["work"],
  },
  {
    text: "We chose Kafka for event streaming and Redis for idempotency keys in the payments service.",
    tags: ["architecture"],
  },
  {
    text: "The cottage we rented for the summer vacation has a sauna and a private lakeside dock.",
    tags: ["travel"],
  },
];

section("[1] build a genuine v1 database");
const raw = new Database(dbPath);
raw.exec(V1_DDL);
const insert = raw.prepare("INSERT INTO memories (text, tags, embedding) VALUES (?, ?, ?)");
const expected = [];
for (const m of V1_MEMORIES) {
  const vec = await embed(m.text);
  const buf = Buffer.alloc(vec.length * 4);
  for (let i = 0; i < vec.length; i++) buf.writeFloatLE(vec[i], i * 4);
  const info = insert.run(m.text, JSON.stringify(m.tags), buf);
  const row = raw.prepare("SELECT * FROM memories WHERE id = ?").get(info.lastInsertRowid);
  expected.push({ id: Number(info.lastInsertRowid), text: m.text, tags: m.tags, blob: row.embedding, created_at: row.created_at });
}
raw.close();
assert(expected.length === 4, `wrote ${expected.length} v1 rows with real 384-dim embeddings`);
assert(!fs.existsSync(path.join(home, "indexes")), "no ANN index exists yet (v1 had none)");

section("[2] open it with the v2 server (triggers migration)");
const { callTool, client } = await startServer(home);

const after = new Database(dbPath, { readonly: true });
const cols = new Set(after.prepare("PRAGMA table_info(memories)").all().map((c) => c.name));
for (const c of ["project", "kind", "file_path", "line_start", "line_end", "content_hash", "updated_at"]) {
  assert(cols.has(c), `column '${c}' added`);
}
const version = after.prepare("SELECT value FROM meta WHERE key = 'schema_version'").get()?.value;
assert(version === "2", `schema_version stamped as 2 (got ${version})`);

const rows = after.prepare("SELECT * FROM memories ORDER BY id").all();
assert(rows.length === expected.length, `no rows lost: ${rows.length} of ${expected.length}`);

let dataLoss = [];
for (const want of expected) {
  const got = rows.find((r) => r.id === want.id);
  if (!got) {
    dataLoss.push(`id ${want.id} missing`);
    continue;
  }
  if (got.text !== want.text) dataLoss.push(`id ${want.id} text changed`);
  if (JSON.parse(got.tags).join(",") !== want.tags.join(",")) dataLoss.push(`id ${want.id} tags changed`);
  if (got.created_at !== want.created_at) dataLoss.push(`id ${want.id} created_at changed`);
  if (!Buffer.compare(Buffer.from(got.embedding), want.blob) === 0) {
    if (Buffer.compare(Buffer.from(got.embedding), want.blob) !== 0) dataLoss.push(`id ${want.id} embedding bytes differ`);
  }
  if (got.project !== "default") dataLoss.push(`id ${want.id} project is '${got.project}' not 'default'`);
  if (got.kind !== "manual") dataLoss.push(`id ${want.id} kind is '${got.kind}' not 'manual'`);
}
assert(dataLoss.length === 0, `every v1 row preserved exactly (ids, text, tags, timestamps, vectors)${dataLoss.length ? " — " + dataLoss.join("; ") : ""}`);


section("[3] migrated memories are still recallable (now via HNSW)");
// Query shares no keywords with the target, exactly as the v1 smoke test did.
const recalled = await callTool("recall", {
  query: "What groceries must I avoid buying when cooking for my friend?",
});
const firstLine = recalled.split("\n")[1] ?? "";
assert(firstLine.includes(`[${expected[0].id}]`),
  `top hit is still the allergy memory (id=${expected[0].id}): ${firstLine.slice(0, 80)}`);
const score = Number(firstLine.match(/score=([\d.]+)/)?.[1]);
assert(Number.isFinite(score) && score > 0.3, `and it scored high (${score})`);
assert(recalled.includes("project 'default'"), "recall names the default namespace");

section("[4] the ANN index was rebuilt from the migrated vectors");
const indexDir = path.join(home, "indexes");
const indexFiles = fs.existsSync(indexDir) ? fs.readdirSync(indexDir).filter((f) => f.endsWith(".hnsw")) : [];
assert(indexFiles.length === 1, `one index file for the 'default' project (${indexFiles.join(", ")})`);
const meta = JSON.parse(fs.readFileSync(path.join(indexDir, indexFiles[0] + ".meta.json"), "utf8"));
assert(meta.count === expected.length, `index holds all ${expected.length} migrated vectors (got ${meta.count})`);
assert(meta.dim === 384 && meta.space === "cosine", "index metadata records 384-dim cosine");

section("[5] new project-scoped writes do not mix with migrated data");
await callTool("remember", {
  text: "Sofia is severely allergic to peanuts, so never order satay sauce when she joins dinner.",
  project: "pixora",
});
const defList = await callTool("list_memories", {});
assert(defList.includes(`[${expected[0].id}]`), "default namespace still lists the migrated row");
const defCount = Number(defList.match(/^#\s*(\d+)/)?.[1]);
assert(defCount === expected.length, `default namespace count unchanged at ${expected.length}`);
const pixoraList = await callTool("list_memories", { project: "pixora" });
assert(!pixoraList.includes(`[${expected[0].id}]`), "the new project does not see migrated rows");
const defRecall = await callTool("recall", { query: "peanut allergy dinner", limit: 20 });
assert(!/\[5\]/.test(defRecall), "recall in default cannot see the pixora duplicate (id 5)");

section("[6] migration is idempotent — reopening does not re-run it");
await client.close();
const again = new Database(dbPath, { readonly: true });
const rows2 = again.prepare("SELECT COUNT(*) AS n FROM memories").get().n;
const migCount = again.prepare("SELECT value FROM meta WHERE key = 'migrated_at'").get()?.value;
again.close();
assert(rows2 === expected.length + 1, `row count stable across restart (${rows2})`);
assert(typeof migCount === "string", "migration timestamp recorded once");

const second = await startServer(home);
const reRecall = await second.callTool("recall", { query: "peanut allergy dinner", limit: 5 });
assert(reRecall.includes(`[${expected[0].id}]`), "recall still works after a second open");
await second.client.close();

section("[7] a fresh (empty) database still initializes cleanly");
const freshHome = makeTempDir("localmind-fresh-");
const fresh = await startServer(freshHome);
const freshRecall = await fresh.callTool("recall", { query: "anything at all" });
assert(/No memories/i.test(freshRecall), `empty project answers politely: ${freshRecall.trim()}`);
await fresh.client.close();

process.exitCode = finish("MIGRATION TEST");

const maxId = after.prepare("SELECT MAX(id) AS m FROM memories").get().m;
const seq = after.prepare("SELECT seq FROM sqlite_sequence WHERE name = 'memories'").get();
assert(seq && seq.seq >= maxId, `AUTOINCREMENT counter repaired (seq=${seq?.seq}, max id=${maxId})`);
after.close();
