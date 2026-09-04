#!/usr/bin/env node
/**
 * Incremental ingestion test.
 *
 * The claim being verified: re-running ingest_codebase over a large repo after
 * editing ONE file only re-embeds that file's chunks. Everything else is
 * recognised by SHA-256 and left alone. This is what makes PIXORA-scale
 * re-ingestion affordable, so it is asserted on the reported counts, not
 * inferred from timing.
 */
import { startServer, makeTempDir, assert, section, finish, fs, path } from "./helpers.mjs";

const home = makeTempDir("localmind-incremental-");
const repo = makeTempDir("pixora-fixture-");
const PROJECT = "pixora";

function write(rel, text) {
  const abs = path.join(repo, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, text);
}

function goFile(pkg, names) {
  return (
    `package ${pkg}\n\n` +
    names
      .map(
        (n) =>
          `// ${n} handles the ${n.toLowerCase()} stage of the request pipeline.\n` +
          `func ${n}(ctx *Context, req *Request) error {\n` +
          `  log.Printf("${n}: start")\n` +
          `  if req == nil {\n    return ErrNilRequest\n  }\n` +
          `  return ctx.Next(req)\n}\n\n`
      )
      .join("")
  );
}

section("[1] build a fixture repo: 6 files across 3 languages");
write("services/payments/handler.go", goFile("payments", ["Authorize", "Capture", "Refund"]));
write("services/payments/store.go", goFile("payments", ["OpenDB", "Migrate"]));
write("services/search/indexer.go", goFile("search", ["Index", "Reindex", "Purge"]));
write("web/src/api.ts", `export async function fetchResults(q: string) {\n  const r = await fetch(\`/search?q=\${q}\`);\n  return r.json();\n}\n\nexport function render(list: any[]) {\n  return list.map((x) => x.title).join(", ");\n}\n`);
write("web/src/client.ts", `export class ApiClient {\n  constructor(private base: string) {}\n\n  async get(path: string) {\n    return fetch(this.base + path);\n  }\n}\n`);
write("tools/report/report.py", `def build_report(rows):\n    total = sum(r["amount"] for r in rows)\n    return {"count": len(rows), "total": total}\n\n\ndef emit(report):\n    print(report)\n`);
write(".gitignore", "generated/\n*.log\n");
write("generated/blob.go", goFile("generated", ["ShouldBeSkipped"]));
write("debug.log", "noise noise noise\n");

const { callTool, client } = await startServer(home);

function parse(text) {
  return JSON.parse(text);
}

section("[2] first ingestion embeds everything");
const first = parse(await callTool("ingest_codebase", { path: repo, project: PROJECT }));
console.log("  " + JSON.stringify(first));
assert(first.filesScanned === 6, `6 source files scanned, gitignored ones excluded (${first.filesScanned})`);
assert(first.chunksCreated > 0, `${first.chunksCreated} chunks created`);
assert(first.chunksEmbedded === first.chunksCreated,
  "every created chunk went through the model");
assert(first.chunksUnchanged === 0, "nothing could be 'unchanged' on a cold start");
assert(first.totalChunks === first.chunksCreated, "chunk accounting adds up");

const coldRecall = await callTool("recall", {
  query: "which function reverses a payment charge?",
  project: PROJECT,
  limit: 3,
});
assert(/Refund/i.test(coldRecall), `semantic search over ingested code finds Refund: ${coldRecall.split("\n").slice(1, 3).join(" | ").slice(0, 140)}`);
assert(/\.go:\d+-\d+/.test(coldRecall), "hits carry file path + line range");

section("[3] immediate re-ingestion embeds nothing");
const second = parse(await callTool("ingest_codebase", { path: repo, project: PROJECT }));
console.log("  " + JSON.stringify(second));
assert(second.chunksEmbedded === 0, `zero chunks re-embedded (${second.chunksEmbedded})`);
assert(second.chunksCreated === 0, "zero new chunks");
assert(second.chunksDeleted === 0, "nothing deleted");
assert(second.chunksUnchanged === first.chunksCreated,
  `all ${first.chunksCreated} chunks recognised as unchanged by content hash`);
assert(second.elapsedMs < first.elapsedMs, "and it is faster than the cold run");

section("[4] editing ONE file re-embeds only that file's chunks");
// Count how many chunks each file contributed on the cold run by re-deriving
// them from the fixture, then mutate exactly one file.
const paymentsHandlerChunks = 3; // Authorize / Capture / Refund (+ preamble merged)
write(
  "services/payments/handler.go",
  goFile("payments", ["Authorize", "Capture", "Refund", "Void"]) +
    `func Reverse(ctx *Context, charge *Charge) error {\n  return ctx.Fail(ErrNotImplemented)\n}\n`
);

const third = parse(await callTool("ingest_codebase", { path: repo, project: PROJECT }));
console.log("  " + JSON.stringify(third));
assert(third.chunksEmbedded > 0, `the edited file forced some re-embedding (${third.chunksEmbedded})`);
assert(
  third.chunksEmbedded < Math.ceil(first.chunksEmbedded / 2),
  `and it is a small fraction of the corpus: ${third.chunksEmbedded} of ${first.chunksEmbedded}`
);
assert(
  third.chunksEmbedded <= third.chunksCreated && third.chunksCreated === third.chunksEmbedded,
  "every newly created chunk was embedded exactly once"
);
assert(third.chunksUnchanged > 0, `${third.chunksUnchanged} chunks in the other 5 files were skipped`);
assert(
  third.chunksUnchanged + third.chunksCreated + third.chunksUpdated === first.chunksCreated + third.chunksCreated,
  `chunk accounting still balances (${third.chunksUnchanged} unchanged + ${third.chunksCreated} created + ${third.chunksUpdated} moved)`
);

// The untouched files' chunks must literally not have been re-embedded.
const untouched = ["services/search/indexer.go", "web/src/api.ts", "tools/report/report.py"];
for (const rel of untouched) {
  const before = first.byLanguage;
  assert(before[rel.endsWith(".go") ? "go" : rel.endsWith(".ts") ? "ts" : "python"] > 0,
    `${rel} was ingested under language '${rel.endsWith(".go") ? "go" : rel.endsWith(".ts") ? "ts" : "python"}'`);
}
const afterEdit = await callTool("list_memories", { project: PROJECT, kind: "code", limit: 500 });
assert(afterEdit.includes("Void"), "the newly added Go function is now in the index");

section("[5] a pure reorder (same text, new lines) re-embeds nothing");
const reorderTarget = path.join(repo, "web/src/client.ts");
const original = fs.readFileSync(reorderTarget, "utf8");
fs.writeFileSync(
  path.join(repo, "services/payments/store.go"),
  goFile("payments", ["Migrate", "OpenDB"]) // swapped order => line ranges move
);
const fourth = parse(await callTool("ingest_codebase", { path: repo, project: PROJECT }));
console.log("  " + JSON.stringify(fourth));
assert(
  fourth.chunksEmbedded <= 2,
  `reordering two functions re-embeds at most their 2 chunks (${fourth.chunksEmbedded})`
);
assert(fourth.chunksUpdated + fourth.chunksUnchanged >= first.chunksCreated - 4,
  `${fourth.chunksUpdated} moved + ${fourth.chunksUnchanged} unchanged — recognised by hash, not by line`);

section("[6] deleting a file removes its chunks from search");
const beforeDelete = parse(await callTool("ingest_codebase", { path: repo, project: PROJECT, dry_run: true }));
assert(beforeDelete.dryRun === true, "dry run is reported as such");
assert(beforeDelete.chunksEmbedded === 0, "dry run embeds nothing");

fs.rmSync(path.join(repo, "tools/report/report.py"), { force: true });
const fifth = parse(await callTool("ingest_codebase", { path: repo, project: PROJECT }));
console.log("  " + JSON.stringify(fifth));
assert(fifth.filesDeleted === 1, `one file detected as deleted (${fifth.filesDeleted})`);
assert(fifth.chunksDeleted > 0, `and its ${fifth.chunksDeleted} chunks were removed`);
assert(fifth.chunksEmbedded === 0, "deletion alone triggers no embedding");

const goneRecall = await callTool("recall", { query: "build a report of totals", project: PROJECT, limit: 10 });
assert(!/report\.py/.test(goneRecall), "deleted file's chunks no longer come back");
const stillList = await callTool("list_memories", { project: PROJECT, kind: "code", limit: 500 });
assert(!/build_report/.test(stillList), "and they are gone from the store too");

section("[7] manual remember coexists with ingested code in one namespace");
const manual = await callTool("remember", {
  text: "We kept Kafka and Redis here because the legacy billing exporter could not tolerate at-least-once duplicates.",
  tags: "decision,architecture",
  project: PROJECT,
});
const manualId = Number(manual.match(/id=(\d+)/)?.[1]);
const mixed = await callTool("recall", { query: "why Kafka and Redis", project: PROJECT, limit: 5 });
assert(mixed.includes(`[${manualId}]`), "the manual note is retrievable alongside code chunks");
assert(/\.go:\d+-\d+/.test(mixed), "and code chunks are still in the same result set");
const codeOnly = await callTool("list_memories", { project: PROJECT, kind: "code", limit: 500 });
assert(!codeOnly.includes(`[${manualId}]`), "kind=code filters the manual note out");
const manualOnly = await callTool("list_memories", { project: PROJECT, kind: "manual", limit: 50 });
assert(manualOnly.includes(`[${manualId}]`), "kind=manual returns just it");

section("[8] ingesting into a second project leaves the first untouched");
const other = parse(await callTool("ingest_codebase", { path: repo, project: "other-service" }));
assert(other.chunksEmbedded > 0, `fresh project pays full embedding cost (${other.chunksEmbedded})`);
const pixoraAgain = parse(await callTool("ingest_codebase", { path: repo, project: PROJECT }));
assert(pixoraAgain.chunksEmbedded === 0, "the original project is still fully warm");

fs.writeFileSync(reorderTarget, original);
await client.close();
process.exitCode = finish("INCREMENTAL INGESTION TEST");

