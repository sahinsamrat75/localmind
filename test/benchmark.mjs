#!/usr/bin/env node
/**
 * localmind scale benchmark.
 *
 * Generates a synthetic multi-service monorepo (Go / TypeScript / Python),
 * ingests it into two separate project namespaces, and measures the things that
 * decide whether this design survives a real PIXORA-scale codebase:
 *
 *   - cold ingestion time and embedding throughput
 *   - recall latency distribution at full corpus size (the ANN index must stay
 *     sub-second; we also time the brute-force cosine scan it replaced, so the
 *     speedup is measured rather than asserted)
 *   - incremental re-ingestion after editing one file (should re-embed roughly
 *     one file's worth of chunks, not the corpus)
 *   - index persistence, cold reload, and rebuild from SQLite
 *   - resident memory and on-disk bytes
 *
 * Usage:
 *   node test/benchmark.mjs [--chunks=20000] [--queries=60] [--keep]
 *                           [--out=test/benchmark-results.json]
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { MemoryStore } from "../dist/store.js";
import { VectorIndex } from "../dist/vectorIndex.js";
import { ingestCodebase } from "../dist/ingest.js";
import { runRecall } from "../dist/recall.js";
import { embed, cosineSimilarity } from "../dist/embeddings.js";

function arg(name, fallback) {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : fallback;
}

const TARGET_CHUNKS = Number(arg("chunks", 20000));
const QUERY_COUNT = Number(arg("queries", 60));
const KEEP = process.argv.includes("--keep");
const OUT = arg("out", path.join(process.cwd(), "test", "benchmark-results.json"));
const REPO_ROOT = arg("repo", fs.mkdtempSync(path.join(os.tmpdir(), "localmind-bench-repo-")));
const HOME = arg("home", fs.mkdtempSync(path.join(os.tmpdir(), "localmind-bench-home-")));

const PROJECTS = [
  { name: "pixora-core", services: 14 },
  { name: "pixora-edge", services: 9 },
];

const VERBS = ["Fetch", "Build", "Validate", "Stream", "Aggregate", "Resolve", "Flush", "Retry", "Map", "Emit"];
const NOUNS = ["Ledger", "Session", "Invoice", "Shipment", "Playlist", "Vector", "Catalog", "Tenant", "Webhook", "Segment"];
const TOPICS = [
  "payment authorization", "session refresh", "invoice reconciliation", "shipment tracking",
  "playlist ranking", "vector indexing", "catalog search", "tenant isolation",
  "webhook delivery", "segment analytics", "rate limiting", "caching policy",
  "retry backoff", "schema migration", "audit logging", "feature flags",
];

function pick(list, seed) {
  return list[Math.abs(seed) % list.length];
}

/** Deterministic pseudo-random, so a rerun measures the same corpus. */
function lcg(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 0x100000000);
}

const FUNCS_PER_FILE = 8;

function goFile(service, idx, rnd) {
  const topic = pick(TOPICS, Math.floor(rnd() * TOPICS.length));
  const name = pick(NOUNS, Math.floor(rnd() * NOUNS.length));
  const verb = pick(VERBS, Math.floor(rnd() * VERBS.length));
  let out = `package ${service}\n\nimport (\n\t"context"\n\t"fmt"\n)\n\n`;
  for (let f = 0; f < FUNCS_PER_FILE; f++) {
    const fn = `${verb}${name}${idx}${f}`;
    out += `// ${fn} handles ${topic} for the ${service} service.\n`;
    out += `func ${fn}(ctx context.Context, req *Request) (*Response, error) {\n`;
    for (let l = 0; l < 6; l++) out += `\tresp := transform${f}(ctx, req, "step ${l} of ${topic}")\n`;
    out += `\treturn resp, nil\n}\n\n`;
  }
  return { path: `services/${service}/handler_${idx}.go`, content: out, funcs: FUNCS_PER_FILE, topic };
}

function tsFile(service, idx, rnd) {
  const topic = pick(TOPICS, Math.floor(rnd() * TOPICS.length));
  const name = pick(NOUNS, Math.floor(rnd() * NOUNS.length));
  let out = `// ${service} client helpers for ${topic}.\n`;
  for (let f = 0; f < FUNCS_PER_FILE; f++) {
    const fn = `get${name}${idx}${f}`;
    out += `export async function ${fn}(input: RequestInput): Promise<Response> {\n`;
    out += `  // ${topic} path\n`;
    for (let l = 0; l < 5; l++) out += `  const v${l} = await decode(input, "${service}", ${l});\n`;
    out += `  return serialize(v4);\n}\n\n`;
  }
  return { path: `web/${service}/api_${idx}.ts`, content: out, funcs: FUNCS_PER_FILE, topic };
}

function pyFile(service, idx, rnd) {
  const topic = pick(TOPICS, Math.floor(rnd() * TOPICS.length));
  const name = pick(NOUNS, Math.floor(rnd() * NOUNS.length));
  let out = `"""${service} ${topic} utilities."""\nimport asyncio\n\n`;
  for (let f = 0; f < FUNCS_PER_FILE; f++) {
    const fn = `${name.toLowerCase()}_${topic.split(" ")[0]}_${idx}_${f}`;
    out += `def ${fn}(ctx, req):\n    """Process ${topic}."""\n`;
    for (let l = 0; l < 5; l++) out += `    value${l} = transform(ctx, req, ${l})\n`;
    out += `    return value4\n\n\n`;
  }
  return { path: `tools/${service}/${name.toLowerCase()}_${idx}.py`, content: out, funcs: FUNCS_PER_FILE, topic };
}


/** Write a synthetic service tree; returns the files written. */
function generateProject(repoDir, spec, filesPerService, rnd) {
  const written = [];
  for (let s = 0; s < spec.services; s++) {
    const service = `${pick(NOUNS, s).toLowerCase()}${s}`;
    for (let i = 0; i < filesPerService; i++) {
      const kind = (s + i) % 3;
      const file = kind === 0 ? goFile(service, i, rnd) : kind === 1 ? tsFile(service, i, rnd) : pyFile(service, i, rnd);
      const abs = path.join(repoDir, spec.name, file.path);
      fs.mkdirSync(path.dirname(abs), { recursive: true });
      fs.writeFileSync(abs, file.content);
      written.push({ abs, rel: `${spec.name}/${file.path}`, funcs: file.funcs, topic: file.topic });
    }
  }
  return written;
}

function ms(n) {
  return n < 1000 ? `${n.toFixed(0)}ms` : `${(n / 1000).toFixed(2)}s`;
}

function bytes(n) {
  if (n > 1024 * 1024 * 1024) return `${(n / 1024 / 1024 / 1024).toFixed(2)} GiB`;
  if (n > 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MiB`;
  return `${(n / 1024).toFixed(1)} KiB`;
}

function dirBytes(dir) {
  let total = 0;
  const stack = [dir];
  while (stack.length) {
    const d = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(d, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) stack.push(p);
      else total += fs.statSync(p).size;
    }
  }
  return total;
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[Math.max(0, idx)];
}

function rss() {
  return process.memoryUsage().rss;
}

function row(label, value) {
  console.log(`  ${String(label).padEnd(46)} ${value}`);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

const results = { meta: {}, generation: {}, ingestion: [], recall: {}, incremental: {}, footprint: {} };

results.meta = {
  ranAt: new Date().toISOString(),
  node: process.version,
  platform: `${process.platform}-${process.arch}`,
  cpus: os.cpus().length,
  totalMemGiB: +(os.totalmem() / 1024 / 1024 / 1024).toFixed(1),
  targetChunks: TARGET_CHUNKS,
  queries: QUERY_COUNT,
};

console.log("=".repeat(70));
console.log("localmind benchmark");
console.log("=".repeat(70));
row("node", process.version);
row("platform", `${process.platform}-${process.arch} (${os.cpus().length} cpus, ${(os.totalmem() / 1024 ** 3).toFixed(0)} GiB)`);
row("target chunks", TARGET_CHUNKS);

// Size the corpus: FUNCS_PER_FILE chunks per file, split across the projects
// in proportion to their service counts. Files are distributed *across*
// services, so the per-service count is the project's share divided by its
// services, rounded up.
const totalServices = PROJECTS.reduce((n, p) => n + p.services, 0);
const estFiles = Math.ceil(TARGET_CHUNKS / FUNCS_PER_FILE);
const genStart = Date.now();
const allFiles = [];
const rnd = lcg(20260904);
for (const spec of PROJECTS) {
  const share = Math.max(1, Math.round((estFiles * spec.services) / totalServices));
  const perService = Math.max(1, Math.ceil(share / spec.services));
  allFiles.push({ spec, files: generateProject(REPO_ROOT, spec, perService, rnd) });
}
const genMs = Date.now() - genStart;
const generatedChunks = allFiles.reduce((n, g) => n + g.files.length * FUNCS_PER_FILE, 0);

console.log("\n[1] corpus generation");
row("repo", REPO_ROOT);
row("projects", PROJECTS.map((p) => p.name).join(", "));
row("files written", allFiles.reduce((n, g) => n + g.files.length, 0));
row("expected chunks", generatedChunks);
row("source bytes on disk", bytes(dirBytes(REPO_ROOT)));
row("generation time", ms(genMs));
results.generation = {
  files: allFiles.reduce((n, g) => n + g.files.length, 0),
  expectedChunks: generatedChunks,
  sourceBytes: dirBytes(REPO_ROOT),
  genMs,
};

const dbPath = path.join(HOME, "memory.db");
const indexDir = path.join(HOME, "indexes");
const store = new MemoryStore(dbPath);
let index = new VectorIndex({
  indexDir,
  liveIds: (p) => store.idsForProject(p),
  vectors: (p) => store.embeddingsForProject(p),
  log: (m) => console.log(`  [index] ${m}`),
});

console.log("\n[2] cold ingestion (embed + index, per project)");
for (const group of allFiles) {
  const root = path.join(REPO_ROOT, group.spec.name);
  const t = Date.now();
  const summary = await ingestCodebase({ project: group.spec.name, root, waveSize: 4000 }, store, index);
  const wall = Date.now() - t;
  const rate = summary.chunksEmbedded / (Math.max(1, summary.embedMs) / 1000);
  row(
    `${group.spec.name}: ${summary.chunksEmbedded} chunks`,
    `${ms(wall)} total | ${ms(summary.embedMs)} embedding | ${rate.toFixed(1)} chunks/s`
  );
  results.ingestion.push({
    project: group.spec.name,
    files: summary.filesScanned,
    chunks: summary.chunksEmbedded,
    wallMs: wall,
    embedMs: summary.embedMs,
    chunksPerSec: +rate.toFixed(1),
    byLanguage: summary.byLanguage,
  });
}
const totalChunks = store.countAll();
row("total memories in sqlite", totalChunks);

console.log("\n[3] recall latency at full corpus size");
// Queries are drawn from the topics actually present in the corpus, so they
// retrieve real matches rather than scoring noise.
const queryTopics = [...new Set(allFiles.flatMap((g) => g.files.map((f) => f.topic)))];
for (let i = queryTopics.length; i < QUERY_COUNT; i++) {
  queryTopics.push(`how does ${pick(TOPICS, i * 7 + 3)} work`);
}
const benchmarkQueries = queryTopics.slice(0, QUERY_COUNT);

// Warm the model and the indexes so we are measuring steady-state search.
await embed("warmup");
for (const spec of PROJECTS) index.size(spec.name);

const latencies = [];
const budgetLatencies = [];
let hitsSeen = 0;
let leaked = 0;
for (let q = 0; q < benchmarkQueries.length; q++) {
  const project = PROJECTS[q % PROJECTS.length].name;
  const t = Date.now();
  const res = await runRecall({ store, index }, { query: benchmarkQueries[q], project, limit: 10 });
  latencies.push(Date.now() - t);
  hitsSeen += res.hits.length;
  // Isolation check at scale: no result may belong to another project.
  for (const h of res.hits) if (h.memory.project !== project) leaked++;

  const tb = Date.now();
  await runRecall({ store, index }, { query: benchmarkQueries[q], project, maxTokens: 2000 });
  budgetLatencies.push(Date.now() - tb);
}

const sorted = latencies.slice().sort((a, b) => a - b);
const sortedB = budgetLatencies.slice().sort((a, b) => a - b);
row(`recall p50 (n=${latencies.length})`, `${percentile(sorted, 50)}ms`);
row("recall p95", `${percentile(sorted, 95)}ms`);
row("recall p99 / max", `${percentile(sorted, 99)}ms / ${sorted[sorted.length - 1]}ms`);
row("recall p50/p95 with max_tokens=2000", `${percentile(sortedB, 50)}ms / ${percentile(sortedB, 95)}ms`);
row("avg hits per query", (hitsSeen / latencies.length).toFixed(1));
row("cross-project leaks", leaked);

results.recall = {
  queries: latencies.length,
  p50Ms: percentile(sorted, 50),
  p95Ms: percentile(sorted, 95),
  p99Ms: percentile(sorted, 99),
  maxMs: sorted[sorted.length - 1],
  budgetP50Ms: percentile(sortedB, 50),
  budgetP95Ms: percentile(sortedB, 95),
  avgHits: +(hitsSeen / latencies.length).toFixed(2),
  crossProjectLeaks: leaked,
};

console.log("\n[4] ANN vs the brute-force cosine scan it replaced");
// Same query vectors, same corpus, measured both ways. The linear scan is what
// v1 did on every recall: load every embedding and dot-product it.
const sampleQueries = [];
for (let i = 0; i < 10; i++) sampleQueries.push(await embed(benchmarkQueries[i]));

// Time the actual vector work: pull every embedding for one project and scan
// it, which is precisely what v1's recall did on every call.
const scanStart = Date.now();
let scanSink = 0;
const vectors = [...store.embeddingsForProject(PROJECTS[0].name)];
const loadMs = Date.now() - scanStart;
const scanT = Date.now();
for (const q of sampleQueries) {
  for (const v of vectors) scanSink += cosineSimilarity(q, v.embedding);
}
const scanMs = (Date.now() - scanT) / sampleQueries.length;
void scanSink;

const annT = Date.now();
for (const q of sampleQueries) index.search(PROJECTS[0].name, q, 60);
const annMs = (Date.now() - annT) / sampleQueries.length;

row(`brute-force scan of ${vectors.length} vectors`, `${scanMs.toFixed(1)}ms per query (+${loadMs}ms blob decode)`);
row("HNSW search over the same corpus", `${annMs.toFixed(2)}ms per query`);
row("speedup", `${(scanMs / Math.max(0.01, annMs)).toFixed(0)}x`);
results.recall.bruteForceMsPerQuery = +scanMs.toFixed(2);
results.recall.annMsPerQuery = +annMs.toFixed(2);
results.recall.bruteForceVectors = vectors.length;

console.log("\n[5] incremental re-ingestion after editing one file");
const group0 = allFiles[0];
const target = group0.files[Math.floor(group0.files.length / 2)];
const original = fs.readFileSync(target.abs, "utf8");
fs.writeFileSync(target.abs, original + `\nfunc AddedByBenchmark(ctx context.Context) error {\n\treturn nil\n}\n`);

const incT = Date.now();
const inc = await ingestCodebase({ project: group0.spec.name, root: path.join(REPO_ROOT, group0.spec.name), waveSize: 4000 }, store, index);
const incMs = Date.now() - incT;
row("files in project", inc.filesScanned);
row("chunks re-embedded", `${inc.chunksEmbedded} (of ${inc.totalChunks} in this run)`);
row("chunks unchanged (hash hit)", inc.chunksUnchanged);
row("incremental wall time", ms(incMs));
row("share of corpus re-embedded", `${((inc.chunksEmbedded / inc.totalChunks) * 100).toFixed(1)}%`);
results.incremental = {
  project: group0.spec.name,
  embedded: inc.chunksEmbedded,
  unchanged: inc.chunksUnchanged,
  total: inc.totalChunks,
  wallMs: incMs,
  pctReembedded: +((inc.chunksEmbedded / inc.totalChunks) * 100).toFixed(2),
};

// No-op re-ingest: the steady state an agent sees between edits.
const noopT = Date.now();
const noop = await ingestCodebase({ project: group0.spec.name, root: path.join(REPO_ROOT, group0.spec.name) }, store, index);
row("no-op re-ingest (nothing changed)", `${ms(Date.now() - noopT)}, ${noop.chunksEmbedded} embedded`);
results.incremental.noopMs = Date.now() - noopT;
results.incremental.noopEmbedded = noop.chunksEmbedded;

console.log("\n[6] persistence, cold reload, and rebuild from SQLite");
const idxStats = PROJECTS.map((p) => ({ project: p.name, ...index.stats(p.name) }));
for (const s of idxStats) row(`${s.project} index`, `${bytes(s.bytes)} on disk, ${s.live} live / ${s.capacity} slots`);

index.close();
store.close();

const reloadT = Date.now();
const store2 = new MemoryStore(dbPath);
const index2 = new VectorIndex({
  indexDir,
  liveIds: (p) => store2.idsForProject(p),
  vectors: (p) => store2.embeddingsForProject(p),
  log: () => {},
});
let reloadHits = 0;
for (const spec of PROJECTS) reloadHits += index2.size(spec.name);
const reloadMs = Date.now() - reloadT;
row(`cold reload of ${reloadHits} vectors from disk`, ms(reloadMs));
const reloadQuery = Date.now();
const reloaded = await runRecall({ store: store2, index: index2 }, { query: benchmarkQueries[0], project: PROJECTS[0].name, limit: 5 });
row("first recall after reload", `${Date.now() - reloadQuery}ms (${reloaded.hits.length} hits)`);
results.footprint.reloadMs = reloadMs;
results.footprint.reloadedVectors = reloadHits;

// Force the self-healing path: delete the index files and let it rebuild from
// the embedding BLOBs SQLite keeps.
index2.close();
store2.close();
for (const f of fs.readdirSync(indexDir)) fs.rmSync(path.join(indexDir, f), { force: true });
const rebuildT = Date.now();
const store3 = new MemoryStore(dbPath);
const index3 = new VectorIndex({
  indexDir,
  liveIds: (p) => store3.idsForProject(p),
  vectors: (p) => store3.embeddingsForProject(p),
  log: () => {},
});
let rebuilt = 0;
for (const spec of PROJECTS) rebuilt += index3.size(spec.name);
const rebuildMs = Date.now() - rebuildT;
row(`rebuild ${rebuilt} vectors from sqlite`, ms(rebuildMs));
const rebuildQuery = Date.now();
const rb = await runRecall({ store: store3, index: index3 }, { query: benchmarkQueries[0], project: PROJECTS[0].name, limit: 5 });
row("recall immediately after rebuild", `${Date.now() - rebuildQuery}ms (${rb.hits.length} hits)`);
results.footprint.rebuildMs = rebuildMs;
results.footprint.rebuiltVectors = rebuilt;

console.log("\n[7] footprint");
const dbBytes = fs.statSync(dbPath).size;
const walBytes = fs.existsSync(`${dbPath}-wal`) ? fs.statSync(`${dbPath}-wal`).size : 0;
const indexBytes = dirBytes(indexDir);
row("sqlite database", bytes(dbBytes + walBytes));
row("hnsw index files", bytes(indexBytes));
row("total localmind data", bytes(dbBytes + walBytes + indexBytes));
row("bytes per stored chunk", (((dbBytes + walBytes + indexBytes) / totalChunks) / 1024).toFixed(1) + " KiB");
row("rss with all indexes loaded", bytes(rss()));
row("peak rss", bytes(process.memoryUsage().heapTotal + (rss() - process.memoryUsage().heapTotal)));
results.footprint.dbBytes = dbBytes + walBytes;
results.footprint.indexBytes = indexBytes;
results.footprint.totalChunks = totalChunks;
results.footprint.rssBytes = rss();
results.footprint.perProject = idxStats.map((s) => ({ project: s.project, bytes: s.bytes, live: s.live }));

index3.close();
store3.close();

console.log("\n" + "=".repeat(70));
const pass =
  totalChunks >= TARGET_CHUNKS &&
  results.recall.p95Ms < 1000 &&
  results.recall.crossProjectLeaks === 0 &&
  results.incremental.pctReembedded < 5 &&
  results.incremental.noopEmbedded === 0;
row("corpus >= target", `${totalChunks} >= ${TARGET_CHUNKS} -> ${totalChunks >= TARGET_CHUNKS ? "PASS" : "FAIL"}`);
row("recall p95 sub-second", `${results.recall.p95Ms}ms -> ${results.recall.p95Ms < 1000 ? "PASS" : "FAIL"}`);
row("project isolation held", `${results.recall.crossProjectLeaks} leaks -> ${results.recall.crossProjectLeaks === 0 ? "PASS" : "FAIL"}`);
row("incremental < 5% re-embedded", `${results.incremental.pctReembedded}% -> ${results.incremental.pctReembedded < 5 ? "PASS" : "FAIL"}`);
row("steady state embeds nothing", `${results.incremental.noopEmbedded} -> ${results.incremental.noopEmbedded === 0 ? "PASS" : "FAIL"}`);
console.log("=".repeat(70));
console.log(pass ? "BENCHMARK: ALL SCALE GATES PASSED" : "BENCHMARK: GATE FAILURE");

fs.writeFileSync(OUT, JSON.stringify(results, null, 2));
console.log(`\nresults written to ${OUT}`);

if (!KEEP) {
  fs.rmSync(REPO_ROOT, { recursive: true, force: true });
  fs.rmSync(HOME, { recursive: true, force: true });
} else {
  console.log(`kept: ${REPO_ROOT}\nkept: ${HOME}`);
}
process.exit(pass ? 0 : 1);


row("rss after ingestion", bytes(rss()));
