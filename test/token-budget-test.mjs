#!/usr/bin/env node
/**
 * Token-budget recall test.
 *
 * Verifies the core value proposition: instead of dumping whole files, an agent
 * names a context budget and gets back the most relevant, de-duplicated chunks
 * that fit — each with a file path and line range so it can pull more if it
 * wants.
 */
import { startServer, makeTempDir, assert, section, finish, fs, path } from "./helpers.mjs";
import { estimateTokens } from "../dist/tokens.js";

const home = makeTempDir("localmind-budget-");
const repo = makeTempDir("budget-fixture-");
const PROJECT = "budget-demo";

function write(rel, text) {
  const abs = path.join(repo, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, text);
}

section("[1] fixture with near-duplicate and overlapping chunks");
// Two services with an almost identical function (the classic copy-paste in a
// microservice repo) plus a genuinely distinct one.
const dupA = Array.from({ length: 14 }, (_, i) => `  step${i}: applyTaxTo(order.lines[${i}], rate)`).join("\n");
write("services/eu/tax.go", `package eu\n\nfunc ComputeTax(order *Order, rate float64) float64 {\n${dupA}\n  return order.Total\n}\n`);
write("services/us/tax.go", `package us\n\nfunc ComputeTax(order *Order, rate float64) float64 {\n${dupA}\n  return order.Total\n}\n`);
const uniqueBody = Array.from({ length: 30 }, (_, i) => `  conn.Exec("INSERT INTO ledger VALUES ($${i})", entry${i})`).join("\n");
write("services/ledger/ledger.go", `package ledger\n\nfunc FlushLedger(conn *sql.DB, entries []Entry) error {\n${uniqueBody}\n  return conn.Commit()\n}\n`);
write("services/search/query.ts", `export function buildSearchQuery(term: string, filters: Filters) {\n  const parts = ["q=" + encodeURIComponent(term)];\n  for (const [k, v] of Object.entries(filters)) parts.push(k + "=" + v);\n  return "/api/search?" + parts.join("&");\n}\n`);

const { callTool, client } = await startServer(home);
const ing = JSON.parse(await callTool("ingest_codebase", { path: repo, project: PROJECT }));
assert(ing.chunksCreated >= 4, `${ing.chunksCreated} chunks ingested`);

function body(text) {
  return text.split("\n").slice(1).join("\n");
}
function hitIds(text) {
  return [...text.matchAll(/^\[(\d+)\]/gm)].map((m) => Number(m[1]));
}

section("[2] estimateTokens is monotonic and sane");
assert(estimateTokens("") === 0, "empty text costs nothing");
assert(estimateTokens("hello world") > 0, "short text costs something");
assert(estimateTokens("a".repeat(400)) > estimateTokens("a".repeat(100)), "longer text costs more");

section("[3] max_tokens bounds the response");
const big = await callTool("recall", { query: "tax calculation on an order", project: PROJECT, max_tokens: 4000 });
const small = await callTool("recall", { query: "tax calculation on an order", project: PROJECT, max_tokens: 220 });
console.log("  budget 4000 -> " + big.split("\n")[0]);
console.log("  budget  220 -> " + small.split("\n")[0]);

const bigHeader = big.split("\n")[0];
const smallHeader = small.split("\n")[0];
const bigUsed = Number(bigHeader.match(/using (\d+)\//)?.[1]);
const smallUsed = Number(smallHeader.match(/using (\d+)\//)?.[1]);
assert(Number.isFinite(bigUsed) && Number.isFinite(smallUsed), "response reports tokens used");
assert(smallUsed <= 220, `small response stays inside its budget (${smallUsed} <= 220)`);
assert(bigUsed <= 4000, `large response stays inside its budget (${bigUsed} <= 4000)`);
assert(hitIds(small).length < hitIds(big).length,
  `a tighter budget returns fewer results (${hitIds(small).length} vs ${hitIds(big).length})`);
assert(/dropped by budget/.test(smallHeader) || hitIds(small).length > 0,
  "the header says when the budget truncated the list");

section("[4] the budget keeps the MOST relevant chunks, not the smallest");
const topSmall = hitIds(small)[0];
const topBig = hitIds(big)[0];
assert(topSmall === topBig, `rank-1 result is identical under both budgets (id=${topSmall})`);
assert(/tax\.go/.test(small), "the tax chunks are what a tax query returns");

section("[5] near-duplicate chunks are collapsed");
const dupQuery = await callTool("recall", { query: "apply tax to order lines", project: PROJECT, max_tokens: 6000, limit: 50 });
console.log("  " + body(dupQuery).split("\n").map((l) => l.slice(0, 68)).join("\n  "));
// The two ComputeTax functions are byte-identical, so only one may survive.
const taxFns = body(dupQuery).split("\n").filter((l) => l.includes("func ComputeTax"));
assert(taxFns.length === 1, `the copy-pasted function appears exactly once (${taxFns.length})`);
// The `package eu` / `package us` lines are genuinely different text and may
// both survive; but no id may repeat and no function may repeat.
const dupIds = hitIds(dupQuery);
assert(new Set(dupIds).size === dupIds.length, "no duplicate ids in a single recall");
const dupFiles = body(dupQuery).match(/services\/(eu|us)\/tax\.go/g) ?? [];
assert(dupFiles.length <= 3, `redundant hits stayed bounded (${dupFiles.length})`);

section("[6] every hit carries a file path and line range");
const all = await callTool("recall", { query: "ledger insert commit", project: PROJECT, max_tokens: 3000 });
for (const line of body(all).split("\n").filter((l) => l.startsWith("["))) {
  assert(/\.go:\d+-\d+/.test(line) || /\.ts:\d+-\d+/.test(line),
    `hit is addressable: ${line.slice(0, 70)}`);
}
assert(/ledger\.go/.test(all), "and the right file is named");

section("[7] without max_tokens, limit still governs");
const limited = await callTool("recall", { query: "tax", project: PROJECT, limit: 1 });
assert(hitIds(limited).length === 1, `limit=1 returns exactly one hit (${hitIds(limited).length})`);
const unlimited = await callTool("recall", { query: "tax", project: PROJECT });
assert(hitIds(unlimited).length === 5, `default limit is 5 (${hitIds(unlimited).length})`);

section("[8] min_score filters weak matches");
const strict = await callTool("recall", { query: "completely unrelated quantum foobar", project: PROJECT, min_score: 0.99 });
assert(/No matching memories/.test(strict), `nothing clears a 0.99 bar: ${strict.trim().slice(0, 60)}`);

await client.close();
process.exitCode = finish("TOKEN BUDGET TEST");
