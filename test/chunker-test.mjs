#!/usr/bin/env node
/**
 * Chunker test: boundaries, coverage, and the fallback path.
 *
 * Unit-level assertions against dist/chunker.js — no model, no database — so
 * they run in milliseconds and pin down exactly what "respects function/class
 * boundaries" means.
 */
import { chunkFile, detectLanguage } from "../dist/chunker.js";
import { assert, section, finish } from "./helpers.mjs";

/** Every non-blank source line must appear in at least one chunk. */
function coverageGaps(source, chunks) {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const covered = new Set();
  for (const c of chunks) for (let l = c.startLine; l <= c.endLine; l++) covered.add(l);
  const gaps = [];
  lines.forEach((line, i) => {
    if (line.trim().length > 0 && !covered.has(i + 1)) gaps.push(i + 1);
  });
  return gaps;
}

function find(chunks, needle) {
  return chunks.find((c) => c.content.includes(needle));
}

section("[1] language detection");
assert(detectLanguage("a/b/service.go") === "go", ".go -> go");
assert(detectLanguage("src/index.ts") === "ts", ".ts -> ts");
assert(detectLanguage("src/App.tsx") === "ts", ".tsx -> ts");
assert(detectLanguage("lib/util.js") === "js", ".js -> js");
assert(detectLanguage("svc/handler.py") === "python", ".py -> python");
assert(detectLanguage("README.md") === "text", ".md -> text (fallback)");
assert(detectLanguage("main.rs") === "text", ".rs -> text (fallback)");

section("[2] Go: one chunk per function/type, doc comments attached");
const go = `package main

import (
	"fmt"
	"net/http"
)

// HandleUpload accepts a multipart file and streams it to object storage.
func HandleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "nope", 405)
		return
	}
	fmt.Fprintln(w, "ok")
}

type UploadService struct {
	bucket string
	limit  int
}

func (u *UploadService) Save(name string) error {
	return nil
}

var ErrTooLarge = errors.New("too large")
`;
const goChunks = chunkFile("internal/upload/handler.go", go);
const handleUpload = find(goChunks, "func HandleUpload");
assert(!!handleUpload, "found a chunk containing func HandleUpload");
assert(handleUpload.unit === "HandleUpload", `unit name extracted (${handleUpload.unit})`);
assert(handleUpload.unitKind === "func", `unit kind is func (${handleUpload.unitKind})`);
assert(handleUpload.content.includes("accepts a multipart file"),
  "the doc comment travels with the function it documents");
assert(handleUpload.content.trimEnd().endsWith("}"), "chunk ends at the function's closing brace");
assert(!handleUpload.content.includes("func (u *UploadService)"),
  "the next function is NOT swallowed into this chunk");

const save = find(goChunks, "func (u *UploadService) Save");
assert(!!save && save.unit === "Save", `method chunk found with receiver (${save?.unit})`);
assert(save.unitKind === "method", "receiver functions are classified as methods");

const svc = find(goChunks, "type UploadService struct");
assert(!!svc && svc.unit === "UploadService" && svc.unitKind === "struct", "struct is its own chunk");
assert(svc.content.includes("bucket string") && !svc.content.includes("func HandleUpload"),
  "struct chunk covers the body and nothing else");

assert(!!find(goChunks, "import ("), "the import block is still captured (no content dropped)");
assert(coverageGaps(go, goChunks).length === 0, `every non-blank Go line is covered: ${coverageGaps(go, goChunks)}`);


section("[3] braces inside strings and comments do not break boundaries");
const tricky = `package main

func Weird() string {
	open := "a { brace in a string"
	closed := "and a } one"
	note := \`raw { backtick string
 spanning lines } still fine\`
	// a comment with { and } in it
	return open + closed + note
}

func After() int {
	return 2
}
`;
const trickyChunks = chunkFile("tricky/tricky.go", tricky);
const weird = find(trickyChunks, "func Weird()");
assert(!!weird, "found func Weird");
// If braces inside strings/comments were miscounted, the chunk would terminate
// at the fake `}` inside the string and the real closing brace would be lost.
assert(weird.content.includes("return open + closed + note"),
  "the closing brace inside a string did not terminate the chunk early");
assert(find(trickyChunks, "func After()")?.unit === "After", "the following function is still identified");
assert(!weird.content.includes("func After()"), "and Weird() does not swallow it");
assert(coverageGaps(tricky, trickyChunks).length === 0, "all lines covered despite tricky literals");

section("[4] TypeScript: functions, classes, arrow consts, interfaces");
const ts = `import { z } from "zod";

export interface RecallRequest {
  query: string;
  limit?: number;
}

export type Outcome = "ok" | "error";

const DEFAULT_LIMIT = 5;

export async function handleRecall(req: RecallRequest): Promise<string[]> {
  const items = await search(req.query);
  return items.slice(0, req.limit ?? DEFAULT_LIMIT);
}

export class MemoryStore {
  private rows: string[] = [];

  add(row: string): void {
    this.rows.push(row);
  }

  size(): number {
    return this.rows.length;
  }
}

const helper = (x: number) => x * 2;

export default handleRecall;
`;
const tsChunks = chunkFile("src/recall.ts", ts);
const iface = find(tsChunks, "export interface RecallRequest");
assert(!!iface && iface.unit === "RecallRequest" && iface.unitKind === "interface", "interface chunk");
assert(!iface.content.includes("export type Outcome"), "interface chunk stops at its brace");

const handle = find(tsChunks, "export async function handleRecall");
assert(!!handle && handle.unit === "handleRecall", "async exported function chunk");
assert(handle.content.includes("return items.slice"), "body fully included");
assert(!handle.content.includes("class MemoryStore"), "stops before the class");

const store = find(tsChunks, "export class MemoryStore");
assert(!!store && store.unit === "MemoryStore" && store.unitKind === "class", "class chunk");
assert(store.content.includes("size(): number"),
  "methods stay inside their class rather than becoming separate chunks");

section("[5] Python: indentation-scoped defs, classes and decorators");
const py = `"""Payments service."""
import os
from dataclasses import dataclass


@dataclass
class Charge:
    """A single card charge."""

    amount: int
    currency: str

    def total(self, tax_rate: float) -> float:
        subtotal = self.amount
        return subtotal + subtotal * tax_rate

    def refund(self) -> None:
        self.amount = 0


def build_gateway(api_key: str, region: str = "eu"):
    # reads config from the environment
    return Gateway(api_key=api_key, region=region)


async def fetch_ledger(account_id: str):
    async with session.get(url) as resp:
        data = await resp.json()
    return data


CONSTANT = 42
`;
const pyChunks = chunkFile("payments/gateway.py", py);
const charge = find(pyChunks, "class Charge");
assert(!!charge && charge.unit === "Charge" && charge.unitKind === "class", "class chunk found");
assert(charge.content.includes("def total"), "methods stay inside the class");
assert(charge.content.includes("def refund"), "and so does the second method");
assert(charge.content.includes("@dataclass"), "the decorator is attached to the class it decorates");
assert(charge.content.includes("A single card charge"), "the class docstring is included");
assert(!charge.content.includes("def build_gateway"), "the class chunk stops at the next top-level def");

const buildGw = find(pyChunks, "def build_gateway");
assert(!!buildGw && buildGw.unit === "build_gateway" && buildGw.unitKind === "function", "top-level def");
assert(!buildGw.content.includes("async def fetch_ledger"), "def chunk stops at the next def");

const fetchLedger = find(pyChunks, "async def fetch_ledger");
assert(!!fetchLedger && fetchLedger.unit === "fetch_ledger", "async def recognised");
assert(!!find(pyChunks, "import os"), "module preamble preserved");
assert(coverageGaps(py, pyChunks).length === 0, "every Python line covered");

section("[6] fallback: unknown file types use an overlapping line window");
const md = Array.from({ length: 200 }, (_, i) => `Line ${i + 1} of the runbook describes step ${i + 1}.`).join("\n");
const mdChunks = chunkFile("docs/runbook.md", md, { fallbackLines: 50, fallbackOverlapLines: 10 });
assert(mdChunks.length > 3, `produced ${mdChunks.length} windowed chunks`);
assert(mdChunks.every((c) => c.unitKind === "lines"), "fallback chunks are labelled 'lines'");
const firstTwo = mdChunks[0].endLine, secondStart = mdChunks[1].startLine;
assert(secondStart <= firstTwo, `consecutive windows overlap (${secondStart} <= ${firstTwo})`);
assert(coverageGaps(md, mdChunks).length === 0, "fallback still covers every line");

section("[7] oversized units are split with overlap, never truncated");
const bigFn =
  "func Huge() {\n" +
  Array.from({ length: 400 }, (_, i) => `  x := compute(${i}) // statement ${i}`).join("\n") +
  "\n}\n";
const bigChunks = chunkFile("big/big.go", bigFn, { maxChunkLines: 100, splitOverlapLines: 20 });
assert(bigChunks.length >= 4, `400-line function split into ${bigChunks.length} pieces`);
assert(bigChunks.every((c) => c.endLine - c.startLine + 1 <= 100), "no piece exceeds maxChunkLines");
assert(coverageGaps(bigFn, bigChunks).length === 0, "splitting loses no lines");
assert(bigChunks.every((c) => c.unit === "Huge"), "every piece keeps the enclosing function name");
const overlapPresent = bigChunks.some((c, i) => i > 0 && c.startLine <= bigChunks[i - 1].endLine);
assert(overlapPresent, "pieces overlap so context is not cut at an arbitrary line");

section("[8] degenerate inputs");
assert(chunkFile("empty.go", "").length === 0, "empty file yields no chunks");
assert(chunkFile("blank.py", "\n\n   \n\t\n").length === 0, "whitespace-only file yields no chunks");
const oneLiner = "export const VERSION = \"0.2.0\";\n";
assert(chunkFile("v.ts", oneLiner).length >= 1, "a single-line file still produces a chunk");

process.exitCode = finish("CHUNKER TEST");


const arrow = find(tsChunks, "const helper = (x: number)");
assert(!!arrow && arrow.unit === "helper", "arrow-function const recognised as a function unit");
assert(coverageGaps(ts, tsChunks).length === 0, "every TS line covered");
