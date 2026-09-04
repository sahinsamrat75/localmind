# localmind

**Local-first semantic memory for AI agents — now at codebase scale.**

localmind is a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that gives your AI agent persistent memory that never leaves your machine. Memories live in a single SQLite file and are searched with embeddings produced **entirely on-device** — no API keys, no accounts, no cloud, no telemetry.

v2 adds what you need for **whole-codebase memory**: an approximate-nearest-neighbour index instead of a linear scan, named project namespaces, an `ingest_codebase` tool that splits source into functions and classes, incremental re-ingestion that skips unchanged code, and token-budgeted recall so an agent pulls in only what fits.

## What it does

Your agent gets 6 MCP tools:

| Tool | What it does |
| --- | --- |
| `remember` | Store a note, fact or decision (optionally tagged) in a project. |
| `recall` | Semantic search over one project via HNSW. Optional `max_tokens` bounds the result. |
| `forget` | Delete one memory by id, inside one project. |
| `list_memories` | Browse a project, filterable by tag and/or kind (`manual` / `code`). |
| `ingest_codebase` | Index a source tree: walk → chunk at function/class boundaries → embed. Incremental. |
| `list_projects` | List namespaces with counts and index sizes. |

## Project namespacing

Every memory belongs to exactly one `project` (e.g. `pixora`, `localmind`). Isolation is **structural, not a filter**: each project gets its own vector index file, so a recall in project A physically cannot reach project B's vectors.

```jsonc
{ "project": "pixora", "query": "why did we pick Kafka here?" }   // only pixora
{ "query": "what must I avoid cooking?" }                          // implicit project "default"
```

Omitting `project` means the `default` namespace, which is where v1 memories land after migration — so existing behaviour is preserved exactly.

## Ingesting a codebase

`ingest_codebase(path=<absolute path>, project=<name>, …)`

- **Walks** the directory recursively, honouring `.gitignore` (comments, negations, nested files, `**` globs) and pruning excluded directories instead of descending into them. `node_modules` and `.git` are always skipped.
- **Chunks by logical unit**: Go (`func`, receiver methods, `type … struct/interface`), TypeScript/JavaScript (functions, classes, arrow-function consts, interfaces, enums), and Python (`def`, `async def`, `class`, with decorators attached). A brace/indentation scanner that strips string literals and comments keeps a `{` inside a string from breaking a function. Anything else — Rust, Markdown, YAML — falls back to an overlapping line window, so no file type is skipped and no content is lost.
- **Stores** per chunk: the source text, project, file path, start/end line, and a SHA-256 of the text. Chunks are embedded as `<path>:<start>-<end> <kind> <name>` plus the source, so queries like *"the upload handler in the payments service"* match.
- **Re-ingestion is incremental.** Chunks are keyed by content hash, so a second run only re-embeds what actually changed:

| Re-run scenario | Re-embedded |
| --- | --- |
| Nothing changed | **0** chunks |
| One function edited in one file | only that function's chunks |
| Function moved to another line (same text) | 0 — metadata-only line-range update |
| File deleted | its chunks are tombstoned out of the index |

- **Batches** embedding calls and buckets them by length, so padding stays minimal (padding measurably perturbs mean-pooled output) and each model call stays cheap.
- **Returns a summary**: files scanned/skipped, chunks created/updated/unchanged/deleted, embedding time.


## Why local-first matters

- **Privacy by construction** — your notes, decisions, and personal facts are embedded and stored on your disk. There is no server to leak them from, because there is no server.
- **Zero network dependency** — the embedding model ([all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2), 384-dim) ships inside the npm package and runs via ONNX in-process, and the vector index is a local native addon. `npx localmind` works offline.
- **No keys, no quota, no cost** — nothing to sign up for, nothing to rate-limit you, nothing to expire.
- **You own the data** — one SQLite file plus one index file per project. Back it up, sync it, grep it, or delete it whenever you like.

## Install & run

```bash
npx localmind
```

That's it. First launch embeds and stores straight from the bundled model — no manual setup steps.

```bash
npx localmind remember "I prefer dark mode in every editor" --tags prefs
npx localmind recall "what theme should the UI use?"   # → [1] (score=0.44…) I prefer dark mode…
npx localmind list
npx localmind forget 1
```

## A realistic example: a large multi-service repo

Take PIXORA — 41 microservices in Go, Node and Python, tens of thousands of functions. `recall` is only useful here if it returns *functions*, not files, and if ingesting the tree doesn't cost an hour of CPU every time a branch is merged.

1. **Preview** — `ingest_codebase { path: "/work/pixora", project: "pixora", dry_run: true }` shows how many chunks you are about to create before spending any embedding time.
2. **Ingest service by service** rather than in one shot, so the agent can start using the index before the whole tree is done:
   `{ "path": "/work/pixora/services/payments", "project": "pixora" }` →
   `{ "path": "/work/pixora/services/search", "project": "pixora" }` →
   `{ "path": "/work/pixora/web", "project": "pixora" }`
3. **Ask with a budget.** `{ "project": "pixora", "query": "where do we validate webhook signatures?", "max_tokens": 2000 }` returns a handful of function chunks with `services/x/webhook.go:88-134`-style references — open those ranges instead of whole files.
4. **Re-run after each merge.** `{ "path": "/work/pixora", "project": "pixora" }` — unchanged functions are skipped by hash, so a 20k-chunk tree re-syncs in seconds (see the benchmark below).
5. Keep **decisions** in the same namespace: `{ "project": "pixora", "text": "We chose Redis for idempotency keys because the billing exporter assumes at-most-once.", "tags": "decision" }`. Notes and code chunks coexist and `recall` ranks across both.
6. **Isolate clients** with a second namespace — `{ "project": "pixora-acme" }` for a client-specific fork — and be certain a query in one can never surface the other's code.

## Token-budget recall

`recall` accepts `max_tokens`. When set, results are ranked, de-duplicated, then packed greedily in rank order until the budget is exhausted — the budget, not a fixed count, decides how much comes back.

Dedup catches three kinds of redundancy before packing: byte-identical chunks (same content hash — the copy-pasted helper that exists in 6 services), overlapping line ranges in the same file, and near-identical text (word-shingle Jaccard ≥ 0.85).

Every hit carries `path:start-end`, so the caller can open exactly that range and read more on demand. Tokens are estimated at ~4 characters each plus a small per-word allowance, biased high so the limit is a ceiling you can rely on.

```
# 3 result(s) from project 'pixora' using 203/220 tokens (3 lower-ranked dropped by budget/dedup)
[4] (score=0.5241, services/ledger/ledger.go:3-35) func FlushLedger(…) { … }
[7] (score=0.2033, services/us/tax.go:3-19) func ComputeTax(…) { … }
```

## Configure in Claude Code / Claude Desktop

### Claude Code

```bash
claude mcp add localmind -- npx -y localmind
```

Or in `.mcp.json` / `~/.claude.json`:

```json
{
  "mcpServers": {
    "localmind": {
      "command": "npx",
      "args": ["-y", "localmind"]
    }
  }
}
```

### Claude Desktop

Edit `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`, Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "localmind": {
      "command": "npx",
      "args": ["-y", "localmind"]
    }
  }
}
```

Restart Claude and the `remember` / `recall` / `forget` / `list_memories` / `ingest_codebase` tools appear.

## MCP client config reference

| Field | Value |
| --- | --- |
| command | `npx` |
| args | `["-y", "localmind"]` |

## Configuration (optional)

| Env var | Default | Purpose |
| --- | --- | --- |
| `LOCALMIND_HOME` | `~/.localmind` | Where `memory.db` and `indexes/` live. |
| `LOCALMIND_MODELS` | bundled `models/` | Directory containing the `Xenova/all-MiniLM-L6-v2` model folder, if you want to point at your own copy. |

## CLI

The same storage and ANN search the MCP server uses, from a terminal:

```bash
npx localmind ingest /work/pixora/services/payments --project pixora
npx localmind recall "where do we validate webhook signatures" --project pixora --max-tokens 2000
npx localmind recall "why Kafka and Redis" --project pixora --limit 5
npx localmind list --project pixora --kind code --limit 20
npx localmind forget 42 --project pixora
npx localmind projects
npx localmind db
```

## How it works

```
you / your agent
      │  MCP (stdio JSON-RPC)
      ▼
localmind server ─┬─► SQLite (better-sqlite3, WAL)          ~/.localmind/memory.db
                  │     text, tags, project, file path, line range,
                  │     content hash, timestamps — and the embedding BLOBs,
                  │     which are the source of truth for every index
                  │
                  ├─► HNSW index per project (hnswlib-node) ~/.localmind/indexes/<project>.hnsw
                  │     cosine space; labels are SQLite memory ids; deletes are
                  │     tombstones; persisted atomically and verified against
                  │     SQLite on open (rebuilt from the BLOBs if they disagree)
                  │
                  └─► all-MiniLM-L6-v2 (ONNX, in-process)   384-dim embeddings
                        bundled under models/; remote fetching hard-disabled
```

- **Vector search** — one [hnswlib](https://github.com/nmslib/hnswlib) graph per project via [hnswlib-node](https://github.com/yoshoku/hnswlib-node) (Apache-2.0, no network). Recall is approximate-nearest-neighbour: sub-second at six-figure vector counts instead of a full cosine scan per query. The benchmark measures the speedup rather than claiming it.
- **Incremental ingestion** — the SHA-256 of a chunk's text is its identity. Unchanged chunks keep their row, embedding and index slot; moved chunks get a metadata-only line-range update; vanished chunks are tombstoned (`markDelete`) and their slots reused for new vectors, so long-lived indexes never need a full rebuild.
- **Self-healing index** — SQLite owns the data, the index is a cache. If an index file is missing, corrupted, or disagrees with the database's id set (e.g. a crash between a write and a flush), it is rebuilt from the stored embedding BLOBs on next open.

## Data & privacy

- All data lives under one directory: `~/.localmind/` — `memory.db` plus one `indexes/<project>.hnsw` file per project (override the directory with `LOCALMIND_HOME`).
- The embedding model is loaded from the package's own `models/` directory; remote model fetching is disabled when the bundled model is present.
- No API keys, no telemetry, no network calls at runtime.
- Deleting a project is a file delete plus a SQL delete — nothing is synced anywhere.

## Platform note

better-sqlite3 and hnswlib-node ship prebuilt native binaries for the common platforms (macOS Intel/ARM, Linux x64/ARM, Windows x64); `npm install` downloads the right ones automatically. If a platform has no prebuilt binary, node-gyp builds from source (needs a C++ toolchain).

## Security notes

- **No network at runtime.** The embedding path loads the bundled ONNX model from disk; remote model fetching is disabled in code. Verify with: `grep -RIn "fetch(\|axios\|http" src/`
- **`npm audit` advisory** — `@xenova/transformers` pins `sharp@0.32.6` (image codec) which carries advisories. localmind's text-embedding path never invokes sharp (it's used only by image/audio pipelines), but if you treat supply-chain alerts as blockers, wait for upstream transformers.js ≥3 or vendor the model with your own loader.

## Development

```bash
git clone https://github.com/sahinsamrat75/localmind
cd localmind
npm install
npm run build

npm test                  # smoke test over a real MCP stdio client
npm run test:isolation    # project namespaces cannot leak into each other
npm run test:migration    # v1 db migrates with zero data loss
npm run test:chunker      # Go/TS/Python boundaries + fallback chunking
npm run test:incremental  # re-ingestion skips unchanged chunks by hash
npm run test:budget       # max_tokens bounds output, dedups, keeps rank order
npm run benchmark         # 20k+ synthetic chunks: latency, memory, disk
```

Every suite drives the real server as a child process over stdio with the official MCP client — the same path an agent takes — except the chunker suite, which is pure unit-level. The benchmark generates a synthetic Go/TS/Python monorepo, ingests it into two projects, and reports recall latency percentiles, ANN-vs-brute-force speedup, incremental re-ingestion cost, index reload/rebuild, and memory/disk footprint; it fails non-zero if any scale gate is missed. Results land in `test/benchmark-results.json`.

## License

[MIT](./LICENSE)

