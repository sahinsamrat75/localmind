# localmind

**Local-first semantic memory for AI agents.**

localmind is an [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that gives your AI agent a persistent memory that never leaves your machine. Memories are stored in a single SQLite file on your disk and searched with embeddings produced **entirely on-device** — no API keys, no accounts, no cloud, no telemetry.

## What it does

Your agent gets 4 MCP tools:

| Tool | What it does |
| --- | --- |
| `remember` | Store a memory (fact, preference, decision) with optional tags. |
| `recall` | Semantic search over memories — finds the right memory even when the query shares no keywords with it. |
| `forget` | Delete a memory by id. |
| `list_memories` | Browse stored memories, filterable by tag. |

Example: remember *"Sofia is allergic to peanuts"* → later ask *"what must I avoid cooking for my friend?"* → the allergy memory comes back ranked first, despite zero word overlap.

## Why local-first matters

- **Privacy by construction** — your notes, decisions, and personal facts are embedded and stored on your disk (`~/.localmind/memory.db`). There is no server to leak them from, because there is no server.
- **Zero network dependency** — the embedding model ([all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2), 384-dim) ships inside the npm package and runs via ONNX in-process. `npx localmind` works offline.
- **No keys, no quota, no cost** — nothing to sign up for, nothing to rate-limit you, nothing to expire.
- **You own the data** — one SQLite file. Back it up, sync it, grep it, or delete it whenever you like.

## Install & run

```bash
npx localmind
```

That's it. First launch embeds and stores straight from the bundled model — no manual setup steps.

Use the bundled CLI to try it from your terminal:

```bash
npx localmind remember "I prefer dark mode in every editor" --tags prefs
npx localmind recall "what theme should the UI use?"   # → [1] (score=0.44…) I prefer dark mode…
npx localmind list
npx localmind forget 1
```

## 30-second example

```bash
$ npx localmind remember "The wifi password at the Lisbon office is capital-lisbon-2024" --tags wifi
Remembered (id=1, tags=wifi): The wifi password at the Lisbon office is capital-lisbon-2024

$ npx localmind recall "how do I get online at the Portugal HQ?"
[1] (score=0.3841, tags=wifi) The wifi password at the Lisbon office is capital-lisbon-2024

$ npx localmind forget 1
Deleted memory (id=1).
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

Restart Claude and the `remember` / `recall` / `forget` / `list_memories` tools appear.

## MCP client config reference

| Field | Value |
| --- | --- |
| command | `npx` |
| args | `["-y", "localmind"]` |

## Configuration (optional)

| Env var | Default | Purpose |
| --- | --- | --- |
| `LOCALMIND_HOME` | `~/.localmind` | Where `memory.db` lives. |
| `LOCALMIND_MODELS` | bundled `models/` | Directory containing the `Xenova/all-MiniLM-L6-v2` model folder, if you want to point at your own copy. |

## How it works

```
you / your agent
      │  MCP (stdio JSON-RPC)
      ▼
localmind server  ──►  SQLite (better-sqlite3)  ──►  ~/.localmind/memory.db
      │
      └──►  all-MiniLM-L6-v2 (ONNX, in-process)  ──►  384-dim embeddings
```

- **Storage** — [better-sqlite3](https://github.com/WiseLibs/better-sqlite3), WAL mode, one table (`memories`: id, text, tags, embedding BLOB, created_at).
- **Embeddings** — [@xenova/transformers](https://github.com/xenova/transformers.js) running the quantized ONNX MiniLM model that ships in this package; vectors are L2-normalized and similarity is cosine (dot product).
- **Search** — full cosine scan at query time; instant for the tens of thousands of memories a personal agent accumulates.

## Data & privacy

- All data lives in a single file: `~/.localmind/memory.db` (override with `LOCALMIND_HOME`).
- The embedding model is loaded from the package's own `models/` directory; remote model fetching is disabled when the bundled model is present.
- No API keys, no telemetry, no network calls at runtime.

## Platform note

better-sqlite3 ships prebuilt native binaries for the common platforms (macOS Intel/ARM, Linux x64/ARM, Windows x64). `npm install` downloads the right one automatically. `npx localmind` uses npx's cache, so the binary is fetched once and reused.

## Security notes

- **No network at runtime.** The embedding path loads the bundled ONNX model from disk; remote model fetching is disabled in code. Verify with: `grep -RIn "fetch(\|axios\|http" src/`
- **`npm audit` advisory** — `@xenova/transformers` pins `sharp@0.32.6` (image codec) which carries advisories. localmind's text-embedding path never invokes sharp (it's used only by image/audio pipelines), but if you treat supply-chain alerts as blockers, wait for upstream transformers.js ≥3 or vendor the model with your own loader.

## Development

```bash
git clone https://github.com/sahinsamrat75/localmind
cd localmind
npm install
npm run build
npm test        # end-to-end test over a real MCP stdio client
```

The test suite spawns the real server, remembers 3 facts, semantically recalls them, deletes one, and verifies `list_memories` reflects the deletion.

## License

[MIT](./LICENSE)
