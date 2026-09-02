#!/usr/bin/env node
/**
 * localmind CLI.
 *
 *   localmind                 Start the MCP server on stdio (what Claude Code /
 *                             Claude Desktop launch).
 *   localmind remember "..." [--tags a,b]   Store a memory from the terminal.
 *   localmind recall "..."    [--limit n]   Semantic search from the terminal.
 *   localmind list [--tag t] [--limit n]    List memories.
 *   localmind forget <id>                   Delete a memory by id.
 *   localmind db                            Print the database location.
 *   localmind --version                     Print version.
 */
import { runStdio, createLocalmindServer, VERSION } from "./index.js";
import { resolveDataDir } from "./paths.js";
import { cosineSimilarity } from "./embeddings.js";

const USAGE = `localmind — local-first memory server for AI agents (v${VERSION})

Usage:
  localmind                                 Start the MCP server on stdio
  localmind remember <text> [--tags a,b]    Store a memory
  localmind recall <query> [--limit n]      Semantic search over memories
  localmind list [--tag <tag>] [--limit n]  List memories (newest first)
  localmind forget <id>                     Delete a memory by id
  localmind db                              Print the database path
  localmind --version                       Print version
  localmind --help                          Show this help

Environment:
  LOCALMIND_HOME   Data directory (default: ~/.localmind)
  LOCALMIND_MODELS Directory containing the all-MiniLM-L6-v2 model folder`;

/** Read a `--flag value` pair out of the raw argv list. */
function flagValue(args: string[], name: string): string | undefined {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}

function parseTags(raw: string | undefined): string[] {
  if (!raw) return [];
  return [...new Set(raw.split(/[,\s]+/).map((t) => t.trim().toLowerCase()).filter(Boolean))];
}

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const command = args[0];

  if (command === "--help" || command === "-h" || command === "help") {
    console.log(USAGE);
    return 0;
  }
  if (command === "--version" || command === "-v" || command === "version") {
    console.log(VERSION);
    return 0;
  }
  if (command === "db") {
    console.log(resolveDataDir().memoryDb);
    return 0;
  }

  // No command (or explicit serve) = run the MCP stdio server.
  if (command === undefined || command === "serve" || command === "mcp") {
    await runStdio();
    return 0;
  }

  const { store } = createLocalmindServer();

  try {
    switch (command) {
      case "remember": {
        const text = args.slice(1).filter((a) => !a.startsWith("--") && a !== flagValue(args, "--tags")).join(" ").trim();
        if (!text) {
          console.error('Usage: localmind remember "some fact" [--tags work,ideas]');
          return 2;
        }
        const { embed } = await import("./embeddings.js");
        const embedding = await embed(text);
        const tags = parseTags(flagValue(args, "--tags"));
        const { id } = store.insert(text, tags, embedding);
        console.log(`Remembered (id=${id}${tags.length ? `, tags=${tags.join(",")}` : ""}): ${text}`);
        return 0;
      }

      case "recall": {
        const query = args.slice(1).filter((a) => !a.startsWith("--") && a !== flagValue(args, "--limit")).join(" ").trim();
        if (!query) {
          console.error('Usage: localmind recall "a question" [--limit 5]');
          return 2;
        }
        const { embed } = await import("./embeddings.js");
        const limit = Number(flagValue(args, "--limit") ?? 5);
        const rows = store.getAllWithEmbeddings();
        if (rows.length === 0) {
          console.log("No memories stored yet.");
          return 0;
        }
        const queryVec = await embed(query);
        const scored = rows
          .map(({ memory, embedding }) => ({ memory, score: cosineSimilarity(queryVec, embedding) }))
          .sort((a, b) => b.score - a.score)
          .slice(0, Number.isFinite(limit) && limit > 0 ? limit : 5);
        for (const { memory, score } of scored) {
          const tags = memory.tags.length ? `, tags=${memory.tags.join(",")}` : "";
          console.log(`[${memory.id}] (score=${score.toFixed(4)}${tags}) ${memory.text}`);
        }
        return 0;
      }

      case "list": {
        const limit = Number(flagValue(args, "--limit") ?? 20);
        const tag = flagValue(args, "--tag");
        const memories = store.list(
          Number.isFinite(limit) && limit > 0 ? limit : 20,
          tag?.trim().toLowerCase() || undefined
        );
        if (memories.length === 0) {
          console.log(tag ? `No memories tagged '${tag}'.` : "No memories stored yet.");
          return 0;
        }
        for (const m of memories) {
          const tags = m.tags.length ? `, tags=${m.tags.join(",")}` : "";
          console.log(`[${m.id}] (${m.created_at}${tags}) ${m.text}`);
        }
        return 0;
      }

      case "forget": {
        const id = Number(args[1]);
        if (!Number.isInteger(id) || id <= 0) {
          console.error("Usage: localmind forget <id>");
          return 2;
        }
        if (store.remove(id)) {
          console.log(`Deleted memory (id=${id}).`);
        } else {
          console.error(`No memory found with id=${id}.`);
          return 1;
        }
        return 0;
      }

      default:
        console.error(`Unknown command '${command}'.\n\n${USAGE}`);
        return 2;
    }
  } finally {
    store.close();
  }
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error("localmind:", err instanceof Error ? err.message : err);
    process.exit(1);
  });
