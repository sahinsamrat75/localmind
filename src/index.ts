#!/usr/bin/env node
/**
 * localmind — a local-first memory server for AI agents.
 *
 * Exposes an MCP server over stdio with tools to remember, recall, forget and
 * list memories. Everything (storage + embeddings) runs on the local machine:
 * SQLite database + ONNX MiniLM embeddings, no network, no API keys.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { MemoryStore } from "./store.js";
import { embed, cosineSimilarity } from "./embeddings.js";
import { resolveDataDir } from "./paths.js";

/** Version reported over MCP and by the CLI. Kept in sync with package.json. */
export const VERSION = "0.1.0";

/** Parse a comma- or whitespace-separated tag string into a clean tag list. */
function parseTags(raw: string | undefined): string[] {
  if (!raw) return [];
  return [
    ...new Set(
      raw
        .split(/[,\s]+/)
        .map((t) => t.trim().toLowerCase())
        .filter((t) => t.length > 0)
    ),
  ];
}

/** Build a text-only MCP tool result. */
function textResult(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

export interface LocalmindServer {
  server: McpServer;
  store: MemoryStore;
  ready: Promise<void>;
}

/**
 * Build the MCP server. `dbPath` is injectable for tests; defaults to
 * <dataDir>/memory.db where dataDir is $LOCALMIND_HOME or ~/.localmind.
 */
export function createLocalmindServer(dbPath?: string): LocalmindServer {
  const resolvedDbPath = dbPath ?? resolveDataDir().memoryDb;
  const store = new MemoryStore(resolvedDbPath);

  // Serialize all tool operations. MCP processes requests concurrently, and
  // embedding + SQLite writes are order-sensitive (e.g. a recall that races a
  // remember must see the remembered row). A single-client local server wants
  // strict FIFO execution.
  let tail: Promise<unknown> = Promise.resolve();
  function serialize<R>(fn: () => Promise<R>): () => Promise<R> {
    return () => {
      const run = tail.then(fn, fn);
      tail = run.catch(() => {});
      return run;
    };
  }

  const server = new McpServer(
    { name: "localmind", version: VERSION },
    { instructions: "Local-first memory server. Use remember/recall/forget/list_memories to manage the agent's persistent memory." }
  );

  // Tool 1: remember — store a memory with its embedding.
  server.tool(
    "remember",
    "Store a memory (a fact, preference, decision, note) with optional tags for later semantic recall.",
    {
      text: z.string().min(1).describe("The memory text to store."),
      tags: z
        .string()
        .optional()
        .describe("Optional comma-separated tags, e.g. 'work, project-alpha'."),
    },
    async ({ text, tags }) => {
      return serialize(async () => {
        const embedding = await embed(text);
        const tagList = parseTags(tags);
        const { id } = store.insert(text, tagList, embedding);
        return textResult(
          `Remembered (id=${id}${tagList.length ? `, tags=${tagList.join(",")}` : ""}): ${text}`
        );
      })();
    }
  );

  // Tool 2: recall — semantic search over stored memories.
  server.tool(
    "recall",
    "Semantically search stored memories for the given query and return ranked results.",
    {
      query: z.string().min(1).describe("The natural-language query to search for."),
      limit: z
        .number()
        .int()
        .positive()
        .max(50)
        .optional()
        .describe("Max number of results to return (default 5)."),
    },
    async ({ query, limit }) => {
      return serialize(async () => {
        const rows = store.getAllWithEmbeddings();
        if (rows.length === 0) {
          return textResult("No memories stored yet.");
        }
        const queryVec = await embed(query);
        const max = limit ?? 5;
        const scored = rows
          .map(({ memory, embedding }) => ({ memory, score: cosineSimilarity(queryVec, embedding) }))
          .sort((a, b) => b.score - a.score)
          .slice(0, max);

        const lines = scored.map(
          ({ memory, score }) =>
            `[${memory.id}] (score=${score.toFixed(4)}${memory.tags.length ? `, tags=${memory.tags.join(",")}` : ""}) ${memory.text}`
        );
        return textResult(lines.join("\n"));
      })();
    }
  );

  // Tool 3: forget — delete a memory by id.
  server.tool(
    "forget",
    "Delete a stored memory by its id.",
    {
      id: z.number().int().positive().describe("The id of the memory to delete."),
    },
    async ({ id }) => {
      return serialize(async () => {
        const deleted = store.remove(id);
        return textResult(
          deleted ? `Deleted memory (id=${id}).` : `No memory found with id=${id}.`
        );
      })();
    }
  );

  // Tool 4: list_memories — enumerate memories, optionally by tag.
  server.tool(
    "list_memories",
    "List stored memories, newest first. Optionally filter by tag and/or limit the number returned.",
    {
      limit: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Max number of memories to return (default 20)."),
      tag: z
        .string()
        .optional()
        .describe("Only return memories carrying this tag."),
    },
    async ({ limit, tag }) => {
      return serialize(async () => {
        const max = limit ?? 20;
        const memories = store.list(max, tag?.trim().toLowerCase() || undefined);
        if (memories.length === 0) {
          return textResult(tag ? `No memories tagged '${tag}'.` : "No memories stored yet.");
        }
        const lines = memories.map(
          (m) => `[${m.id}] (${m.created_at}${m.tags.length ? `, tags=${m.tags.join(",")}` : ""}) ${m.text}`
        );
        return textResult(lines.join("\n"));
      })();
    }
  );

  const ready = Promise.resolve();

  return { server, store, ready };
}

/** Run the server over stdio. Used by the CLI entry point. */
export async function runStdio(): Promise<void> {
  const { server } = createLocalmindServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // Keep the process alive; stdio transport handles the message loop.
  await new Promise<void>(() => {});
}

// When invoked directly (node dist/index.js), start on stdio.
if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  runStdio().catch((err) => {
    console.error("localmind failed to start:", err);
    process.exit(1);
  });
}
