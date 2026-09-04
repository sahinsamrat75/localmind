#!/usr/bin/env node
/**
 * localmind — a local-first memory server for AI agents.
 *
 * MCP server over stdio. Everything runs on the local machine: SQLite for
 * metadata, ONNX MiniLM for embeddings, and one HNSW index per project for
 * approximate-nearest-neighbour recall. No network, no API keys.
 *
 * Scale model: memories live in named `project` namespaces. Manual notes
 * (`remember`) and ingested code (`ingest_codebase`) coexist inside a project;
 * a search in one project can never see another project's data, because each
 * project is a physically separate vector index.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import path from "node:path";
import { z } from "zod";
import { MemoryStore, DEFAULT_PROJECT, KIND_MANUAL, KIND_CODE } from "./store.js";
import { VectorIndex } from "./vectorIndex.js";
import { embed } from "./embeddings.js";
import { resolveDataDir } from "./paths.js";
import { ingestCodebase } from "./ingest.js";
import { runRecall } from "./recall.js";
import { formatHit } from "./render.js";

/** Version reported over MCP and by the CLI. Kept in sync with package.json. */
export const VERSION = "0.2.0";

/** Parse a comma- or whitespace-separated tag string into a clean tag list. */
export function parseTags(raw: string | undefined): string[] {
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

/** Normalise a caller-supplied project name; blank or absent means 'default'. */
export function normalizeProject(raw: string | undefined): string {
  const trimmed = raw?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : DEFAULT_PROJECT;
}

/** Build a text-only MCP tool result. */
function textResult(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

/**
 * Render one recall hit.
 *
 * Code chunks are labelled `path:start-end` so the caller can open the file and
 * read more around them; manual memories show their tags instead.
 */
export { formatHit, hitCost } from "./render.js";

export interface ServerOptions {
  dbPath?: string;
  indexDir?: string;
  /** Log destination. Defaults to stderr, never stdout (stdio MCP owns it). */
  log?: (msg: string) => void;
}

export interface Localmind {
  server: McpServer;
  store: MemoryStore;
  index: VectorIndex;
  ready: Promise<void>;
  close: () => void;
}

/**
 * Build the MCP server plus the storage and index handles behind it.
 *
 * Paths are injectable for tests. When only a dbPath is given the index dir
 * sits beside it, so an isolated test directory stays fully isolated.
 */
export function createLocalmindServer(
  dbPathOrOptions?: string | ServerOptions,
  legacyIndexDir?: string
): Localmind {
  const options: ServerOptions =
    typeof dbPathOrOptions === "string"
      ? { dbPath: dbPathOrOptions, indexDir: legacyIndexDir }
      : (dbPathOrOptions ?? {});

  const dataDir = resolveDataDir();
  const dbPath = options.dbPath ?? dataDir.memoryDb;
  const indexDir = options.indexDir ?? path.join(path.dirname(dbPath), "indexes");
  const log = options.log ?? ((msg: string) => console.error(`[localmind] ${msg}`));

  // Opening the store runs the v1 -> v2 schema migration when required.
  const store = new MemoryStore(dbPath);
  const index = new VectorIndex({
    indexDir,
    liveIds: (project) => store.idsForProject(project),
    vectors: (project) => store.embeddingsForProject(project),
    log,
  });

  // Serialize all tool operations. MCP processes requests concurrently, and
  // embedding + SQLite + index writes are order-sensitive (a recall that races
  // a remember must see the remembered row). A single-client local server
  // wants strict FIFO execution.
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
    {
      instructions:
        "Local-first memory server. Memories are grouped into named projects; pass 'project' to scope any call. " +
        "Use remember/recall/forget/list_memories for notes and ingest_codebase to index a repository. " +
        "recall accepts max_tokens to bound how much context the results consume.",
    }
  );

  const projectSchema = z
    .string()
    .optional()
    .describe(
      `Project namespace to operate in (e.g. 'pixora'). Defaults to '${DEFAULT_PROJECT}'. ` +
        `Namespaces are fully isolated: a recall in one project never returns another project's memories.`
    );

  // ---------------------------------------------------------------------
  // Tool 1: remember — store a manual memory in a project namespace.
  // ---------------------------------------------------------------------
  server.tool(
    "remember",
    "Store a memory (a fact, preference, decision, note) with optional tags, inside a project namespace. Coexists with code chunks ingested into the same project.",
    {
      text: z.string().min(1).describe("The memory text to store."),
      tags: z
        .string()
        .optional()
        .describe("Optional comma-separated tags, e.g. 'work, project-alpha'."),
      project: projectSchema,
    },
    async ({ text, tags, project }) =>
      serialize(async () => {
        const ns = normalizeProject(project);
        const embedding = await embed(text);
        const tagList = parseTags(tags);
        const { id } = store.insert({
          text,
          tags: tagList,
          embedding,
          project: ns,
          kind: KIND_MANUAL,
        });
        index.add(ns, id, embedding);
        return textResult(
          `Remembered (id=${id}, project=${ns}${tagList.length ? `, tags=${tagList.join(",")}` : ""}): ${text}`
        );
      })()
  );

  // ---------------------------------------------------------------------
  // Tool 2: recall — ANN search within one project, optionally budgeted.
  // ---------------------------------------------------------------------
  server.tool(
    "recall",
    "Semantically search one project's memories using the HNSW index. Returns ranked results; each code hit carries its file path and line range. Set max_tokens to get back as many top results as fit a context budget instead of a fixed count.",
    {
      query: z.string().min(1).describe("The natural-language query to search for."),
      limit: z
        .number()
        .int()
        .positive()
        .max(200)
        .optional()
        .describe("Max number of results to return (default 5, ignored when max_tokens is set)."),
      project: projectSchema,
      max_tokens: z
        .number()
        .int()
        .positive()
        .max(200000)
        .optional()
        .describe(
          "Token budget. When set, results are deduplicated and packed into the budget in rank order, and the budget — not the limit — decides how many come back."
        ),
      min_score: z
        .number()
        .min(-1)
        .max(1)
        .optional()
        .describe("Drop hits below this cosine similarity (e.g. 0.2)."),
      include_text: z
        .boolean()
        .optional()
        .describe("Include the full memory text in each hit (default true)."),
    },
    async ({ query, limit, project, max_tokens, min_score, include_text }) =>
      serialize(async () => {
        const ns = normalizeProject(project);
        const withText = include_text ?? true;
        const result = await runRecall(
          { store, index },
          {
            query,
            project: ns,
            limit,
            maxTokens: max_tokens,
            minScore: min_score,
          }
        );

        if (result.candidates === 0) {
          return textResult(`No memories stored in project '${ns}' yet.`);
        }
        if (result.hits.length === 0) {
          return textResult(
            `No matching memories in project '${ns}'${min_score !== undefined ? ` above score=${min_score}` : ""}.`
          );
        }

        const lines = result.hits.map((c) =>
          withText ? formatHit(c.memory, c.score) : formatHit(c.memory, c.score).split(" ").slice(0, 2).join(" ")
        );
        const dropped = result.deduped - result.hits.length;
        const header =
          max_tokens !== undefined
            ? `# ${result.hits.length} result(s) from project '${ns}' using ${result.tokensUsed}/${max_tokens} tokens` +
              (dropped > 0 ? ` (${dropped} lower-ranked dropped by budget/dedup)` : "")
            : `# ${result.hits.length} result(s) from project '${ns}' (~${result.tokensUsed} tokens)`;
        return textResult([header, ...lines].join("\n"));
      })()
  );

  // ---------------------------------------------------------------------
  // Tool 3: forget — delete one memory from one project.
  // ---------------------------------------------------------------------
  server.tool(
    "forget",
    "Delete a stored memory by id. The id is only resolved inside the given project, so a memory in another namespace cannot be deleted by accident.",
    {
      id: z.number().int().positive().describe("The id of the memory to delete."),
      project: projectSchema,
    },
    async ({ id, project }) =>
      serialize(async () => {
        const ns = normalizeProject(project);
        const deleted = store.remove(id, ns);
        if (!deleted) {
          return textResult(`No memory found with id=${id} in project '${ns}'.`);
        }
        index.remove(ns, id);
        index.flush(ns);
        return textResult(`Deleted memory (id=${id}) from project '${ns}'.`);
      })()
  );

  // ---------------------------------------------------------------------
  // Tool 4: list_memories — enumerate one project.
  // ---------------------------------------------------------------------
  server.tool(
    "list_memories",
    "List memories in a project, newest first. Optionally filter by tag or by kind ('manual' for notes, 'code' for ingested chunks).",
    {
      limit: z
        .number()
        .int()
        .positive()
        .max(1000)
        .optional()
        .describe("Max number of memories to return (default 20)."),
      tag: z.string().optional().describe("Only return memories carrying this tag."),
      project: projectSchema,
      kind: z
        .enum([KIND_MANUAL, KIND_CODE])
        .optional()
        .describe("Filter by memory kind: 'manual' (remember) or 'code' (ingest_codebase)."),
    },
    async ({ limit, tag, project, kind }) =>
      serialize(async () => {
        const ns = normalizeProject(project);
        const max = limit ?? 20;
        const memories = store.list(ns, max, tag?.trim().toLowerCase() || undefined, kind);
        if (memories.length === 0) {
          return textResult(`No memories${tag ? ` tagged '${tag}'` : ""} in project '${ns}'.`);
        }
        const lines = memories.map((m) => {
          const meta = [m.created_at];
          if (m.file_path && m.line_start !== null) meta.push(`${m.file_path}:${m.line_start}-${m.line_end}`);
          if (m.tags.length) meta.push(`tags=${m.tags.join(",")}`);
          const body = m.text.length > 200 ? `${m.text.slice(0, 200)}…` : m.text;
          return `[${m.id}] (${meta.join(", ")}) ${body}`;
        });
        return textResult([`# ${store.count(ns)} memories in project '${ns}'`, ...lines].join("\n"));
      })()
  );

  // ---------------------------------------------------------------------
  // Tool 5: ingest_codebase — index a repository into a project namespace.
  // ---------------------------------------------------------------------
  server.tool(
    "ingest_codebase",
    "Index a source tree into a project namespace: walks the directory respecting .gitignore, splits files at function/class boundaries (Go, JS/TS, Python; overlapping line windows otherwise), and embeds each chunk. Incremental — re-running only re-embeds chunks whose SHA-256 changed and removes chunks whose code disappeared. Returns a summary of files scanned and chunks created/updated/unchanged/deleted.",
    {
      path: z.string().min(1).describe("Absolute path of the directory to ingest."),
      project: z.string().min(1).describe("Project namespace to ingest into (e.g. 'pixora')."),
      include: z
        .string()
        .optional()
        .describe("Comma-separated path substrings or globs; only matching files are ingested."),
      exclude: z
        .string()
        .optional()
        .describe("Comma-separated path substrings or globs to skip."),
      respect_gitignore: z
        .boolean()
        .optional()
        .describe("Honour .gitignore files while walking (default true)."),
      max_file_bytes: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Skip files larger than this (default 524288)."),
      max_files: z.number().int().positive().optional().describe("Safety cap on files walked."),
      chunk_lines: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Window size for the line-based fallback chunker (default 60)."),
      chunk_overlap: z
        .number()
        .int()
        .min(0)
        .optional()
        .describe("Overlap between fallback line windows (default 12)."),
      batch_size: z
        .number()
        .int()
        .positive()
        .max(256)
        .optional()
        .describe("Texts per embedding call (default 32)."),
      dry_run: z
        .boolean()
        .optional()
        .describe("Report what would change without writing anything."),
    },
    async ({
      path: root,
      project,
      include,
      exclude,
      respect_gitignore,
      max_file_bytes,
      max_files,
      chunk_lines,
      chunk_overlap,
      batch_size,
      dry_run,
    }) =>
      serialize(async () => {
        const ns = normalizeProject(project);
        const split = (s: string | undefined) =>
          s ? s.split(",").map((x) => x.trim()).filter(Boolean) : undefined;
        const summary = await ingestCodebase(
          {
            project: ns,
            root,
            include: split(include),
            exclude: split(exclude),
            respectGitignore: respect_gitignore,
            maxFileBytes: max_file_bytes,
            maxFiles: max_files,
            chunkLines: chunk_lines,
            chunkOverlap: chunk_overlap,
            batchSize: batch_size,
            dryRun: dry_run,
            log,
          },
          store,
          index
        );
        return textResult(JSON.stringify(summary, null, 2));
      })()
  );

  // ---------------------------------------------------------------------
  // Tool 6: list_projects — discover namespaces.
  // ---------------------------------------------------------------------
  server.tool(
    "list_projects",
    "List the project namespaces that currently hold memories, with counts and index sizes.",
    {},
    async () =>
      serialize(async () => {
        const rows = store.projectCounts();
        if (rows.length === 0) return textResult("No memories stored in any project yet.");
        const lines = rows.map((r) => {
          let indexed = 0;
          try {
            indexed = index.size(r.project);
          } catch {
            indexed = 0;
          }
          return `${r.project}: ${r.count} memories (${indexed} indexed)`;
        });
        return textResult(lines.join("\n"));
      })()
  );

  const ready = Promise.resolve();

  function close(): void {
    index.close();
    store.close();
  }

  return { server, store, index, ready, close };
}

/** Run the server over stdio. Used by the CLI entry point. */
export async function runStdio(): Promise<void> {
  const { server, close } = createLocalmindServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);

  // The ANN index is written on a debounce, so make sure a shutdown does not
  // drop the last mutations — otherwise the next start pays for a rebuild.
  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.once(signal, () => {
      try {
        close();
      } catch {
        /* best effort */
      }
      process.exit(0);
    });
  }
  process.once("exit", () => {
    try {
      close();
    } catch {
      /* best effort */
    }
  });

  // Keep the process alive; the stdio transport handles the message loop.
  await new Promise<void>(() => {});
}


