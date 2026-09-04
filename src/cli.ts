#!/usr/bin/env node
/**
 * localmind CLI — the same storage, index and ANN search the MCP server uses.
 *
 *   localmind                              Start the MCP server on stdio.
 *   localmind remember "..." [--tags a,b] [--project p]
 *   localmind recall "..." [--limit n] [--project p] [--max-tokens N]
 *   localmind list [--tag t] [--limit n] [--project p] [--kind manual|code]
 *   localmind forget <id> [--project p]
 *   localmind ingest <path> --project p [--include g1,g2] [--exclude g1,g2]
 *   localmind projects | db | --version | --help
 */
import path from "node:path";
import {
  createLocalmindServer,
  runStdio,
  VERSION,
  parseTags,
  normalizeProject,
  formatHit,
  hitCost,
} from "./server.js";
import { embed } from "./embeddings.js";
import { resolveDataDir } from "./paths.js";
import { ingestCodebase } from "./ingest.js";
import { dedupeCandidates, packToBudget, type Candidate } from "./tokens.js";
import { DEFAULT_PROJECT, KIND_MANUAL } from "./store.js";

const USAGE = `localmind — local-first memory server for AI agents (v${VERSION})

Usage:
  localmind                                  Start the MCP server on stdio
  localmind remember <text> [--tags a,b] [--project p]
  localmind recall <query> [--limit n] [--project p] [--max-tokens N]
  localmind list [--tag <tag>] [--limit n] [--project p] [--kind manual|code]
  localmind forget <id> [--project p]
  localmind ingest <path> --project <p> [--include g1,g2] [--exclude g1,g2]
  localmind projects                         List namespaces with counts
  localmind db                               Print database + index paths
  localmind --version | --help

Environment:
  LOCALMIND_HOME   Data directory (default: ~/.localmind)
  LOCALMIND_MODELS Directory containing the all-MiniLM-L6-v2 model folder`;

/** Read a `--flag value` pair out of the raw argv list. */
function flagValue(args: string[], name: string): string | undefined {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}

/** Positional arguments: everything that is not a flag or a flag's value. */
function positionals(args: string[], valueFlags: string[]): string[] {
  const consumed = new Set<string>();
  for (const f of valueFlags) {
    const i = args.indexOf(f);
    if (i >= 0) consumed.add(args[i + 1]);
  }
  return args.filter((a) => !a.startsWith("--") && !consumed.has(a));
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
    const d = resolveDataDir();
    console.log(`database: ${d.memoryDb}`);
    console.log(`indexes:  ${d.indexDir}`);
    return 0;
  }

  // No command (or explicit serve) = run the MCP stdio server.
  if (command === undefined || command === "serve" || command === "mcp") {
    await runStdio();
    return 0;
  }

  const { store, index, close } = createLocalmindServer();
  const project = normalizeProject(flagValue(args, "--project"));

  try {
    switch (command) {
      case "remember": {
        const text = positionals(args, ["--tags", "--project"]).join(" ").trim();
        if (!text) {
          console.error('Usage: localmind remember "some fact" [--tags work,ideas] [--project p]');
          return 2;
        }
        const embedding = await embed(text);
        const tags = parseTags(flagValue(args, "--tags"));
        const { id } = store.insert({ text, tags, embedding, project, kind: KIND_MANUAL });
        index.add(project, id, embedding);
        index.flush(project);
        console.log(
          `Remembered (id=${id}, project=${project}${tags.length ? `, tags=${tags.join(",")}` : ""}): ${text}`
        );
        return 0;
      }

      case "recall": {
        const query = positionals(args, ["--limit", "--project", "--max-tokens", "--min-score"])
          .slice(1)
          .join(" ")
          .trim();
        if (!query) {
          console.error('Usage: localmind recall "a question" [--limit 5] [--project p] [--max-tokens 2000]');
          return 2;
        }
        const limit = Number(flagValue(args, "--limit") ?? 5);
        const maxTokens = Number(flagValue(args, "--max-tokens"));
        const minScore = flagValue(args, "--min-score");
        const queryVec = await embed(query);
        const budgeted = Number.isFinite(maxTokens) && maxTokens > 0;
        const fetchK = budgeted ? 400 : Math.min(400, (Number.isFinite(limit) ? limit : 5) * 6);
        const hits = index.search(project, queryVec, fetchK);
        if (hits.length === 0) {
          console.log(`No memories stored in project '${project}' yet.`);
          return 0;
        }
        const scoreById = new Map<number, number>();
        for (const h of hits) {
          if (minScore === undefined || h.score >= Number(minScore)) scoreById.set(h.id, h.score);
        }
        const candidates: Candidate[] = store
          .getByIds([...scoreById.keys()], project)
          .map((memory) => ({ memory, score: scoreById.get(memory.id) ?? 0 }))
          .sort((a, b) => b.score - a.score);
        const deduped = dedupeCandidates(candidates);
        const chosen = budgeted
          ? packToBudget(deduped, maxTokens, (c) => hitCost(c.memory)).items
          : deduped.slice(0, Number.isFinite(limit) && limit > 0 ? limit : 5);
        for (const c of chosen) console.log(formatHit(c.memory, c.score));
        return 0;
      }

      case "list": {
        const limit = Number(flagValue(args, "--limit") ?? 20);
        const tag = flagValue(args, "--tag");
        const kind = flagValue(args, "--kind");
        const memories = store.list(
          project,
          Number.isFinite(limit) && limit > 0 ? limit : 20,
          tag?.trim().toLowerCase() || undefined,
          kind || undefined
        );
        if (memories.length === 0) {
          console.log(`No memories${tag ? ` tagged '${tag}'` : ""} in project '${project}'.`);
          return 0;
        }
        for (const m of memories) {
          const meta = [m.created_at];
          if (m.file_path) meta.push(`${m.file_path}:${m.line_start}-${m.line_end}`);
          if (m.tags.length) meta.push(`tags=${m.tags.join(",")}`);
          const body = m.text.length > 160 ? `${m.text.slice(0, 160)}…` : m.text;
          console.log(`[${m.id}] (${meta.join(", ")}) ${body}`);
        }
        return 0;
      }

      case "forget": {
        const id = Number(args[1]);
        if (!Number.isInteger(id) || id <= 0) {
          console.error("Usage: localmind forget <id> [--project p]");
          return 2;
        }
        const deleted = store.remove(id, project);
        if (!deleted) {
          console.error(`No memory found with id=${id} in project '${project}'.`);
          return 1;
        }
        index.remove(project, id);
        index.flush(project);
        console.log(`Deleted memory (id=${id}) from project '${project}'.`);
        return 0;
      }

      case "ingest": {
        const root = positionals(args, ["--project", "--include", "--exclude"])[1];
        if (!root) {
          console.error('Usage: localmind ingest <path> --project <name> [--include g1,g2] [--exclude g1,g2]');
          return 2;
        }
        if (!flagValue(args, "--project")) {
          console.error(
            `ingest requires --project (pass --project ${DEFAULT_PROJECT} if that is what you want).`
          );
          return 2;
        }
        const split = (s: string | undefined) =>
          s ? s.split(",").map((x) => x.trim()).filter(Boolean) : undefined;
        const summary = await ingestCodebase(
          {
            project,
            root: path.resolve(root),
            include: split(flagValue(args, "--include")),
            exclude: split(flagValue(args, "--exclude")),
            log: (m) => console.error(`[localmind] ${m}`),
          },
          store,
          index
        );
        console.log(JSON.stringify(summary, null, 2));
        return 0;
      }

      case "projects": {
        const rows = store.projectCounts();
        if (rows.length === 0) {
          console.log("No memories stored in any project yet.");
          return 0;
        }
        for (const r of rows) console.log(`${r.project}: ${r.count} memories`);
        return 0;
      }

      default:
        console.error(`Unknown command '${command}'.\n\n${USAGE}`);
        return 2;
    }
  } finally {
    close();
  }
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error("localmind:", err instanceof Error ? err.message : err);
    process.exit(1);
  });
