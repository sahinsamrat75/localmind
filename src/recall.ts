import { embed } from "./embeddings.js";
import { dedupeCandidates, packToBudget } from "./tokens.js";
import { hitCost } from "./render.js";
import type { Memory } from "./store.js";
import type { MemoryStore } from "./store.js";
import type { VectorIndex } from "./vectorIndex.js";

/**
 * The recall pipeline, shared by the MCP tool, the CLI and the benchmark so
 * that what gets measured is exactly what agents get.
 *
 *   embed query -> ANN search in ONE project -> over-fetched candidates
 *   -> dedupe -> (optional) pack into a token budget
 *
 * The over-fetch matters: dedup and the budget both throw candidates away, so
 * asking the index for exactly `limit` neighbours would quietly under-deliver
 * the moment a repo contains copy-pasted code.
 */

export interface RecallParams {
  query: string;
  project: string;
  limit?: number;
  maxTokens?: number;
  minScore?: number;
}

export interface RecallHit {
  memory: Memory;
  score: number;
}

export interface RecallResult {
  hits: RecallHit[];
  /** Candidates the ANN index returned before dedup. */
  candidates: number;
  /** Candidates removed as duplicates. */
  deduped: number;
  tokensUsed: number;
  /** Deduped candidates dropped because the budget ran out. */
  truncated: number;
  /** Milliseconds spent in the ANN index lookup alone. */
  searchMs: number;
  /** Milliseconds spent embedding the query. */
  embedMs: number;
}

/** How many neighbours to ask the index for, given what the caller wants. */
export function candidateFetchCount(limit: number, budgeted: boolean): number {
  if (budgeted) return 400;
  return Math.min(400, Math.max(limit * 6, 12));
}

export interface RecallDeps {
  store: MemoryStore;
  index: VectorIndex;
}

export async function runRecall(deps: RecallDeps, params: RecallParams): Promise<RecallResult> {
  const limit = params.limit ?? 5;
  const budgeted = params.maxTokens !== undefined && params.maxTokens > 0;

  const t0 = Date.now();
  const queryVec = await embed(params.query);
  const embedMs = Date.now() - t0;

  const t1 = Date.now();
  const raw = deps.index.search(params.project, queryVec, candidateFetchCount(limit, budgeted));
  const searchMs = Date.now() - t1;

  const scoreById = new Map<number, number>();
  for (const h of raw) {
    if (params.minScore === undefined || h.score >= params.minScore) scoreById.set(h.id, h.score);
  }

  // The ANN index gave us ids + scores; SQLite resolves them to content. This
  // is a keyed lookup of a few dozen rows, not a scan.
  const memories = deps.store.getByIds([...scoreById.keys()], params.project);
  const candidates = memories
    .map((memory) => ({ memory, score: scoreById.get(memory.id) ?? 0 }))
    .sort((a, b) => b.score - a.score);

  const deduped = dedupeCandidates(candidates);

  let chosen = deduped;
  let tokensUsed = 0;
  let truncated = 0;
  if (budgeted) {
    const packed = packToBudget(deduped, params.maxTokens as number, (c) => hitCost(c.memory));
    chosen = packed.items;
    tokensUsed = packed.tokensUsed;
    truncated = packed.truncated;
  } else {
    chosen = deduped.slice(0, limit);
    tokensUsed = chosen.reduce((n, c) => n + hitCost(c.memory), 0);
  }

  return {
    hits: chosen,
    candidates: raw.length,
    deduped: deduped.length,
    tokensUsed,
    truncated,
    searchMs,
    embedMs,
  };
}
