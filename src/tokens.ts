import type { Memory } from "./store.js";

/**
 * Token accounting and result deduplication for budgeted recall.
 *
 * The point of `max_tokens` is that an agent can say "give me back at most
 * this much context" and get the *most useful* slice that fits, instead of
 * dumping whole files. That needs three things: a token estimate, a dedup pass
 * (the same code frequently comes back two or three times as overlapping
 * chunks, which would eat the budget on near-duplicates), and a greedy packer.
 */

/**
 * Rough token estimate: ~4 characters per token.
 *
 * Deliberately an approximation — a real tokenizer would need to run BPE over
 * every candidate, and the budget only has to be a ceiling, not an exact
 * count. We bias slightly high (see the +1 word-boundary allowance below) so
 * callers can rely on the limit being respected rather than being exceeded.
 */
export function estimateTokens(text: string): number {
  if (text.length === 0) return 0;
  // Whitespace-heavy and symbol-heavy code tokenizes worse than 4 chars/token,
  // so add a small per-word allowance on top of the character estimate.
  const words = text.trim() === "" ? 0 : text.trim().split(/\s+/).length;
  return Math.ceil(text.length / 4 + words * 0.1);
}

/** A ranked search hit: the memory plus its similarity score. */
export interface Candidate {
  memory: Memory;
  score: number;
}

export interface DedupeOptions {
  /** Word-shingle Jaccard above which two chunks count as near-duplicates. */
  similarityThreshold?: number;
  /** Drop chunks whose line range covers an already-accepted range. */
  dropContainedRanges?: boolean;
}

/** Word 5-grams (3-grams for short texts) used for cheap near-dup detection. */
function shingles(text: string): Set<string> {
  const norm = text.replace(/\s+/g, " ").trim().toLowerCase();
  const words = norm.split(" ");
  const n = words.length >= 8 ? 5 : 3;
  const out = new Set<string>();
  if (words.length < n) {
    for (const w of words) out.add(w);
    return out;
  }
  for (let i = 0; i + n <= words.length; i++) out.add(words.slice(i, i + n).join(" "));
  return out;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  for (const s of small) if (large.has(s)) inter++;
  return inter / (a.size + b.size - inter);
}

/** Do two chunks cover overlapping lines of the same file? */
function overlapsLines(a: Memory, b: Memory): boolean {
  if (a.file_path === null || b.file_path === null) return false;
  if (a.file_path !== b.file_path) return false;
  if (a.line_start === null || a.line_end === null) return false;
  if (b.line_start === null || b.line_end === null) return false;
  return a.line_start <= b.line_end && b.line_start <= a.line_end;
}

/**
 * Remove redundant candidates, keeping the highest-scored member of each
 * redundancy group. Input is expected to be sorted best-first.
 *
 * Three kinds of redundancy are caught:
 *  1. byte-identical chunks (same content hash) — e.g. a helper duplicated
 *     across services, which is common in a microservice monorepo;
 *  2. overlapping line ranges in the same file — an oversized function split
 *     into pieces, or a class chunk and a method chunk inside it;
 *  3. near-identical text — the same function with a renamed variable.
 */
export function dedupeCandidates(cands: Candidate[], opts: DedupeOptions = {}): Candidate[] {
  const threshold = opts.similarityThreshold ?? 0.85;
  const keep: Candidate[] = [];
  const keptHashes = new Set<string>();
  const keptShingles: Array<{ set: Set<string>; len: number }> = [];

  for (const cand of cands) {
    const m = cand.memory;
    if (m.content_hash && keptHashes.has(m.content_hash)) continue;

    let duplicate = false;
    for (const kept of keep) {
      if (overlapsLines(m, kept.memory)) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;

    const sh = shingles(m.text);
    for (const other of keptShingles) {
      if (jaccard(sh, other.set) >= threshold) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;

    keep.push(cand);
    if (m.content_hash) keptHashes.add(m.content_hash);
    keptShingles.push({ set: sh, len: m.text.length });
  }
  return keep;
}

/** What a packed recall result consumed, for reporting back to the caller. */
export interface PackedResult<T> {
  items: T[];
  tokensUsed: number;
  /** Candidates dropped because the budget ran out. */
  truncated: number;
}

/**
 * Greedily take candidates in rank order while they fit the budget.
 *
 * Strictly stops at the first item that does not fit rather than skipping over
 * it to find a smaller one: rank order is the user's relevance order, and
 * silently preferring a short, less relevant chunk over a long, more relevant
 * one makes results unpredictable.
 */
export function packToBudget<T>(
  ranked: T[],
  maxTokens: number,
  cost: (item: T) => number
): PackedResult<T> {
  const items: T[] = [];
  let used = 0;
  let i = 0;
  for (; i < ranked.length; i++) {
    const c = cost(ranked[i]);
    if (used + c > maxTokens) break;
    used += c;
    items.push(ranked[i]);
  }
  return { items, tokensUsed: used, truncated: ranked.length - items.length };
}
