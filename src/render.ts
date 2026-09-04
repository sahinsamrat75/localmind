import { KIND_CODE, type Memory } from "./store.js";
import { estimateTokens } from "./tokens.js";

/**
 * Presentation of a recall hit, shared by the MCP tool, the CLI and the
 * benchmark so token accounting and rendering never disagree.
 */

/**
 * Render one hit as a single line.
 *
 * Code chunks are labelled `path:start-end` so the caller can open the file and
 * read more around them; manual memories show their tags instead.
 */
export function formatHit(m: Memory, score: number): string {
  const meta = [`score=${score.toFixed(4)}`];
  if (m.file_path && m.line_start !== null && m.line_end !== null) {
    meta.push(`${m.file_path}:${m.line_start}-${m.line_end}`);
  }
  if (m.tags.length > 0 && m.kind !== KIND_CODE) meta.push(`tags=${m.tags.join(",")}`);
  return `[${m.id}] (${meta.join(", ")}) ${m.text}`;
}

/** The line a hit will occupy, without its body — used for budget maths. */
export function hitHeader(m: Memory): string {
  const loc =
    m.file_path && m.line_start !== null ? `${m.file_path}:${m.line_start}-${m.line_end}` : m.kind;
  return `[${m.id}] (${loc}) `;
}

/**
 * Token cost of including one hit in a budgeted response.
 *
 * Measured against the rendered line (header + body), so the budget covers the
 * file/line references too and not just the code.
 */
export function hitCost(m: Memory): number {
  return estimateTokens(hitHeader(m) + m.text);
}
