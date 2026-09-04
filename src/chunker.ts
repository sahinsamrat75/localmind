/**
 * Source-code chunking for `ingest_codebase`.
 *
 * The goal is that each stored chunk is a *retrievable unit of meaning* — one
 * function, one class, one top-level declaration — rather than an arbitrary
 * slice of lines. That is what makes token-budgeted recall useful: an agent
 * asks a question and gets back exactly the handful of functions it needed,
 * with file + line ranges so it can pull more if it wants.
 *
 * Boundaries are found with a small line-oriented scanner rather than a full
 * parser: we blank out string literals and comments (so braces inside them do
 * not confuse us) and then track bracket depth to find where a top-level
 * declaration ends. Python is indentation-based instead. Anything we cannot
 * parse falls back to a line window with overlap, so no file type is ever
 * skipped and no content is ever lost.
 */

export interface Chunk {
  /** Exact source text of the chunk. */
  content: string;
  /** 1-based inclusive line range in the source file. */
  startLine: number;
  endLine: number;
  /** Name of the function/class/declaration, or '' when unknown. */
  unit: string;
  /** 'func' | 'method' | 'class' | 'interface' | 'block' | 'lines' | ... */
  unitKind: string;
  lang: string;
}

export interface ChunkOptions {
  /** A single logical unit longer than this is split into overlapping pieces. */
  maxChunkLines?: number;
  maxChunkChars?: number;
  /** Lines repeated between the pieces of a split oversized unit. */
  splitOverlapLines?: number;
  /** Window size for the line-based fallback chunker. */
  fallbackLines?: number;
  fallbackOverlapLines?: number;
  /** Units shorter than this get merged with the previous unit. */
  minChunkLines?: number;
}

export const DEFAULT_CHUNK_OPTIONS = {
  maxChunkLines: 160,
  maxChunkChars: 6000,
  splitOverlapLines: 20,
  fallbackLines: 60,
  fallbackOverlapLines: 12,
  minChunkLines: 4,
};

const EXT_LANGS: Record<string, string> = {
  go: "go",
  js: "js",
  jsx: "js",
  mjs: "js",
  cjs: "js",
  ts: "ts",
  tsx: "ts",
  py: "python",
  pyi: "python",
};

/** Map a file path onto a chunking strategy name. */
export function detectLanguage(filePath: string): string {
  const ext = filePath.slice(filePath.lastIndexOf(".") + 1).toLowerCase();
  return EXT_LANGS[ext] ?? "text";
}

/** A raw declaration span found by a language scanner. */
interface Span {
  start: number; // 0-based line index
  end: number; // 0-based, inclusive
  unit: string;
  kind: string;
}

/**
 * Blank out string literals and comments in a line so that bracket counting
 * only sees real code. `state` carries block-comment / multi-line-string
 * openness across lines.
 */
interface ScanState {
  blockComment: boolean;
  template: boolean; // JS backtick / Go raw string spanning lines
}

function stripLine(line: string, st: ScanState, opts: {slashComments: boolean; hashComments: boolean; blockComments: boolean; templates: boolean}): string {
  let out = "";
  let i = 0;
  while (i < line.length) {
    const ch = line[i];
    const next = line[i + 1];

    if (st.blockComment) {
      if (opts.blockComments && ch === "*" && next === "/") {
        st.blockComment = false;
        out += "  ";
        i += 2;
        continue;
      }
      out += " ";
      i++;
      continue;
    }
    if (st.template) {
      if (ch === "\\") {
        out += "  ";
        i += 2;
        continue;
      }
      if (ch === "`") {
        st.template = false;
        out += "  ";
        i++;
        continue;
      }
      out += " ";
      i++;
      continue;
    }

    if (opts.blockComments && ch === "/" && next === "*") {
      st.blockComment = true;
      out += "  ";
      i += 2;
      continue;
    }
    if (opts.slashComments && ch === "/" && next === "/") {
      break; // rest of the line is a comment
    }
    if (opts.hashComments && ch === "#") {
      break;
    }
    if (ch === '"' || ch === "'" || (opts.templates && ch === "`")) {
      const quote = ch;
      if (quote === "`") {
        st.template = true;
        out += " ";
        i++;
        continue;
      }
      // Single-quoted / double-quoted: consume to the closing quote on this
      // line (unterminated strings are treated as ending at the newline).
      let j = i + 1;
      while (j < line.length) {
        if (line[j] === "\\") {
          j += 2;
          continue;
        }
        if (line[j] === quote) {
          j++;
          break;
        }
        j++;
      }
      out += " ".repeat(j - i);
      i = j;
      continue;
    }
    out += ch;
    i++;
  }
  return out;
}

/** Net bracket depth change of a stripped line, and whether it opens a block. */
function depthDelta(code: string): number {
  let d = 0;
  for (const ch of code) {
    if (ch === "{" || ch === "(" || ch === "[") d++;
    else if (ch === "}" || ch === ")" || ch === "]") d--;
  }
  return d;
}

function isCommentLine(line: string, hash: boolean): boolean {
  const t = line.trim();
  if (t.length === 0) return false;
  if (hash) return t.startsWith("#");
  return t.startsWith("//") || t.startsWith("/*") || t.startsWith("*");
}

/** Walk up from `index` absorbing contiguous comment/blank lines above it. */
function absorbLeadingComment(lines: string[], index: number, hash: boolean): number {
  let j = index - 1;
  let sawComment = false;
  while (j >= 0) {
    const t = lines[j].trim();
    if (t.length === 0) {
      // A blank line directly above the declaration is part of its preamble
      // only if comments follow immediately.
      if (sawComment) {
        j--;
        continue;
      }
      break;
    }
    if (isCommentLine(lines[j], hash)) {
      sawComment = true;
      j--;
      continue;
    }
    break;
  }
  return sawComment ? j + 1 : index;
}

/* ------------------------------------------------------------------ */
/* Brace-based languages (Go, JS, TS)                                  */
/* ------------------------------------------------------------------ */

interface DeclRule {
  re: RegExp;
  kind: string;
  name: number; // capture group holding the identifier
}

const GO_DECLS: DeclRule[] = [
  { re: /^func\s+\([^)]*\)\s*([A-Za-z_]\w*)/, kind: "method", name: 1 },
  { re: /^func\s+([A-Za-z_]\w*)/, kind: "func", name: 1 },
  { re: /^type\s+([A-Za-z_]\w*)\s+struct\b/, kind: "struct", name: 1 },
  { re: /^type\s+([A-Za-z_]\w*)\s+interface\b/, kind: "interface", name: 1 },
  { re: /^type\s+([A-Za-z_]\w*)/, kind: "type", name: 1 },
  { re: /^var\s*\(/, kind: "varblock", name: 0 },
  { re: /^const\s*\(/, kind: "constblock", name: 0 },
  { re: /^var\s+([A-Za-z_]\w*)/, kind: "var", name: 1 },
  { re: /^const\s+([A-Za-z_]\w*)/, kind: "const", name: 1 },
];

const TS_DECLS: DeclRule[] = [
  { re: /^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)/, kind: "function", name: 1 },
  { re: /^(?:export\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)/, kind: "class", name: 1 },
  { re: /^(?:export\s+)?(?:interface\s+)([A-Za-z_$][\w$]*)/, kind: "interface", name: 1 },
  { re: /^(?:export\s+)?(?:type\s+)([A-Za-z_$][\w$]*)\s*[=<]/, kind: "type", name: 1 },
  { re: /^(?:export\s+)?(?:enum\s+)([A-Za-z_$][\w$]*)/, kind: "enum", name: 1 },
  { re: /^(?:export\s+)?(?:namespace|module)\s+([A-Za-z_$][\w$]*)/, kind: "namespace", name: 1 },
  // Arrow-function / function-valued bindings are the JS equivalent of a def.
  {
    re: /^(?:export\s+)?(?:declare\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\b[^=]*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)/,
    kind: "function",
    name: 1,
  },
  { re: /^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\b/, kind: "binding", name: 1 },
];

/**
 * Find top-level declaration spans in a brace-delimited language.
 *
 * A unit starts on a line at column 0 that matches a declaration rule while we
 * are at bracket depth 0, and ends once the brackets it opened have all closed
 * (and the statement is not continuing onto further lines).
 */
function scanBraceLang(lines: string[], rules: DeclRule[]): Span[] {
  const state: ScanState = { blockComment: false, template: false };
  const stripped = lines.map((l) => stripLine(l, state, {
    slashComments: true,
    hashComments: false,
    blockComments: true,
    templates: true,
  }));

  const spans: Span[] = [];
  let depth = 0;
  let i = 0;

  while (i < lines.length) {
    // Only consider declarations that open at the outermost level, at column
    // zero, and while we are not inside a multi-line construct.
    if (depth === 0 && lines[i].length > 0 && !/^\s/.test(lines[i])) {
      const line = stripped[i];
      let matched = false;
      for (const rule of rules) {
        const m = rule.re.exec(line);
        if (!m) continue;
        let start = absorbLeadingComment(lines, i, false);
        // Never let a comment preamble swallow the previous unit.
        if (spans.length > 0) start = Math.max(start, spans[spans.length - 1].end + 1);
        const unit = rule.name > 0 && m[rule.name] ? m[rule.name] : "";
        let end = i;
        let d = depthDelta(line);
        // Consume until every bracket this declaration opened has closed.
        while (d > 0 && end + 1 < lines.length) {
          end++;
          d += depthDelta(stripped[end]);
        }
        if (d < 0) d = 0;
        depth = d;
        spans.push({ start, end, unit, kind: rule.kind });
        i = end + 1;
        matched = true;
        break;
      }
      if (matched) continue;
    }
    depth += depthDelta(stripped[i]);
    if (depth < 0) depth = 0;
    i++;
  }
  return spans;
}

/* ------------------------------------------------------------------ */
/* Python (indentation-based)                                          */
/* ------------------------------------------------------------------ */

const PY_TOP_LEVEL = /^(?:async\s+def|def|class)\s+([A-Za-z_]\w*)/;

/**
 * Python blocks run from a column-0 `def` / `class` / decorator until the next
 * column-0 statement at bracket depth 0. Nested defs and methods stay inside
 * their enclosing block, which is what we want: a class is one chunk.
 */
function scanPython(lines: string[]): Span[] {
  const state: ScanState = { blockComment: false, template: false };
  const stripped = lines.map((l) => stripLine(l, state, {
    slashComments: false,
    hashComments: true,
    blockComments: false,
    templates: false,
  }));

  const starts: number[] = [];
  let depth = 0;
  for (let i = 0; i < lines.length; i++) {
    const atCol0 = !/^\s/.test(lines[i]) && lines[i].length > 0;
    if (depth === 0 && atCol0 && PY_TOP_LEVEL.test(stripped[i])) {
      starts.push(i);
    }
    depth += depthDelta(stripped[i]);
    if (depth < 0) depth = 0;
  }

  const spans: Span[] = [];
  for (let s = 0; s < starts.length; s++) {
    let start = starts[s];
    // Decorators belong to the def/class below them.
    while (start > 0 && lines[start - 1].startsWith("@")) {
      start--;
    }
    start = absorbLeadingComment(lines, start, true);
    const nextStart = s + 1 < starts.length ? starts[s + 1] : lines.length;
    // The block runs to the line before the next top-level statement, minus
    // trailing blank lines that belong to the gap between blocks.
    let end = nextStart - 1;
    while (end > start && lines[end].trim().length === 0) end--;
    const header = stripped[starts[s]].match(PY_TOP_LEVEL);
    const kind = /^class\b/.test(stripped[starts[s]])
      ? "class"
      : /^(?:async\s+)?def\b/.test(stripped[starts[s]])
        ? "function"
        : "decorated";
    spans.push({ start, end, unit: header?.[1] ?? "", kind });
  }
  return spans;
}
/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

/** Units that are meaningful on their own and must never be folded away. */
const MAJOR_KINDS = new Set([
  "func",
  "method",
  "function",
  "class",
  "struct",
  "interface",
  "enum",
  "namespace",
  "type",
]);

/** Turn declaration spans into chunks, covering the gaps in between. */
function fillGaps(spans: Span[], lineCount: number): Span[] {
  const out: Span[] = [];
  let cursor = 0;
  for (const span of spans) {
    if (span.start > cursor) {
      out.push({ start: cursor, end: span.start - 1, unit: "", kind: "preamble" });
    }
    out.push(span);
    cursor = Math.max(cursor, span.end + 1);
  }
  if (cursor < lineCount) {
    out.push({ start: cursor, end: lineCount - 1, unit: "", kind: "trailing" });
  }
  return out;
}

/**
 * Merge runs of very small units together.
 *
 * A file of twenty one-line `const` exports should not become twenty chunks:
 * each is a poor standalone retrieval target and they flood the candidate
 * pool. Adjacent small units fold into their neighbour, while real functions
 * and classes are never merged away.
 */
/**
 * Merge runs of very small units together.
 *
 * A file of twenty one-line `const` exports should not become twenty chunks:
 * each is a poor standalone retrieval target and they flood the candidate
 * pool. Small non-major units fold into their neighbour, but a run never
 * swallows a major span (function/class) that follows it — a "block" that has
 * grown past minLines because it absorbed neighbours stops absorbing, so the
 * next real declaration keeps its own chunk.
 */
function mergeSmall(spans: Span[], minLines: number): Span[] {
  const out: Span[] = [];
  for (const span of spans) {
    const size = span.end - span.start + 1;
    const prev = out[out.length - 1];
    const major = MAJOR_KINDS.has(span.kind);
    const prevMajor = prev !== undefined && MAJOR_KINDS.has(prev.kind);

    if (prev === undefined) {
      out.push({ ...span });
      continue;
    }
    if (major) {
      // A function/class/struct always keeps its own chunk.
      out.push({ ...span });
      continue;
    }
    // Non-major span: fold into the previous chunk only when the previous chunk
    // is itself non-major (never mutate a real declaration) and one of the two
    // sides is under the minimum size.
    const prevSize = prev.end - prev.start + 1;
    if (!prevMajor && (size < minLines || prevSize < minLines)) {
      prev.end = span.end;
      prev.unit = "";
      prev.kind = "block";
      continue;
    }
    out.push({ ...span });
  }
  return out;
}

/** Line-based chunking with overlap — the universal fallback. */
function chunkByLines(lines: string[], lang: string, opts: Required<ChunkOptions>): Chunk[] {
  const step = Math.max(1, opts.fallbackLines - opts.fallbackOverlapLines);
  const chunks: Chunk[] = [];
  for (let start = 0; start < lines.length; start += step) {
    let end = Math.min(lines.length - 1, start + opts.fallbackLines - 1);
    // Prefer to break on a blank line near the end of the window.
    for (let probe = end; probe > end - 8 && probe > start; probe--) {
      if (lines[probe].trim().length === 0) {
        end = probe - 1;
        break;
      }
    }
    const body = lines.slice(start, end + 1);
    if (body.join("").trim().length > 0) {
      chunks.push({
        content: body.join("\n"),
        startLine: start + 1,
        endLine: end + 1,
        unit: "",
        unitKind: "lines",
        lang,
      });
    }
    if (end >= lines.length - 1) break;
  }
  return chunks;
}

/**
 * Chunk one source file into retrievable units.
 *
 * Guarantees: every non-blank line of the file appears in at least one chunk,
 * and boundaries follow function/class declarations for Go, JS, TS and Python.
 * Unsupported file types fall back to an overlapping line window, so no file
 * type is ever skipped and no content is ever lost.
 */
export function chunkFile(filePath: string, source: string, options: ChunkOptions = {}): Chunk[] {
  const opts = { ...DEFAULT_CHUNK_OPTIONS, ...options } as Required<ChunkOptions>;
  const lang = detectLanguage(filePath);
  const normalized = source.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  let lines = normalized.split("\n");
  // Drop the empty line produced by a trailing newline.
  if (lines.length > 1 && lines[lines.length - 1].length === 0) lines = lines.slice(0, -1);
  if (lines.length === 0 || lines.every((l) => l.trim().length === 0)) return [];

  let spans: Span[];
  if (lang === "go") spans = scanBraceLang(lines, GO_DECLS);
  else if (lang === "ts" || lang === "js") spans = scanBraceLang(lines, TS_DECLS);
  else if (lang === "python") spans = scanPython(lines);
  else return chunkByLines(lines, lang, opts);

  if (spans.length === 0) return chunkByLines(lines, lang, opts);

  spans = mergeSmall(fillGaps(spans, lines.length), opts.minChunkLines);

  const chunks: Chunk[] = [];
  for (const span of spans) {
    const body = lines.slice(span.start, span.end + 1);
    if (body.join("").trim().length === 0) continue;
    const size = span.end - span.start + 1;
    const chars = body.reduce((n, l) => n + l.length + 1, 0);

    if (size > opts.maxChunkLines || chars > opts.maxChunkChars) {
      // Oversized unit: emit overlapping pieces so the whole body is covered
      // and each piece still carries the enclosing unit's name.
      const step = Math.max(1, opts.maxChunkLines - opts.splitOverlapLines);
      for (let s = span.start; s <= span.end; s += step) {
        const e = Math.min(span.end, s + opts.maxChunkLines - 1);
        const part = lines.slice(s, e + 1);
        if (part.join("").trim().length > 0) {
          chunks.push({
            content: part.join("\n"),
            startLine: s + 1,
            endLine: e + 1,
            unit: span.unit,
            unitKind: span.kind,
            lang,
          });
        }
        if (e === span.end) break;
      }
      continue;
    }

    chunks.push({
      content: body.join("\n"),
      startLine: span.start + 1,
      endLine: span.end + 1,
      unit: span.unit,
      unitKind: span.kind,
      lang,
    });
  }
  return chunks;
}



