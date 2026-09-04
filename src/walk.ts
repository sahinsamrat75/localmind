import fs from "node:fs";
import path from "node:path";

/**
 * Recursive directory walk with .gitignore semantics.
 *
 * Implemented locally (no `ignore` dependency) so ingestion keeps the
 * zero-network, minimal-dependency property of the rest of the server. The
 * subset of gitignore supported covers what real repos actually use: comments,
 * negations, trailing-slash directory-only rules, anchored patterns (a leading
 * or interior `/`), `*` / `?` / `[...]` globs, and `**` across directories.
 * Nested `.gitignore` files apply to their own subtree and the last matching
 * rule wins — the same precedence git uses.
 */

export interface WalkOptions {
  /** Honour .gitignore files while walking (default: true). */
  respectGitignore?: boolean;
  /** Directory names skipped unconditionally at any depth. */
  skipDirs?: string[];
  /** Hard cap on returned files, so a mistyped path cannot run forever. */
  maxFiles?: number;
  /** Only return files whose relative path passes this predicate. */
  filter?: (relPath: string) => boolean;
  maxDepth?: number;
}

export interface WalkedFile {
  /** Absolute path, for reading. */
  absPath: string;
  /** Root-relative path with `/` separators, for storage and display. */
  relPath: string;
  size: number;
}

interface Rule {
  re: RegExp;
  negate: boolean;
  dirOnly: boolean;
}

const DEFAULT_SKIP_DIRS = [".git", ".hg", ".svn", "node_modules"];

/** Translate one gitignore glob into a RegExp source string. */
function translate(pattern: string): string {
  let out = "";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === "*") {
      if (pattern[i + 1] === "*") {
        if (pattern[i + 2] === "/") {
          out += "(?:.*/)?";
          i += 2;
        } else {
          out += ".*";
          i += 1;
        }
      } else {
        out += "[^/]*";
      }
    } else if (c === "?") {
      out += "[^/]";
    } else if (c === "[") {
      let j = i + 1;
      let cls = "";
      if (pattern[j] === "!" || pattern[j] === "^") {
        cls = "^";
        j++;
      }
      while (j < pattern.length && pattern[j] !== "]") {
        cls += /[\\^\]]/.test(pattern[j]) ? "\\" + pattern[j] : pattern[j];
        j++;
      }
      out += "[" + cls + "]";
      i = j;
    } else {
      out += /[.+^${}()|[\]\\/-]/.test(c) ? "\\" + c : c;
    }
  }
  return out;
}

/** Parse the contents of one .gitignore file into ordered rules. */
function parseGitignore(text: string, base: string): Rule[] {
  const rules: Rule[] = [];
  for (let raw of text.split(/\r?\n/)) {
    if (raw.trim().length === 0 || raw.trimStart().startsWith("#")) continue;
    raw = raw.replace(/\s+$/, "");
    let negate = false;
    if (raw.startsWith("!")) {
      negate = true;
      raw = raw.slice(1);
    }
    let dirOnly = false;
    if (raw.endsWith("/")) {
      dirOnly = true;
      raw = raw.slice(0, -1);
    }
    if (raw.startsWith("/")) raw = raw.slice(1);
    // A pattern containing a slash is anchored to the .gitignore's directory;
    // otherwise it may match at any depth below it.
    const anchored = raw.includes("/");
    const prefix = base ? base.replace(/\/$/, "") + "/" : "";
    const body = anchored ? prefix + translate(raw) : prefix + "(?:.*/)?" + translate(raw);
    rules.push({ re: new RegExp("^" + body + "(?:/.*)?$"), negate, dirOnly });
  }
  return rules;
}
/**
 * Decide whether a path is ignored, given the accumulated rule sets from the
 * root down to the file's directory. Deeper rule sets win, and within a file
 * the last matching line wins — matching git's precedence.
 */
function isIgnored(relPath: string, isDir: boolean, ruleSets: Rule[][]): boolean {
  let ignored = false;
  for (const rules of ruleSets) {
    for (const rule of rules) {
      if (!rule.re.test(relPath)) continue;
      if (rule.dirOnly && !isDir) {
        // A directory-only rule does not match the file itself; the traversal
        // prunes the excluded directory instead, which covers its contents.
        continue;
      }
      ignored = !rule.negate;
    }
  }
  return ignored;
}

/** True when the file looks binary (NUL byte in the first 8 KiB). */
export function looksBinary(absPath: string): boolean {
  let fd: number | null = null;
  try {
    fd = fs.openSync(absPath, "r");
    const buf = Buffer.alloc(8192);
    const bytes = fs.readSync(fd, buf, 0, buf.length, 0);
    for (let i = 0; i < bytes; i++) if (buf[i] === 0) return true;
    return false;
  } catch {
    return true;
  } finally {
    if (fd !== null) fs.closeSync(fd);
  }
}

/**
 * Walk `root` and return every non-ignored file.
 *
 * Directories excluded by an ignore rule are pruned rather than descended into
 * and re-filtered, which is both what git does and dramatically faster on a
 * real repo — a single `node_modules/` hit skips tens of thousands of files.
 */
export function walkDirectory(root: string, options: WalkOptions = {}): WalkedFile[] {
  const respectGitignore = options.respectGitignore ?? true;
  const skipDirs = new Set(options.skipDirs ?? DEFAULT_SKIP_DIRS);
  const maxFiles = options.maxFiles ?? Number.POSITIVE_INFINITY;
  const maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY;
  const rootAbs = path.resolve(root);

  const out: WalkedFile[] = [];

  function recurse(absDir: string, relDir: string, ruleSets: Rule[][], depth: number): void {
    if (depth > maxDepth || out.length >= maxFiles) return;

    let currentSets = ruleSets;
    if (respectGitignore) {
      const gi = path.join(absDir, ".gitignore");
      try {
        if (fs.statSync(gi).isFile()) {
          currentSets = [...ruleSets, parseGitignore(fs.readFileSync(gi, "utf8"), relDir)];
        }
      } catch {
        /* no .gitignore in this directory */
      }
    }

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(absDir, { withFileTypes: true });
    } catch {
      return; // unreadable directory
    }

    for (const entry of entries) {
      if (out.length >= maxFiles) return;
      const rel = relDir ? `${relDir}/${entry.name}` : entry.name;
      const abs = path.join(absDir, entry.name);

      if (entry.isDirectory()) {
        if (skipDirs.has(entry.name)) continue;
        if (respectGitignore && isIgnored(rel, true, currentSets)) continue;
        recurse(abs, rel, currentSets, depth + 1);
        continue;
      }
      if (!entry.isFile()) continue; // skip symlinks, sockets, devices

      if (respectGitignore && isIgnored(rel, false, currentSets)) continue;
      if (options.filter && !options.filter(rel)) continue;

      let size = 0;
      try {
        size = fs.statSync(abs).size;
      } catch {
        continue;
      }
      out.push({ absPath: abs, relPath: rel, size });
    }
  }

  recurse(rootAbs, "", [], 0);
  return out;
}

