import os from "node:os";
import path from "node:path";

export interface DataDir {
  dir: string;
  memoryDb: string;
  /** Directory holding the persisted per-project HNSW index files. */
  indexDir: string;
}

/**
 * Resolve the localmind data directory: $LOCALMIND_HOME or ~/.localmind.
 * The SQLite memory database lives at <dir>/memory.db and the HNSW vector
 * indexes at <dir>/indexes/<project>.hnsw.
 */
export function resolveDataDir(): DataDir {
  const dir = process.env.LOCALMIND_HOME
    ? path.resolve(process.env.LOCALMIND_HOME)
    : path.join(os.homedir(), ".localmind");
  return {
    dir,
    memoryDb: path.join(dir, "memory.db"),
    indexDir: path.join(dir, "indexes"),
  };
}

/**
 * Filesystem-safe encoding of a project identifier into a single path segment.
 * Projects are arbitrary user-supplied strings, so this must be injective
 * (different projects -> different files) and must never escape the index dir.
 */
export function projectSlug(project: string): string {
  return encodeURIComponent(project).replace(/[*]/g, "_");
}

/** Path of the persisted HNSW index for a project. */
export function indexPath(indexDir: string, project: string): string {
  return path.join(indexDir, `${projectSlug(project)}.hnsw`);
}

