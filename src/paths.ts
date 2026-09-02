import os from "node:os";
import path from "node:path";

export interface DataDir {
  dir: string;
  memoryDb: string;
}

/**
 * Resolve the localmind data directory: $LOCALMIND_HOME or ~/.localmind.
 * The SQLite memory database lives at <dir>/memory.db.
 */
export function resolveDataDir(): DataDir {
  const dir = process.env.LOCALMIND_HOME
    ? path.resolve(process.env.LOCALMIND_HOME)
    : path.join(os.homedir(), ".localmind");
  return { dir, memoryDb: path.join(dir, "memory.db") };
}
