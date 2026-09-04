#!/usr/bin/env node
/**
 * localmind entry point.
 *
 * The server itself lives in ./server.ts; this module re-exports it and starts
 * the stdio transport when invoked directly (`node dist/index.js`), which is
 * what MCP clients such as Claude Code launch.
 */
export {
  VERSION,
  createLocalmindServer,
  runStdio,
  parseTags,
  normalizeProject,
  formatHit,
  hitCost,
} from "./server.js";
export type { Localmind, ServerOptions } from "./server.js";

import { runStdio } from "./server.js";

// Only start serving when this file is the process entry point, not when it is
// imported by the CLI or the tests.
const invokedDirectly =
  process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href;

if (invokedDirectly) {
  runStdio().catch((err) => {
    console.error("localmind failed to start:", err);
    process.exit(1);
  });
}

