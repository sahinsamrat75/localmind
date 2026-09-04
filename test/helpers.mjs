/**
 * Shared helpers for the localmind test suite.
 *
 * Every test drives the *real* server: a child process running dist/index.js
 * over stdio, spoken to with the official MCP client. That keeps the tests
 * honest about what an agent actually sees, including the JSON-schema layer.
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const SERVER_PATH = path.join(here, "..", "dist", "index.js");

let failures = 0;
let checks = 0;

export function assert(cond, msg) {
  checks++;
  if (cond) {
    console.log(`  PASS  ${msg}`);
  } else {
    failures++;
    console.error(`  FAIL  ${msg}`);
  }
}

export function section(title) {
  console.log(`\n${title}`);
}

/** Create an isolated data directory so tests never touch ~/.localmind. */
export function makeTempDir(prefix = "localmind-test-") {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

/**
 * Spawn a server rooted at `home` and return a client plus a callTool helper
 * that surfaces tool errors as exceptions and flattens text content.
 */
export async function startServer(home) {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [SERVER_PATH],
    env: { ...process.env, LOCALMIND_HOME: home },
  });
  const client = new Client({ name: "localmind-test", version: "0.0.0" });
  await client.connect(transport);

  async function callTool(name, args = {}) {
    const res = await client.callTool({ name, arguments: args });
    if (res.isError) {
      throw new Error(`tool '${name}' errored: ${JSON.stringify(res.content)}`);
    }
    return (res.content ?? []).map((c) => c.text ?? "").join("\n");
  }

  async function callToolRaw(name, args = {}) {
    return client.callTool({ name, arguments: args });
  }

  async function listTools() {
    const res = await client.listTools();
    return res.tools.map((t) => t.name);
  }

  return { client, callTool, callToolRaw, listTools, home };
}

/** Print a summary and exit non-zero on failure. */
export function finish(label) {
  console.log("\n==============================================");
  if (failures === 0) {
    console.log(`${label}: ALL ${checks} CHECKS PASSED`);
    return 0;
  }
  console.error(`${label}: ${failures} of ${checks} CHECK(S) FAILED`);
  return 1;
}

export { fs, os, path };
