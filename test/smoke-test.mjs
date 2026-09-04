#!/usr/bin/env node
/**
 * End-to-end smoke test for localmind.
 *
 * Spawns the real MCP server (dist/index.js) as a child process over stdio,
 * talks to it with the official MCP client, and verifies the full workflow:
 *   1. remember 3 distinct facts
 *   2. recall with a semantically-related query (NO keyword overlap) and
 *      assert the right memory ranks first
 *   3. forget one memory
 *   4. list_memories and confirm the deleted memory is gone
 *
 * Run with: npm test   (builds first, then runs this file)
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const serverPath = path.join(here, "..", "dist", "index.js");

let failures = 0;
function assert(cond, msg) {
  if (cond) {
    console.log(`  PASS  ${msg}`);
  } else {
    failures++;
    console.error(`  FAIL  ${msg}`);
  }
}

// Isolated data dir so the test never touches the user's real memory DB.
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "localmind-test-"));

const transport = new StdioClientTransport({
  command: process.execPath,
  args: [serverPath],
  env: { ...process.env, LOCALMIND_HOME: tmpDir },
});
const client = new Client({ name: "localmind-test", version: "0.1.0" });
await client.connect(transport);

async function callTool(name, args) {
  const res = await client.callTool({ name, arguments: args });
  if (res.isError) {
    throw new Error(`tool '${name}' returned error: ${JSON.stringify(res.content)}`);
  }
  return res.content.map((c) => c.text).join("\n");
}

try {
  // ---------------------------------------------------------------------
  // 1. remember three distinct facts
  // ---------------------------------------------------------------------
  console.log("\n[1] remember three distinct facts");
  const r1 = await callTool("remember", {
    text: "The quarterly budget review meeting was moved to the first Monday of every month.",
    tags: "work",
  });
  const r2 = await callTool("remember", {
    text: "Sofia is severely allergic to peanuts, so never order satay sauce when she joins dinner.",
    tags: "health, friends",
  });
  const r3 = await callTool("remember", {
    text: "The cottage we rented for the summer vacation has a sauna and a private lakeside dock.",
    tags: "travel",
  });
  const id1 = Number(r1.match(/id=(\d+)/)?.[1]);
  const id2 = Number(r2.match(/id=(\d+)/)?.[1]);
  const id3 = Number(r3.match(/id=(\d+)/)?.[1]);
  assert(Number.isInteger(id1) && Number.isInteger(id2) && Number.isInteger(id3),
    `all three memories stored with ids (${id1}, ${id2}, ${id3})`);
  assert(new Set([id1, id2, id3]).size === 3, "stored ids are distinct");

  // ---------------------------------------------------------------------
  // 2. semantic recall — query shares NO keywords with the target memory
  //    (target: the peanut allergy memory; no "allerg*", "peanut", "satay")
  // ---------------------------------------------------------------------
  console.log("\n[2] semantic recall (query has no keyword overlap with the target)");
  const recallOut = await callTool("recall", {
    query: "What groceries must I avoid buying when cooking for my friend?",
  });
  console.log("  recall output:\n" + recallOut.split("\n").map((l) => "    " + l).join("\n"));
  // v2 recall prefixes its output with a summary line starting '#'; the ranked
  // hits start on the next line.
  const recallLines = recallOut.split("\n").filter((l) => l.trim() && !l.startsWith("#"));
  const firstLine = recallLines[0] ?? "";
  assert(recallLines.length <= 5, `recall honours the default limit of 5 (${recallLines.length} hits)`);
  assert(firstLine.includes(`[${id2}]`), `top-ranked result is the allergy memory (id=${id2})`);
  const allergyScore = Number(firstLine.match(/score=([-\d.]+)/)?.[1]);
  assert(Number.isFinite(allergyScore) && allergyScore > 0.3, `allergy memory scored high (${allergyScore})`);


  // ---------------------------------------------------------------------
  // 3. forget one memory
  // ---------------------------------------------------------------------
  console.log("\n[3] forget one memory");
  const forgetOut = await callTool("forget", { id: id1 });
  assert(forgetOut.includes(`id=${id1}`) || forgetOut.toLowerCase().includes("deleted"),
    `forget confirmed deletion: ${forgetOut.trim()}`);

  const badForget = await client.callTool({ name: "forget", arguments: { id: 999999 } });
  assert(badForget.isError || /not found|no memory/i.test(badForget.content?.[0]?.text ?? ""),
    "forget with unknown id fails gracefully");

  // ---------------------------------------------------------------------
  // 4. list_memories — deleted memory must be gone; tag filter must work
  // ---------------------------------------------------------------------
  console.log("\n[4] list_memories after deletion");
  const listOut = await callTool("list_memories", {});
  console.log("  list output:\n" + listOut.split("\n").map((l) => "    " + l).join("\n"));
  assert(!listOut.includes(`[${id1}]`) && !listOut.includes("budget review"),
    `deleted memory (id=${id1}) no longer listed`);
  assert(listOut.includes(`[${id2}]`), "remaining memory id2 still listed");
  assert(listOut.includes(`[${id3}]`), "remaining memory id3 still listed");

  const tagOut = await callTool("list_memories", { tag: "health" });
  assert(tagOut.includes(`[${id2}]`) && !tagOut.includes(`[${id3}]`),
    "tag filter returns only memories tagged 'health'");

  const limitOut = await callTool("list_memories", { limit: 1 });
  const limitRows = limitOut.split("\n").filter((l) => /^\[\d+\]/.test(l));
  assert(limitRows.length === 1, `limit=1 returns exactly one memory (${limitRows.length})`);
} catch (err) {
  failures++;
  console.error("\nUNEXPECTED ERROR:", err);
} finally {
  await client.close();
}

console.log("\n==============================================");
if (failures === 0) {
  console.log("SMOKE TEST: ALL CHECKS PASSED ✅");
} else {
  console.error(`SMOKE TEST: ${failures} CHECK(S) FAILED ❌`);
  process.exit(1);
}
