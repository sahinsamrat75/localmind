#!/usr/bin/env node
/**
 * Project namespacing isolation test.
 *
 * Proves the core isolation property: a recall (or list/forget) scoped to one
 * project can never observe another project's data — even when the two projects
 * contain semantically identical text, which is exactly the case a
 * filter-based implementation would get wrong.
 */
import { startServer, makeTempDir, assert, section, finish, fs, path } from "./helpers.mjs";

const home = makeTempDir("localmind-isolation-");
const { callTool, listTools, client } = await startServer(home);

const ALPHA = "pixora-payments";
const BETA = "pixora-search";

let alphaId, betaId, sharedId;

try {
  section("[1] tools are registered");
  const tools = await listTools();
  for (const t of ["remember", "recall", "forget", "list_memories", "ingest_codebase"]) {
    assert(tools.includes(t), `tool '${t}' exposed over MCP`);
  }

  section("[2] remember writes into the named project");
  const note = "We chose Kafka for event streaming and Redis for idempotency keys in the payments service.";
  const a = await callTool("remember", { text: note, tags: "architecture", project: ALPHA });
  const b = await callTool("remember", { text: note, tags: "architecture", project: BETA });
  alphaId = Number(a.match(/id=(\d+)/)?.[1]);
  betaId = Number(b.match(/id=(\d+)/)?.[1]);
  assert(Number.isInteger(alphaId) && Number.isInteger(betaId), `stored ids ${alphaId} and ${betaId}`);
  assert(alphaId !== betaId, "the two namespaces got distinct ids");
  assert(a.includes(`project=${ALPHA}`), "remember echoes the project it wrote to");

  const shared = await callTool("remember", {
    text: "The nightly index rebuild is scheduled for 02:00 UTC on the search cluster.",
    project: ALPHA,
  });
  sharedId = Number(shared.match(/id=(\d+)/)?.[1]);

  section("[3] recall in one project never returns the other's memories");
  const query = "why did we pick Kafka and Redis?";
  const inAlpha = await callTool("recall", { query, project: ALPHA, limit: 10 });
  const inBeta = await callTool("recall", { query, project: BETA, limit: 10 });

  assert(inAlpha.includes(`[${alphaId}]`), `recall(${ALPHA}) finds ${alphaId}`);
  assert(!inAlpha.includes(`[${betaId}]`), `recall(${ALPHA}) does NOT leak ${betaId} from ${BETA}`);
  assert(inBeta.includes(`[${betaId}]`), `recall(${BETA}) finds ${betaId}`);
  assert(!inBeta.includes(`[${alphaId}]`), `recall(${BETA}) does NOT leak ${alphaId} from ${ALPHA}`);
  assert(inAlpha.includes(`project '${ALPHA}'`), "recall output names the project it searched");

  section("[4] unrelated queries do not cross namespaces");
  const searchQ = await callTool("recall", { query: "when does the index rebuild run?", project: BETA, limit: 10 });
  assert(!searchQ.includes(`[${sharedId}]`), `that memory lives only in ${ALPHA} and stayed there`);
  const searchA = await callTool("recall", { query: "when does the index rebuild run?", project: ALPHA, limit: 10 });
  assert(searchA.includes(`[${sharedId}]`), `and is visible from ${ALPHA}`);


  section("[5] list_memories is project-scoped");
  const listAlpha = await callTool("list_memories", { project: ALPHA });
  const listBeta = await callTool("list_memories", { project: BETA });
  assert(listAlpha.includes(`[${alphaId}]`) && !listAlpha.includes(`[${betaId}]`),
    `list(${ALPHA}) shows only ${ALPHA} ids`);
  assert(listBeta.includes(`[${betaId}]`) && !listBeta.includes(`[${alphaId}]`),
    `list(${BETA}) shows only ${BETA} ids`);

  section("[6] forget cannot reach across namespaces");
  const crossDelete = await callTool("forget", { id: betaId, project: ALPHA });
  assert(/No memory found/i.test(crossDelete),
    `forget(${betaId}) scoped to ${ALPHA} refuses: ${crossDelete.trim()}`);
  const stillThere = await callTool("list_memories", { project: BETA });
  assert(stillThere.includes(`[${betaId}]`), `${betaId} survives the cross-namespace forget attempt`);

  const ownDelete = await callTool("forget", { id: betaId, project: BETA });
  assert(/Deleted/.test(ownDelete), `forget(${betaId}) in ${BETA} succeeds`);
  const afterDelete = await callTool("recall", { query, project: BETA, limit: 10 });
  assert(!afterDelete.includes(`[${betaId}]`), "deleted memory is gone from recall in its own project");
  const alphaAfter = await callTool("recall", { query, project: ALPHA, limit: 10 });
  assert(alphaAfter.includes(`[${alphaId}]`), "the other project is untouched by that delete");

  section("[7] omitted project means the 'default' namespace");
  const dflt = await callTool("remember", { text: "I prefer dark mode in every editor." });
  const dId = Number(dflt.match(/id=(\d+)/)?.[1]);
  assert(dflt.includes("project=default"), "unscoped remember lands in 'default'");
  const dRecall = await callTool("recall", { query: "what editor theme do I like?" });
  assert(dRecall.includes(`[${dId}]`), "unscoped recall sees the default namespace");
  assert(!dRecall.includes(`[${alphaId}]`), "unscoped recall does NOT see named projects");
  const alphaRecall = await callTool("recall", { query: "what editor theme do I like?", project: ALPHA });
  assert(!alphaRecall.includes(`[${dId}]`), "a named project does not see 'default' either");

  section("[8] unknown project returns nothing rather than everything");
  const empty = await callTool("recall", { query, project: "some-project-that-does-not-exist" });
  assert(!empty.includes(`[${alphaId}]`) && !empty.includes(`[${betaId}]`),
    "a recall in an empty project leaks nothing");
  assert(/No memories/i.test(empty), `and says so: ${empty.trim()}`);

  section("[9] index files are physically separate per project");
  const indexFiles = fs.readdirSync(path.join(home, "indexes")).filter((f) => f.endsWith(".hnsw"));
  assert(indexFiles.length >= 3, `one index file per project on disk (${indexFiles.length} found)`);
  assert(new Set(indexFiles).size === indexFiles.length, "index file names are distinct");
} catch (err) {
  console.error("\nUNEXPECTED ERROR:", err);
  process.exitCode = 1;
} finally {
  await client.close();
}

process.exitCode = finish("ISOLATION TEST") || process.exitCode;
