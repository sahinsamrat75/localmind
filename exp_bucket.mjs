import { env, pipeline } from '@xenova/transformers';
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = process.cwd() + '/models';
const fx = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { quantized: true });

function cos(p, q) { let d = 0; for (let i = 0; i < 384; i++) d += p[i] * q[i]; return d; }

// Realistic mixed corpus of code-ish chunks with varying lengths
const texts = [];
for (let i = 0; i < 64; i++) {
  const n = 1 + (i % 30);
  const body = [];
  for (let j = 0; j < n; j++) body.push('  result[' + j + '] = transform(ctx, req, cfg) // ' + i);
  texts.push('func Fn' + i + '(w http.ResponseWriter, r *http.Request) {\n' + body.join('\n') + '\n}');
}
const singles = [];
for (const t of texts) singles.push((await fx(t, { pooling: 'mean', normalize: true })).data);

// Naive batching: original order (random lengths)
function batched(order) {
  const out = new Array(texts.length);
  const B = 16;
  for (let s = 0; s < order.length; s += B) {
    const idxs = order.slice(s, s + B);
    // results filled by caller
    out[s] = idxs;
  }
  return out;
}

async function runSorted(label, order) {
  const B = 16;
  let worst = 1, sum = 0, n = 0;
  for (let s = 0; s < order.length; s += B) {
    const idxs = order.slice(s, s + B);
    const out = await fx(idxs.map((i) => texts[i]), { pooling: 'mean', normalize: true, padding: true });
    for (let k = 0; k < idxs.length; k++) {
      const v = out.data.slice(k * 384, (k + 1) * 384);
      const c = cos(singles[idxs[k]], v);
      worst = Math.min(worst, c); sum += c; n++;
    }
  }
  console.log(label.padEnd(34), 'worst cos=' + worst.toFixed(6), 'mean cos=' + (sum / n).toFixed(6));
}

await runSorted('naive (random lengths)', [...texts.keys()]);
await runSorted('length-sorted buckets', [...texts.keys()].sort((a, b) => texts[a].length - texts[b].length));

// throughput of sorted vs naive
async function thr(label, order) {
  const B = 16;
  const t = Date.now();
  for (let r = 0; r < 2; r++)
    for (let s = 0; s < order.length; s += B) {
      await fx(order.slice(s, s + B).map((i) => texts[i]), { pooling: 'mean', normalize: true, padding: true });
    }
  const ms = Date.now() - t;
  console.log(label.padEnd(34), (128 * 1000 / ms).toFixed(1) + ' texts/sec');
}
await thr('naive throughput', [...texts.keys()]);
await thr('sorted throughput', [...texts.keys()].sort((a, b) => texts[a].length - texts[b].length));
