import { env, pipeline } from '@xenova/transformers';
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = process.cwd() + '/models';
const fx = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { quantized: true });

const a = 'The payments service uses Kafka for event streaming and Redis for idempotency keys.';
const b = 'x';
const c = 'a rather long sentence that is going to pad the batch out quite a lot more than the short ones do, yes indeed it is quite long.';

function cos(p, q) { let d = 0; for (let i = 0; i < 384; i++) d += p[i] * q[i]; return d; }

const single = await fx(a, { pooling: 'mean', normalize: true });
for (const grp of [[a], [a, a, a], [b, a, c], [c, a, b]]) {
  const out = await fx(grp, { pooling: 'mean', normalize: true, padding: true });
  const idx = grp.indexOf(a);
  const v = out.data.slice(idx * 384, (idx + 1) * 384);
  console.log('batch len ' + grp.length + ' -> cos(single, batched) =', cos(single.data, v).toFixed(6));
}
const o1 = await fx([a, a], { pooling: 'mean', normalize: true, padding: true });
console.log('identical texts same batch cos =', cos(o1.data.slice(0, 384), o1.data.slice(384)).toFixed(6));
