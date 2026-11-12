// Backtest harness: builds synthetic contest days from hourly pump.fun pool candles (collected by tools/gt_fetch.mjs into
// <data>/pools.json and <data>/ohlcv/<pool>.json), runs an entry on each day's history and scores it with the contest's
// log score (order + size, floored at 1e-4) at each of its answer reads.
//
//   node tools/backtest.mjs <data dir> <work dir> [entry.ts ...]   (MAKEFUN_PARAMS passes through to the entry)
//
// A day: t0 on the hour; marks hourly from t0+1h to t0+CUT h (the cap at a mark is the close of the hour before it, on
// 10^9 tokens); answer reads hourly from t0+18h to t0+24h, each chart the 100 largest coins created at or after t0.
// Only graduated coins (pools) are known, so the charts miss the curve coins, which never reach the top places.
import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';

const DATA = process.argv[2];
const WORK = process.argv[3];
const ENTRIES = process.argv.slice(4);
const CUT = Number(process.env.BT_CUT ?? 11);
const T0S = (process.env.BT_T0S ?? '2026-11-10T22:00:00Z,2026-11-11T00:00:00Z,2026-11-11T02:00:00Z,2026-11-11T04:00:00Z').split(',');
const HOUR = 3_600_000;

const pools = JSON.parse(readFileSync(join(DATA, 'pools.json'), 'utf8'));
const coins = [];
for (const [addr, p] of Object.entries(pools)) {
  if (!p.name.endsWith('/ SOL')) continue; // pump coins quoted in SOL only (other quotes are copycat pools)
  const f = join(DATA, 'ohlcv', addr + '.json');
  if (!existsSync(f)) continue;
  const list = JSON.parse(readFileSync(f, 'utf8'));
  if (!Array.isArray(list) || list.length === 0) continue;
  const close = new Map();
  for (const c of list) close.set(c[0] * 1000, +c[4]);
  // copycat pools with implausible caps (10^5.6 SOL and up) are left out
  if ([...close.values()].some((px) => Math.log10(px * 1e9) > Number(process.env.BT_MAXCAP ?? 5.6))) continue;
  coins.push({ mint: p.mint, name: p.name, created: Date.parse(p.created), close });
}
/** log10 cap in SOL at instant t: the close of the hourly candle that ended at t (or the latest before it, up to 3 h). */
function capAt(c, t) {
  if (c.created > t) return null;
  for (let back = 1; back <= 3; back++) {
    const px = c.close.get(t - back * HOUR);
    if (px !== undefined && px > 0) return Math.log10(px * 1e9);
  }
  // created within the last hour: no closed candle yet
  return null;
}

const EDGES = Array.from({ length: 81 }, (_, i) => Math.round((2 + i * 0.05) * 100) / 100);
function binOf(x) {
  let lo = 0;
  let hi = EDGES.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (EDGES[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function score(fc, charts) {
  const named = Array.from({ length: 5 }, () => new Map());
  const sums = [0, 0, 0, 0, 0];
  for (const r of fc.probabilities) {
    named[r.place - 1].set(r.mint, r.p);
    sums[r.place - 1] += r.p;
  }
  const sz = Array.from({ length: 5 }, () => new Map());
  const ssum = [0, 0, 0, 0, 0];
  for (const r of fc.sizes) {
    sz[r.place - 1].set(r.bin, r.p);
    ssum[r.place - 1] += r.p;
  }
  let order = 0;
  let size = 0;
  let unnamedHits = 0;
  for (const chart of charts) {
    for (let k = 0; k < Math.min(5, chart.length); k++) {
      const at = named[k];
      let p = at.get(chart[k].mint);
      if (p === undefined) {
        let un = 0;
        for (const c of chart) if (!at.has(c.mint)) un++;
        p = un === 0 ? 0 : Math.max(0, 1 - sums[k]) / un;
        unnamedHits++;
      }
      order += Math.log(Math.max(p, 1e-4));
      const b = binOf(chart[k].x);
      let q = sz[k].get(b);
      if (q === undefined) q = Math.max(0, 1 - ssum[k]) / (82 - sz[k].size);
      size += Math.log(Math.max(q, 1e-4));
    }
  }
  return { order: order / charts.length, size: size / charts.length, total: (order + size) / charts.length, unnamedHits };
}

const results = {};
for (const t0s of T0S) {
  const t0 = Date.parse(t0s);
  const dir = join(WORK, t0s.replace(/[:]/g, ''));
  mkdirSync(join(dir, 'input'), { recursive: true });
  const lines = [];
  const known = new Set();
  for (let h = 1; h <= CUT; h++) {
    const t = t0 + h * HOUR;
    const rows = [];
    for (const c of coins) {
      if (c.created < t0) continue;
      const x = capAt(c, t);
      if (x === null) continue;
      rows.push({ mint: c.mint, x, created: c.created });
    }
    rows.sort((a, b) => b.x - a.x);
    const top = rows.slice(0, 200);
    for (const r of top) known.add(r.mint);
    lines.push(
      JSON.stringify({
        schema: 'bt',
        at: new Date(t).toISOString(),
        ok: true,
        error: null,
        kind: 'mark',
        t0: new Date(t0).toISOString(),
        counts: {},
        rows: top.map((r, i) => ({
          rank: i + 1,
          mint: r.mint,
          market_cap_lamports: String(Math.floor(10 ** (r.x + 9))),
          venue: 'pool',
          created_at: new Date(r.created).toISOString(),
          complete: true,
          mayhem: false,
        })),
      }),
    );
  }
  const reads = [18, 19, 20, 21, 22, 23, 24].map((h) => new Date(t0 + h * HOUR).toISOString());
  writeFileSync(join(dir, 'input', 'history.ndjson'), lines.join('\n') + '\n');
  writeFileSync(
    join(dir, 'input', 'task.json'),
    JSON.stringify({ competition: 'bt', t0: new Date(t0).toISOString(), history_until: new Date(t0 + CUT * HOUR).toISOString(), reads, places: 5, chart_size: 100, size: { edges: EDGES }, cadence_minutes: 60, output: 'forecast.json', marks: CUT }),
  );
  const charts = reads.map((iso) => {
    const t = Date.parse(iso);
    const rows = [];
    for (const c of coins) {
      if (c.created < t0) continue;
      const x = capAt(c, t);
      if (x !== null) rows.push({ mint: c.mint, x, name: c.name, created: c.created });
    }
    rows.sort((a, b) => b.x - a.x);
    return rows.slice(0, 100);
  });
  const newShare = charts.reduce((a, ch) => a + ch.slice(0, 5).filter((r) => !known.has(r.mint)).length, 0) / (5 * charts.length);
  results[t0s] = { newShare: +newShare.toFixed(2), top: charts.map((ch) => ch.slice(0, 5).map((r) => `${r.name.split(' /')[0]}:${r.x.toFixed(2)}${known.has(r.mint) ? '' : '*'}`).join(' ')) };
  for (const entry of ENTRIES) {
    const out = join(dir, 'out-' + entry.replace(/[^a-z0-9]/gi, '_'));
    mkdirSync(out, { recursive: true });
    let fc = { probabilities: [], sizes: [] };
    try {
      execFileSync('node', [entry], { env: { ...process.env, MAKEFUN_INPUT_DIR: join(dir, 'input'), MAKEFUN_OUTPUT_DIR: out }, stdio: ['ignore', 'ignore', 'ignore'] });
      fc = JSON.parse(readFileSync(join(out, 'forecast.json'), 'utf8'));
    } catch {
      // a failed entry scores the even spread (the empty forecast)
    }
    results[t0s][entry] = score(fc, charts);
  }
}
const verbose = process.env.BT_VERBOSE === '1';
const means = {};
for (const [t0s, r] of Object.entries(results)) {
  if (verbose) {
    console.log(t0s, 'newcomer share of top-5 slots', r.newShare);
    for (const line of r.top) console.log('   ', line);
  }
  for (const e of ENTRIES) {
    const s = r[e];
    console.log(t0s, e.split(/[\\/]/).slice(-2).join('/'), 'total', s.total.toFixed(3), 'order', s.order.toFixed(3), 'size', s.size.toFixed(3));
    (means[e] ??= []).push(s.total);
  }
}
for (const [e, v] of Object.entries(means)) console.log('MEAN', e.split(/[\\/]/).slice(-2).join('/'), (v.reduce((a, b) => a + b, 0) / v.length).toFixed(3));
