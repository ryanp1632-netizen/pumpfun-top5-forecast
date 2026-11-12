// pumpfun-top5-forecast-01: a Monte Carlo forecaster of the 5 largest pump.fun coins (launched since t0) at each answer read.
//
// Node 22, standard library only, offline and deterministic (seeded PRNG). It reads /input/task.json and
// /input/history.ndjson (the house's 10-minute marks) and writes /output/forecast.json.
//
// The model, for each answer read r (horizon H hours after the last mark):
//   - incumbents: every coin ranked at the last mark. Its log10 cap moves by a horizon-scaled drift (mean reversion on
//     its level and drawdown, a little momentum), a fat-tailed (Student-t) diffusion, a crash hazard (most coins fade)
//     and a market factor shared by every coin on the path;
//   - newcomers: coins launched after the last mark, which no forecast can name. Their field at read r (the caps of the
//     coins born in a window of length L = r - last mark, seen at the window's end) is drawn from the history's own
//     analog, the field of coins born in the first L hours seen at t0 + L, as a Poisson process with a random activity
//     multiplier, smoothed with an exponential tail;
//   - each simulated chart gives places 1-5 (coin or "unnamed") and their caps. The forecast is the mixture over the
//     seven reads (the one forecast is scored at each, so the best single forecast is their average).
// Place probabilities are tempered and shrunk toward the unnamed share; sizes are a smoothed histogram of each place's
// simulated cap, mixed with a wide normal so no plausible bin is ever near zero.
//
// Outside the grader: MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DEFAULTS = {
  seed: 20261112,
  paths: 4000,
  // incumbent kernel (log10 cap in SOL), horizon H in hours
  muA: -0.006, // alive drift per H^bMu at x0 = 3.5
  bMu: 1.0,
  muLevel: -0.004, // extra drift per unit of (x0 - 3.5), per H^bMu
  muDD: 0.0, // per unit of drawdown from the coin's peak, per H^bMu
  muMom: 0.1, // times the last hour's log change (not horizon-scaled)
  sig: 0.12, // sd per H^bSig for a coin 3+ hours old
  bSig: 0.2,
  youth: 0.8, // sd multiplier 1 + youth * exp(-age / youthTau)
  youthTau: 2.0,
  nu: 4,
  crashRate: 0.05, // base hazard per hour
  youthHaz: 0.25, // extra hazard per hour at age 0, decaying as exp(-age / youthHazTau)
  youthHazTau: 1.5,
  crashMean: -1.6,
  crashSd: 0.6,
  curveSigMult: 1.5,
  curveCrashMult: 2.0,
  sigM: 0.035, // market factor sd per sqrt hour
  // newcomers
  fieldLo: 2.8,
  a0: 1.4, // the newcomer window (13:00 to 02:00 UTC) spans US hours, busier than the history's 02:00 to 13:00
  sA: 0.5,
  jitter: 0.15,
  tailMix: 0.5,
  tailPriorMean: 0.45,
  tailPriorN: 4,
  fallbackRate: 3.0, // field coins above fieldLo per hour when the history has none
  // runtime self-calibration of sig and crashRate from the history (weight calW, prior strength calN0 pairs)
  calW: 0.5,
  calN0: 150,
  calLagH: 1.0,
  calWindowH: 6,
  calTop: 40,
  calLo: 3.0,
  calCrash: 0.8,
  // output
  shrink: 0.04,
  temper: 1.0,
  pMin: 1e-4,
  maxRows: 100,
  sizeKernel: 0.1,
  sizeWide: 0.08,
  sizeWideSd: 0.5,
};

type Params = typeof DEFAULTS;
const P: Params = { ...DEFAULTS, ...(process.env.MAKEFUN_PARAMS ? JSON.parse(process.env.MAKEFUN_PARAMS) : {}) };

// ---------- PRNG (deterministic) ----------
let rs = P.seed >>> 0;
function rand(): number {
  rs = (rs + 0x6d2b79f5) >>> 0;
  let t = rs;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
let spare = NaN;
function gauss(): number {
  if (!Number.isNaN(spare)) {
    const s = spare;
    spare = NaN;
    return s;
  }
  let u = 0;
  let v = 0;
  let s = 0;
  do {
    u = 2 * rand() - 1;
    v = 2 * rand() - 1;
    s = u * u + v * v;
  } while (s >= 1 || s === 0);
  const m = Math.sqrt((-2 * Math.log(s)) / s);
  spare = v * m;
  return u * m;
}
function chi2(nu: number): number {
  let s = 0;
  for (let i = 0; i < nu; i++) {
    const g = gauss();
    s += g * g;
  }
  return s;
}
/** Student-t with nu degrees of freedom, scaled to unit variance. */
function tUnit(nu: number): number {
  return (gauss() / Math.sqrt(chi2(nu) / nu)) * Math.sqrt((nu - 2) / nu);
}
function poisson(lam: number): number {
  if (lam <= 0) return 0;
  if (lam > 60) return Math.max(0, Math.round(lam + Math.sqrt(lam) * gauss()));
  const L = Math.exp(-lam);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rand();
  } while (p > L);
  return k - 1;
}

// ---------- input ----------
interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue?: string;
  created_at?: string;
  complete?: boolean;
  mayhem?: boolean;
}
interface Mark {
  t: number; // hours since t0
  rows: Row[];
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8'));
const PLACES: number = task.places ?? 5;
const EDGES: number[] = task.size.edges;
const NBINS = EDGES.length + 1;
const t0 = Date.parse(task.t0);
const HOUR = 3_600_000;
const readsAt: number[] = (task.reads as string[]).map((s) => (Date.parse(s) - t0) / HOUR);

const marks: Mark[] = [];
for (const line of readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n')) {
  if (line.trim() === '') continue;
  let m: any;
  try {
    m = JSON.parse(line);
  } catch {
    continue;
  }
  if (!m || m.ok === false || !Array.isArray(m.rows) || m.rows.length === 0) continue;
  const t = (Date.parse(m.at) - t0) / HOUR;
  if (!Number.isFinite(t)) continue;
  const rows = (m.rows as Row[]).filter((r) => r && typeof r.mint === 'string' && Number(r.market_cap_lamports) > 0);
  if (rows.length > 0) marks.push({ t, rows: rows.sort((a, b) => a.rank - b.rank) });
}
marks.sort((a, b) => a.t - b.t);

const lg = (lamports: string | number): number => Math.log10(Number(lamports)) - 9;

function binOf(x: number): number {
  let lo = 0;
  let hi = EDGES.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (EDGES[mid]! <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// ---------- forecast ----------
interface Forecast {
  probabilities: { mint: string; place: number; p: number }[];
  top5: string[];
  sizes: { place: number; bin: number; p: number }[];
}

const floor6 = (p: number): number => Math.floor(p * 1e6) / 1e6;

function normalSizes(place: number, mean: number, sd: number, out: Forecast['sizes']): void {
  const cdf = (x: number) => 0.5 * (1 + erf((x - mean) / (sd * Math.SQRT2)));
  for (let b = 0; b < NBINS; b++) {
    const lo = b === 0 ? 0 : cdf(EDGES[b - 1]!);
    const hi = b === NBINS - 1 ? 1 : cdf(EDGES[b]!);
    const p = floor6(hi - lo);
    if (p > 0) out.push({ place, bin: b, p });
  }
}
function erf(x: number): number {
  const t = 1 / (1 + 0.5 * Math.abs(x));
  const y = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? y : -y;
}

/** The house-style persistence forecast: the fallback if the model cannot run. */
function persistence(): Forecast {
  const chart = marks.length > 0 ? marks[marks.length - 1]!.rows : [];
  const probabilities: Forecast['probabilities'] = [];
  const sizes: Forecast['sizes'] = [];
  for (let k = 1; k <= PLACES; k++) {
    if (chart[k - 1]) probabilities.push({ mint: chart[k - 1]!.mint, place: k, p: 0.3 });
    if (k > 1 && chart[k - 2]) probabilities.push({ mint: chart[k - 2]!.mint, place: k, p: 0.1 });
    if (chart[k]) probabilities.push({ mint: chart[k]!.mint, place: k, p: 0.1 });
    normalSizes(k, chart[k - 1] ? lg(chart[k - 1]!.market_cap_lamports) : 3.5, 0.5, sizes);
  }
  const top5 = chart.slice(0, PLACES).map((r) => r.mint);
  for (let i = 0; top5.length < PLACES; i++) top5.push(`${'1'.repeat(40)}${i}pump`.slice(0, 44));
  return { probabilities, top5, sizes };
}

/** A coin's age in hours at time t (hours since t0), from its CreateEvent time (else taken as 2 hours). */
function ageOf(r: Row, t: number): number {
  const c = r.created_at ? (Date.parse(r.created_at) - t0) / HOUR : NaN;
  return Number.isFinite(c) ? Math.max(0, t - c) : 2;
}
const ageMult = (age: number): number => 1 + P.youth * Math.exp(-age / P.youthTau);
/** The cumulative crash hazard over h hours for a coin aged `age`: a base rate plus a youth hazard that decays with age. */
const cumHaz = (age: number, h: number): number => P.crashRate * h + P.youthHaz * P.youthHazTau * (Math.exp(-age / P.youthHazTau) - Math.exp(-(age + h) / P.youthHazTau));

function model(): Forecast {
  if (marks.length === 0) throw new Error('no marks');
  const last = marks[marks.length - 1]!;
  const T = last.t;

  // per-coin series
  const series = new Map<string, { t: number; x: number }[]>();
  for (const m of marks) {
    for (const r of m.rows) {
      let s = series.get(r.mint);
      if (!s) series.set(r.mint, (s = []));
      s.push({ t: m.t, x: lg(r.market_cap_lamports) });
    }
  }

  // self-calibration: the history's own recent volatility and crash rate for coins in contention, blended with the priors
  if (P.calW > 0 && marks.length >= 3) {
    const cadence = (T - marks[0]!.t) / Math.max(1, marks.length - 1);
    const lagH = Math.min(P.calLagH, Math.max(cadence, (T - marks[0]!.t) / 2));
    const devs: number[] = [];
    let crashes = 0;
    let expected = 0;
    let n = 0;
    for (let a = 0; a < marks.length; a++) {
      const ma = marks[a]!;
      if (ma.t < T - P.calWindowH - lagH) continue;
      let b = a + 1;
      while (b < marks.length && marks[b]!.t < ma.t + lagH - 1e-6) b++;
      if (b >= marks.length) break;
      const mb = marks[b]!;
      const h = mb.t - ma.t;
      if (h <= 0) continue;
      const at = new Map<string, number>();
      for (const r of mb.rows) at.set(r.mint, lg(r.market_cap_lamports));
      const floorB = mb.rows.length >= 150 ? lg(mb.rows[mb.rows.length - 1]!.market_cap_lamports) : -Infinity;
      for (const r of ma.rows.slice(0, P.calTop)) {
        const x = lg(r.market_cap_lamports);
        if (x < P.calLo) continue;
        const y = at.get(r.mint);
        n++;
        expected += 1 - Math.exp(-cumHaz(ageOf(r, ma.t), h));
        if (y === undefined) {
          if (x - floorB > P.calCrash) crashes++;
          continue;
        }
        if (y - x < -P.calCrash) crashes++;
        else devs.push((y - x) / Math.pow(h, P.bSig) / ageMult(ageOf(r, ma.t)));
      }
    }
    if (devs.length >= 8) {
      const med = [...devs].sort((p, q) => p - q)[devs.length >> 1]!;
      const mad = [...devs.map((d) => Math.abs(d - med))].sort((p, q) => p - q)[devs.length >> 1]!;
      const sigHat = (1.4826 * mad) / Math.sqrt(P.nu / (P.nu - 2)) / 0.85; // t(nu) has a smaller MAD than its sd
      const w = (P.calW * devs.length) / (devs.length + P.calN0);
      P.sig = Math.exp((1 - w) * Math.log(P.sig) + w * Math.log(Math.min(Math.max(sigHat, P.sig / 3), P.sig * 3)));
      // the crash hazard scales by observed / expected crashes (ages and the lag accounted for), shrunk toward 1
      const ratio = Math.min(2.5, Math.max(0.4, (crashes + 2) / (expected + 2)));
      const wc = (P.calW * n) / (n + P.calN0);
      const mult = Math.exp(wc * Math.log(ratio));
      P.crashRate *= mult;
      P.youthHaz *= mult;
      if (process.env.FC_DEBUG) process.stderr.write(`selfcal lag ${lagH.toFixed(2)}h pairs ${n} sigHat ${sigHat.toFixed(3)} crashes ${crashes} expected ${expected.toFixed(1)} -> sig ${P.sig.toFixed(3)} hazard x${mult.toFixed(2)}\n`);
    } else if (process.env.FC_DEBUG) process.stderr.write(`selfcal skipped: ${devs.length} pairs\n`);
  }

  // incumbents: the last mark's ranked coins
  const cand = last.rows.map((r) => {
    const s = series.get(r.mint)!;
    const x0 = lg(r.market_cap_lamports);
    let peak = x0;
    let ref = x0;
    let refFound = false;
    for (const p of s) {
      if (p.x > peak) peak = p.x;
      if (!refFound && p.t >= T - 1.0 - 1e-6) {
        ref = p.x;
        refFound = true;
      }
    }
    const complete = r.complete === true || r.venue === 'pool' || r.venue === 'migrating';
    return { mint: r.mint, x0, dd: peak - x0, mom: x0 - ref, complete, age: ageOf(r, T), sdMult: ageMult(ageOf(r, T)) * (complete ? 1 : P.curveSigMult) };
  });
  const NC = cand.length;

  // newcomer analog fields
  const span = T; // hours from t0 to the last mark
  const fields = readsAt.map((R) => {
    const L = Math.max(R - T, 1 / 6);
    let best = last;
    if (L <= span) {
      let bd = Infinity;
      for (const m of marks) {
        const d = Math.abs(m.t - L);
        if (m.t > 0 && d < bd) {
          bd = d;
          best = m;
        }
      }
    }
    const pts = best.rows.map((r) => lg(r.market_cap_lamports)).filter((x) => x >= P.fieldLo);
    const scale = best.t > 0 ? L / best.t : 1;
    let sumEx = 0;
    for (const x of pts) sumEx += x - P.fieldLo;
    const tailMean = (sumEx + P.tailPriorMean * P.tailPriorN) / (pts.length + P.tailPriorN);
    const n = pts.length > 0 ? pts.length * scale : P.fallbackRate * L;
    return { L, pts, n, tailMean };
  });

  const NR = readsAt.length;
  const cnt: Float64Array[] = Array.from({ length: PLACES }, () => new Float64Array(NC + 1));
  const sizeCnt: Float64Array[] = Array.from({ length: PLACES }, () => new Float64Array(NBINS));
  const topV = new Float64Array(PLACES);
  const topI = new Int32Array(PLACES);
  const xs = new Float64Array(NC);

  for (let r = 0; r < NR; r++) {
    const H = Math.max(readsAt[r]! - T, 1 / 60);
    const muScale = Math.pow(H, P.bMu);
    const sdBase = P.sig * Math.pow(H, P.bSig);
    const pcs = cand.map((c) => 1 - Math.exp(-cumHaz(c.age, H) * (c.complete ? 1 : P.curveCrashMult)));
    const sdM = P.sigM * Math.sqrt(H);
    const mu = cand.map((c) => (P.muA + P.muLevel * (c.x0 - 3.5) + P.muDD * c.dd) * muScale + P.muMom * c.mom);
    const f = fields[r]!;
    for (let s = 0; s < P.paths; s++) {
      const MF = sdM * gauss();
      for (let k = 0; k < PLACES; k++) {
        topV[k] = -Infinity;
        topI[k] = -2;
      }
      const push = (v: number, idx: number) => {
        if (v <= topV[PLACES - 1]!) return;
        let k = PLACES - 1;
        while (k > 0 && topV[k - 1]! < v) {
          topV[k] = topV[k - 1]!;
          topI[k] = topI[k - 1]!;
          k--;
        }
        topV[k] = v;
        topI[k] = idx;
      };
      for (let i = 0; i < NC; i++) {
        const c = cand[i]!;
        const pc = pcs[i]!;
        let x: number;
        if (rand() < pc) x = c.x0 + P.crashMean + P.crashSd * gauss();
        else x = c.x0 + mu[i]! + sdBase * c.sdMult * tUnit(P.nu);
        x += MF;
        xs[i] = x;
        push(x, i);
      }
      // newcomers
      const A = P.a0 * Math.exp(P.sA * gauss() - (P.sA * P.sA) / 2);
      const K = poisson(A * f.n);
      for (let j = 0; j < K; j++) {
        let x: number;
        if (f.pts.length === 0 || rand() < P.tailMix) x = P.fieldLo - f.tailMean * Math.log(1 - rand());
        else x = f.pts[Math.floor(rand() * f.pts.length)]! + P.jitter * gauss();
        push(x + MF, NC);
      }
      for (let k = 0; k < PLACES; k++) {
        const idx = topI[k]!;
        if (idx === -2) continue;
        cnt[k]![idx] += 1;
        sizeCnt[k]![binOf(topV[k]!)] += 1;
      }
    }
  }

  const total = NR * P.paths;
  const probabilities: Forecast['probabilities'] = [];
  const sizes: Forecast['sizes'] = [];
  const qAt: number[][] = [];
  for (let k = 0; k < PLACES; k++) {
    const c = cnt[k]!;
    const q = new Array<number>(NC);
    let named = 0;
    let tempered = 0;
    for (let i = 0; i < NC; i++) {
      q[i] = c[i]! / total;
      named += q[i]!;
      tempered += Math.pow(q[i]!, P.temper);
    }
    if (tempered > 0) for (let i = 0; i < NC; i++) q[i] = (Math.pow(q[i]!, P.temper) / tempered) * named * (1 - P.shrink);
    qAt.push(q);
    const order = [...q.keys()].sort((a, b) => q[b]! - q[a]! || cand[a]!.x0 - cand[b]!.x0 || (cand[a]!.mint < cand[b]!.mint ? -1 : 1));
    const chosen = order.slice(0, Math.min(P.maxRows, NC));
    let sum = 0;
    const ps = chosen.map((i) => {
      const p = Math.max(q[i]!, P.pMin);
      sum += p;
      return p;
    });
    const scale = sum > 0.999 ? 0.999 / sum : 1;
    chosen.forEach((i, j) => {
      const p = floor6(ps[j]! * scale);
      if (p > 0) probabilities.push({ mint: cand[i]!.mint, place: k + 1, p });
    });

    // sizes: smoothed histogram + a wide normal around its median
    const h = sizeCnt[k]!;
    let hs = 0;
    for (let b = 0; b < NBINS; b++) hs += h[b]!;
    const sm = new Float64Array(NBINS);
    const kb = P.sizeKernel / 0.05;
    const R = Math.ceil(kb * 4);
    for (let b = 0; b < NBINS; b++) {
      if (h[b] === 0) continue;
      let wsum = 0;
      const w: number[] = [];
      for (let d = -R; d <= R; d++) {
        const bb = b + d;
        const ww = bb >= 0 && bb < NBINS ? Math.exp(-0.5 * (d / kb) * (d / kb)) : 0;
        w.push(ww);
        wsum += ww;
      }
      for (let d = -R; d <= R; d++) {
        const bb = b + d;
        if (bb >= 0 && bb < NBINS) sm[bb] += (h[b]! / hs) * (w[d + R]! / wsum);
      }
    }
    let acc = 0;
    let medBin = 0;
    for (let b = 0; b < NBINS; b++) {
      acc += sm[b]!;
      if (acc >= 0.5) {
        medBin = b;
        break;
      }
    }
    const med = medBin === 0 ? EDGES[0]! - 0.025 : medBin === NBINS - 1 ? EDGES[EDGES.length - 1]! + 0.025 : (EDGES[medBin - 1]! + EDGES[medBin]!) / 2;
    const wide: Forecast['sizes'] = [];
    normalSizes(k + 1, med, P.sizeWideSd, wide);
    const wideP = new Float64Array(NBINS);
    for (const w of wide) wideP[w.bin] = w.p;
    for (let b = 0; b < NBINS; b++) {
      const p = floor6((1 - P.sizeWide) * sm[b]! + P.sizeWide * wideP[b]!);
      if (p > 0) sizes.push({ place: k + 1, bin: b, p });
    }
  }

  // top5: greedy by place
  const top5: string[] = [];
  for (let k = 0; k < PLACES; k++) {
    let bi = -1;
    for (let i = 0; i < NC; i++) if (!top5.includes(cand[i]!.mint) && (bi < 0 || qAt[k]![i]! > qAt[k]![bi]!)) bi = i;
    if (bi >= 0) top5.push(cand[bi]!.mint);
  }
  for (const mint of series.keys()) if (top5.length < PLACES && !top5.includes(mint)) top5.push(mint);
  if (top5.length < PLACES) throw new Error('fewer than 5 coins');
  return { probabilities, top5, sizes };
}

function valid(f: Forecast): boolean {
  if (!Array.isArray(f.top5) || new Set(f.top5).size !== PLACES) return false;
  const sums = new Array<number>(PLACES + 1).fill(0);
  const counts = new Array<number>(PLACES + 1).fill(0);
  for (const r of f.probabilities) {
    if (!(r.p >= 0 && r.p <= 1) || r.place < 1 || r.place > PLACES) return false;
    sums[r.place] += r.p;
    counts[r.place] += 1;
  }
  if (sums.some((s) => s > 1 + 1e-9) || counts.some((c) => c > 100)) return false;
  const ss = new Array<number>(PLACES + 1).fill(0);
  for (const r of f.sizes) {
    if (!(r.p >= 0 && r.p <= 1) || r.bin < 0 || r.bin >= NBINS) return false;
    ss[r.place] += r.p;
  }
  return !ss.some((s) => s > 1 + 1e-9);
}

let forecast: Forecast;
try {
  forecast = model();
  if (!valid(forecast)) throw new Error('invalid model output');
} catch (err) {
  process.stderr.write(`model failed (${(err as Error).message}); writing persistence\n`);
  forecast = persistence();
}
mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output ?? 'forecast.json'), `${JSON.stringify(forecast)}\n`);
