// Collects recent pump.fun graduates' hourly price history from GeckoTerminal's public API (rate-limited, ~25 calls/min).
// Output: <out>/pools.json (pool list) and <out>/ohlcv/<pool>.json (hourly candles in SOL per token).
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const OUT = process.argv[2];
const SINCE = Date.parse(process.argv[3] ?? '2026-11-08T00:00:00Z');
mkdirSync(join(OUT, 'ohlcv'), { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const BASE = 'https://api.geckoterminal.com/api/v2';
async function get(url) {
  for (let i = 0; i < 4; i++) {
    const res = await fetch(url, { headers: { accept: 'application/json' } });
    if (res.status === 429) { await sleep(20000); continue; }
    if (!res.ok) return null;
    await sleep(2600);
    return res.json();
  }
  return null;
}
const pools = existsSync(join(OUT, 'pools.json')) ? JSON.parse(readFileSync(join(OUT, 'pools.json'), 'utf8')) : {};
const sorts = ['h24_volume_usd_desc', 'h24_tx_count_desc'];
if (Object.keys(pools).length === 0) {
  for (const sort of sorts) {
    for (let page = 1; page <= 10; page++) {
      const d = await get(`${BASE}/networks/solana/dexes/pumpswap/pools?page=${page}&sort=${sort}`);
      if (!d || !d.data) break;
      for (const p of d.data) {
        const a = p.attributes;
        const mint = p.relationships.base_token.data.id.replace('solana_', '');
        if (!mint.endsWith('pump')) continue;
        if (Date.parse(a.pool_created_at) < SINCE) continue;
        pools[a.address] = { mint, name: a.name, created: a.pool_created_at, fdv: +a.fdv_usd, px_sol: +a.base_token_price_native_currency };
      }
      console.error(sort, page, Object.keys(pools).length);
    }
  }
  writeFileSync(join(OUT, 'pools.json'), JSON.stringify(pools, null, 1));
}
for (const addr of Object.keys(pools)) {
  const f = join(OUT, 'ohlcv', addr + '.json');
  if (existsSync(f)) continue;
  const d = await get(`${BASE}/networks/solana/pools/${addr}/ohlcv/hour?aggregate=1&limit=1000&currency=token`);
  if (!d) { console.error('fail', addr); continue; }
  writeFileSync(f, JSON.stringify(d.data?.attributes?.ohlcv_list ?? []));
  console.error('ok', addr);
}
console.error('done');
