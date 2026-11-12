# pumpfun-top5-forecast

A deterministic, standard-library-only forecaster for make.fun's "Predict the top 5 new Solana coins by market cap,
24 hours out" bounty (`pumpfun-top5-forecast-01`). It reads the house's 10-minute market-cap marks of every pump.fun
coin launched since the opening and writes, for each of the top 5 places at the seven answer reads (18 to 24 hours
after the opening), a probability for each coin and a probability for each market-cap bin.

## Model

A seeded Monte Carlo over every answer read:

- **Incumbents**: each coin ranked at the last mark gets a horizon-scaled log-cap move: mean reversion on its level,
  a little momentum, a fat-tailed (Student-t) diffusion, a crash hazard (most pump.fun coins fade) and a market factor
  that moves every coin on a path together.
- **Newcomers**: coins launched after the history ends count, and no forecast can name them. Their field at a read
  that is L hours after the cut is drawn from the history's own analog (the caps of the coins born in the first L hours
  after the opening, seen at opening + L) as a Poisson process with a random activity multiplier and a smoothed
  exponential tail. Their wins go to the unnamed share of each place.
- **Output**: the place probabilities are the mixture over the seven reads (one forecast is scored at all of them),
  tempered and shrunk toward the unnamed share, with up to 100 coins named per place so the unnamed share goes to
  the newcomers. Sizes are a smoothed histogram of each place's simulated cap mixed with a wide normal.

## Files

- `main.ts`: the entry. Reads `/input` (or `$MAKEFUN_INPUT_DIR`), writes `/output/forecast.json` (or
  `$MAKEFUN_OUTPUT_DIR`).
- `tools/gt_fetch.mjs`: collects hourly candles of recent pump.fun graduates from GeckoTerminal's public API.
- `tools/backtest.mjs`: builds synthetic contest days from those candles (cut at opening + 11 h, reads at 18 to 24 h)
  and scores an entry with the contest's log score.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Built by an AI coding agent (Claude Code) for the make.fun bounty. MIT licensed.
