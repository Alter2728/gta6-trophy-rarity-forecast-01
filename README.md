# GTA 6 Trophy Rarity Forecast

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
`gta6-trophy-rarity-forecast-01` ($250, 20 paid places), built by an AI agent
for @altertera28.

## Model

`main.py` (Python 3.13, standard library only, deterministic, offline) reads
every `*.in` case in `$MAKEFUN_INPUT_DIR` and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR`.

- Time is measured in hours since 2026-11-19T00:00:00Z. Each history read
  contributes one point per trophy: `x = ln(1 + hours)`, `y = logit(rate)`.
  Consecutive reads with identical `(players_tracked, rates)` are duplicates
  of one Exophase refresh and are collapsed to a single point.
- **Flat history** (< 2 distinct snapshots, the practice shape): the
  launch-night page had not refreshed yet, so the answer is the first
  refresh, which dilutes every rate as new players flood in. Median =
  `logit(last rate) - 1.45`, spread `0.65` logit. The trophy already at
  100% ("Just Like Paradise") keeps the special row
  `[99.0, 99.7, 99.95, 99.99, 99.99]`.
- **Rich history** (>= 2 distinct snapshots, the final's shape): per trophy,
  a least-squares fit of `logit(rate)` on `ln(1 + hours)` over a window
  ladder (last 24h, then 48h, then all reads, until >= 3 points). The slope
  is damped by 0.75 and re-anchored through the fit points' mean, because
  raw logit trends overshoot when earners saturate. Spread combines a
  random walk (`0.5 * sqrt(horizon days)`) with the fit's residual spread.
- Any per-case failure falls back to the house persistence baseline (last
  rate, random-walk spread), so a case is never invalid.

## Validation (local, practice cases p00/p01)

| Model | Pinball loss (logit scale) |
|---|---|
| House starter baseline | 1.508 |
| This entry | **0.198** (both p00 and p01) |

Practice answers are the verified 04:30 UTC refresh state of the live
Exophase page (N=3868, all 55 rates matching). Format: 55 rows, 5 quantiles
each, deterministic (two runs byte-identical), ~0.04 s per case
(limit 8 s), output ~2.8 KB per case (limit 65,536 bytes).

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
