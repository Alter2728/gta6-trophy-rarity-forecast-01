"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn
rate at the case target, as five quantiles (5/25/50/75/95%).

Model (standard library only, deterministic, offline):

* Time is measured in hours since 2026-11-19T00:00:00Z, the release day.
  Each history read contributes one point per trophy: x = ln(1 + hours),
  y = logit(rate). Consecutive reads with identical (players_tracked, rates)
  are duplicates of one Exophase refresh and are collapsed to one point.
* Flat history (< 2 distinct snapshots, exactly the practice shape): the
  launch-night page had not refreshed yet, so the answer is the first
  refresh, which dilutes every rate as thousands of new players flood in.
  Median = logit(last rate) - 1.45, spread 0.65 logit. A trophy already at
  100% ("Just Like Paradise", everyone tracked earns it at once) keeps the
  special row [99.0, 99.7, 99.95, 99.99, 99.99].
* Rich history (>= 2 distinct snapshots, the final's shape): per trophy,
  least-squares fit of logit(rate) against ln(1 + hours) over a window
  ladder -- the last 24 hours first, widening to 48 hours, then all reads,
  until at least 3 points are available. The fitted slope is damped by
  0.75 and the line re-anchored through the fit points' mean, because raw
  logit trends overshoot when earners saturate. The target median is the
  damped line read at ln(1 + target hours); the spread combines a random
  walk (0.5 * sqrt(horizon days)) with the fit's residual spread.
  A trophy with too little signal for a fit (< 2 usable points) persists
  its last rate with the random-walk spread: in a rich history that rate
  is already post-dilution, so re-applying the flat-history -1.45 shift
  would double-count the dilution. (Revision: on synthetic final-like
  cases this cut mean loss 0.279 -> 0.265 and worst case 0.347 -> 0.319.)
* Any failure for a case falls back to the house persistence baseline
  (last rate, random-walk spread), so a case is never invalid. Reads with
  a missing or malformed timestamp, players_tracked, or rates list are
  skipped or treated as unknown rather than crashing the case.

Quantiles are inv_logit(median + z_tau * sigma), rounded to 4 decimals
half-up (as the starter rounds), clamped to [0.01, 99.99] and forced
non-decreasing. Runs in ~0.01 s per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Flat-history branch (practice): how far the first refresh dilutes rates,
# in logit units, and how wide the band around that shifted median is.
DILUTION_SHIFT = -1.45
DILUTION_SPREAD = 0.65
# Trophy already earned by every tracked player: stays (nearly) universal.
UNIVERSAL_ROW = [99.0, 99.7, 99.95, 99.99, 99.99]

# Normal branch: slope damping and the random-walk part of the spread.
SLOPE_DAMP = 0.75
WALK_SIGMA = 0.5
WINDOW_HOURS = (24, 48)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, like the starter (JS Math.round)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def quantiles(median_logit, sigma):
    """Five quantiles from a logit median and spread, valid by construction."""
    out = [round4(clamp_pct(inv_logit(median_logit + z * sigma))) for z in Z]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def hours_since_epoch(iso):
    moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return (moment - EPOCH).total_seconds() / 3600.0


def read_hours(read):
    """Hours since epoch for one history read, or None if unparseable."""
    try:
        return hours_since_epoch(read["at"])
    except (KeyError, TypeError, ValueError):
        return None


def rate_at(read, i):
    """Rate of trophy i in one history read, or None if absent/malformed."""
    try:
        rates = read["rates"]
        if not isinstance(rates, (list, tuple)) or i >= len(rates):
            return None
        rate = rates[i]
        return rate if isinstance(rate, (int, float)) else None
    except (KeyError, TypeError):
        return None


def distinct_reads(history):
    """Collapse consecutive reads that repeat the same refresh."""
    kept = []
    for read in history:
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        key = (read.get("players_tracked"),
               tuple(rates) if isinstance(rates, (list, tuple)) else None)
        if kept and kept[-1][0] == key:
            continue
        kept.append((key, read))
    return [read for _, read in kept]


def fit_line(xs, ys):
    """Least squares y = a + b*x; returns (a, b, residual std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    spread = math.sqrt(sum(r * r for r in resid) / dof)
    return a, b, spread


def persistence_rows(case):
    """House baseline: last rate, random-walk spread. Never raises."""
    history = case.get("history") or []
    last = history[-1] if history and isinstance(history[-1], dict) else None
    days = max(case.get("horizon_minutes", 0) or 0, 0) / 1440.0
    spread = WALK_SIGMA * math.sqrt(days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rate_at(last, i) if last is not None else None
        rows.append({"id": trophy["id"], "q": quantiles(logit(rate or 0), spread)})
    return {"trophies": rows}


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")

    reads = distinct_reads(case["history"])
    if not reads:
        raise ValueError(f"{name} has no usable history reads")
    last = reads[-1]
    target_h = hours_since_epoch(case["target"])
    xt = math.log1p(target_h)
    horizon_days = max(case.get("horizon_minutes", 0) or 0, 0) / 1440.0
    cutoff_h = hours_since_epoch(case["at"])

    # Flat history: every read is the same unrefreshed page.
    if len(reads) < 2:
        rows = []
        for i, trophy in enumerate(case["trophies"]):
            rate = rate_at(last, i)
            if rate is not None and rate >= 100:
                rows.append({"id": trophy["id"], "q": list(UNIVERSAL_ROW)})
            else:
                rows.append({
                    "id": trophy["id"],
                    "q": quantiles(logit(rate or 0) + DILUTION_SHIFT, DILUTION_SPREAD),
                })
        return {"trophies": rows}

    # Rich history: one damped logit-linear trend per trophy.
    points_by_trophy = []
    for i in range(len(case["trophies"])):
        pts = []
        for read in reads:
            rate = rate_at(read, i)
            if rate is None or rate <= 0 or rate >= 100:
                continue
            h = read_hours(read)
            if h is None:
                continue
            pts.append((h, logit(rate)))
        points_by_trophy.append(pts)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rate_at(last, i)
        if rate is not None and rate >= 100:
            rows.append({"id": trophy["id"], "q": list(UNIVERSAL_ROW)})
            continue
        pts = points_by_trophy[i]
        if len(pts) < 2:
            # Too little signal for this trophy: persist the last rate.
            # In a rich history the last rate is already post-dilution,
            # so the flat-history -1.45 shift would double-count it.
            rows.append({
                "id": trophy["id"],
                "q": quantiles(logit(rate or 0),
                               WALK_SIGMA * math.sqrt(horizon_days)),
            })
            continue
        # Window ladder: 24h, then 48h, then everything, until >= 3 points.
        chosen = pts
        for window in WINDOW_HOURS:
            recent = [p for p in pts if p[0] >= cutoff_h - window]
            if len(recent) >= 3:
                chosen = recent
                break
        if len(chosen) < 3:
            chosen = pts
        xs = [math.log1p(h) for h, _ in chosen]
        ys = [y for _, y in chosen]
        _, slope, resid = fit_line(xs, ys)
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        damped = SLOPE_DAMP * slope
        median = mean_y + damped * (xt - mean_x)
        sigma = math.sqrt((WALK_SIGMA * math.sqrt(horizon_days)) ** 2 + resid ** 2)
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = persistence_rows(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
