"""
scripts/era_cross_test.py — Cross-era PA stress test.

Loads era pools built by build_era_samples.py, selects 3 representative
batters + 3 pitchers per era (6 players per pool), runs 20,000 PAs for
every batter × pitcher pair across all eras, and aggregates results.

Deterministic seed: sha256("{card_id}|{card_id}|era-cross") per matchup.

Usage:
    python scripts/era_cross_test.py [--n 20000]

Output:
    outputs/summaries/era_cross_summary.csv
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import tracemalloc

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from pa_wrapper import resolve_pa_seeded  # noqa: E402

_POOLS_DIR   = os.path.join(_REPO, "data",    "era_pools")
_SUMMARY_DIR = os.path.join(_REPO, "outputs", "summaries")
_OUT_CSV     = os.path.join(_SUMMARY_DIR, "era_cross_summary.csv")

YEARS        = [1906, 1927, 1968, 1999]
TOP_N_EACH   = 3   # top-N batters and top-N pitchers selected per era pool

ERA_TAG = {
    1906: "Deadball",
    1927: "LiveBall-Power",
    1968: "PitcherDom",
    1999: "ModernHR",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def sha32(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 32)

def pa_seed(mseed: int, i: int) -> int:
    return sha32(f"{mseed}:{i}")


def load_pool(year: int) -> tuple[list, list]:
    """Returns (batters, pitchers) lists from era pool JSON."""
    path = os.path.join(_POOLS_DIR, f"{year}.json")
    with open(path) as f:
        cards = json.load(f)
    batters  = [c for c in cards if c["primary_role"] == "Hitter"]
    pitchers = [c for c in cards if c["primary_role"] == "Pitcher"]
    return batters, pitchers


def select_representatives(batters: list, pitchers: list, n: int) -> tuple[list, list]:
    """
    Pick n batters + n pitchers that span the trait range of the era pool.

    Strategy:
      Batters:
        1. Highest POW  (power archetype)
        2. Highest EYE  (contact/discipline archetype)
        3. Highest CON  (pure contact — often different from EYE leader)
      Pitchers:
        1. Highest STF  (flamethrower)
        2. Highest CTL  (control artist)
        3. Highest CMD  (BABIP suppressor)

    All selections are from the pool already sorted by PA/BFP (top-N most
    reliable cards), so we never pick a fringe card.
    """
    def top_by(lst, trait):
        return max(lst, key=lambda c: c["traits"].get(trait, 0))

    if n == 3:
        sel_bat = _dedup([
            top_by(batters, "POW"),
            top_by(batters, "EYE"),
            top_by(batters, "CON"),
        ])
        sel_pit = _dedup([
            top_by(pitchers, "STF"),
            top_by(pitchers, "CTL"),
            top_by(pitchers, "CMD"),
        ])
    else:
        sel_bat = batters[:n]
        sel_pit = pitchers[:n]

    # Pad with top-PA/BFP cards if dedup shrunk below n
    def _pad(selected, pool):
        ids = {c["card_id"] for c in selected}
        for c in pool:
            if len(selected) >= n:
                break
            if c["card_id"] not in ids:
                selected.append(c)
                ids.add(c["card_id"])
        return selected[:n]

    sel_bat = _pad(sel_bat, batters)
    sel_pit = _pad(sel_pit, pitchers)
    return sel_bat, sel_pit


def _dedup(lst: list) -> list:
    """Remove duplicates by card_id, preserving order."""
    seen, out = set(), []
    for c in lst:
        if c["card_id"] not in seen:
            out.append(c)
            seen.add(c["card_id"])
    return out


def run_matchup(batter: dict, pitcher: dict, n: int) -> dict:
    bid  = batter["card_id"]
    pid  = pitcher["card_id"]
    mseed = sha32(f"{bid}|{pid}|era-cross")

    counts = {"K":0,"BB":0,"HBP":0,"HR":0,
              "Single":0,"Double":0,"Triple":0,"Out":0,"Error":0}
    for i in range(n):
        r = resolve_pa_seeded(batter, pitcher, seed=pa_seed(mseed, i))
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    total   = sum(counts.values())
    hits    = counts["Single"] + counts["Double"] + counts["Triple"] + counts["HR"]
    ab      = total - counts["BB"] - counts["HBP"]
    bip_den = ab - counts["K"] - counts["HR"]
    slg_num = (counts["Single"] + 2*counts["Double"] +
               3*counts["Triple"] + 4*counts["HR"])

    return {
        "batter_era":   batter.get("_era", "?"),
        "batter_id":    batter["player_id"],
        "batter_name":  batter["name"],
        "batter_season":batter["season"],
        "pitcher_era":  pitcher.get("_era", "?"),
        "pitcher_id":   pitcher["player_id"],
        "pitcher_name": pitcher["name"],
        "pitcher_season":pitcher["season"],
        "n_pa":         total,
        "k_pct":        round(counts["K"]  / total, 4),
        "bb_pct":       round(counts["BB"] / total, 4),
        "hr_pct":       round(counts["HR"] / total, 4),
        "avg":          round(hits / ab,     3) if ab      else 0.0,
        "obp":          round((hits + counts["BB"] + counts["HBP"]) / total, 3),
        "slg":          round(slg_num / ab,  3) if ab      else 0.0,
        "babip":        round((hits - counts["HR"]) / bip_den, 3) if bip_den else 0.0,
        "matchup_seed": mseed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n: int = 20_000) -> None:
    os.makedirs(_SUMMARY_DIR, exist_ok=True)
    tracemalloc.start()
    t0 = time.perf_counter()

    # ── Load & select ─────────────────────────────────────────────────────────
    all_batters: list[dict] = []
    all_pitchers: list[dict] = []

    for year in YEARS:
        pool_path = os.path.join(_POOLS_DIR, f"{year}.json")
        if not os.path.exists(pool_path):
            print(f"[era-cross] Pool not found: {pool_path} — run build_era_samples.py first.")
            sys.exit(1)

        batters, pitchers = load_pool(year)
        sel_bat, sel_pit  = select_representatives(batters, pitchers, TOP_N_EACH)

        era_label = ERA_TAG.get(year, str(year))
        for c in sel_bat:
            c["_era"] = era_label
        for c in sel_pit:
            c["_era"] = era_label

        all_batters.extend(sel_bat)
        all_pitchers.extend(sel_pit)

        print(f"[era-cross] {year} ({era_label}):  "
              f"batters selected = {[c['name'] for c in sel_bat]}  |  "
              f"pitchers selected = {[c['name'] for c in sel_pit]}")

    total_matchups = len(all_batters) * len(all_pitchers)
    total_pas      = total_matchups * n
    print(f"\n[era-cross] {len(all_batters)} batters × {len(all_pitchers)} pitchers "
          f"= {total_matchups} matchups × {n:,} PAs = {total_pas:,} total PAs\n")

    # ── Simulate ──────────────────────────────────────────────────────────────
    rows = []
    completed = 0
    for batter in all_batters:
        for pitcher in all_pitchers:
            row = run_matchup(batter, pitcher, n)
            rows.append(row)
            completed += 1
            if completed % 12 == 0 or completed == total_matchups:
                elapsed = time.perf_counter() - t0
                print(f"  [{completed:>3}/{total_matchups}] "
                      f"{batter['name'][:20]:<20} vs {pitcher['name'][:20]:<20}  "
                      f"K%={row['k_pct']:.3f}  BB%={row['bb_pct']:.3f}  "
                      f"HR%={row['hr_pct']:.3f}  AVG={row['avg']:.3f}  "
                      f"elapsed={elapsed:.1f}s")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = ["batter_era","batter_season","batter_name","batter_id",
                  "pitcher_era","pitcher_season","pitcher_name","pitcher_id",
                  "n_pa","k_pct","bb_pct","hr_pct","avg","obp","slg","babip",
                  "matchup_seed"]
    with open(_OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - t0
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n[era-cross] Done. {total_pas:,} PAs in {elapsed:.1f}s "
          f"({total_pas/elapsed:,.0f} PA/s)  peak mem {peak_kb/1024:.1f} MB")
    print(f"[era-cross] CSV → {_OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000,
                        help="PAs per matchup (default: 20000)")
    args = parser.parse_args()
    run(n=args.n)
