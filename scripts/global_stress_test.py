#!/usr/bin/env python3
"""
scripts/global_stress_test.py — Global stress test across extreme historical player pools.

Batter pools (from full Lahman history, no era limit except Deadball):
  Moonshot    — Any hitter with >40 HR in a season           (HR-logic ceiling)
  Gap-King    — Any hitter with >50 2B in a season           (ISO/GAP logic)
  Deadball    — Top hitters from 1900–1915 by season AVG     (AVG-logic edge)

Pitcher pools:
  K-Kings     — Any pitcher with >250 K in a season          (K-ceiling)
  Control     — Any pitcher with <20 BB in >150 IP           (CTL floor)
  Wild-Flames — Any pitcher with >100 BB in a season         (CTL ceiling)

Gauntlet: 10,000 PAs for every batter × pitcher combination.
Red flags: K%>50%, BB%>25%, HR%>15%, BABIP<.240 or >.360, AVG>.410.

Usage:
    python scripts/global_stress_test.py [--n 10000]

Outputs:
    outputs/summaries/global_stress_summary.csv
    outputs/summaries/global_red_flags.csv
"""

import argparse
import csv
import hashlib
import os
import sys
import time
import tracemalloc

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from ingestion import (  # noqa: E402
    LahmanAdapter,
    normalize_player_stats,
    normalize_pitcher_stats,
    LAHMAN_DIR,
)
from ratings import build_hitter_card, build_pitcher_card, PlayerCard  # noqa: E402
from pa_wrapper import resolve_pa_seeded  # noqa: E402


# ── Config ────────────────────────────────────────────────────────────────────

POOL_CAP  = 10    # max unique players per sub-pool (after within-pool dedup)
MIN_PA    = 200   # batter eligibility floor
MIN_BFP   = 100   # pitcher eligibility floor
ERA_TAG   = "global-stress"

_SUMMARY_DIR = os.path.join(_REPO, "outputs", "summaries")
_FULL_CSV    = os.path.join(_SUMMARY_DIR, "global_stress_summary.csv")
_FLAG_CSV    = os.path.join(_SUMMARY_DIR, "global_red_flags.csv")

# Red-flag thresholds
FLAG_K_CEIL   = 0.50
FLAG_BB_CEIL  = 0.25
FLAG_HR_CEIL  = 0.15
FLAG_BABIP_LO = 0.240
FLAG_BABIP_HI = 0.360
FLAG_AVG_HI   = 0.410


# ── Helpers ───────────────────────────────────────────────────────────────────

def sha32(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 32)


def pa_seed(mseed: int, i: int) -> int:
    return sha32(f"{mseed}:{i}")


def _card_to_dict(card: PlayerCard, pool: str) -> dict:
    """Convert a PlayerCard to the runtime dict expected by resolve_pa_seeded."""
    return {
        "player_id":    card.player_id,
        "card_id":      f"{card.player_id}|{card.season}",
        "season":       card.season,
        "name":         card.name,
        "bats":         card.bats,
        "throws":       card.throws,
        "primary_role": card.primary_role,
        "pool":         pool,
        "traits":       card.trait_dict(),
    }


# ── Lahman data (loaded once) ─────────────────────────────────────────────────

print("[gst] Loading Lahman CSVs …")

_BAT_COLS = ["playerID", "yearID", "G", "AB", "H", "2B", "3B", "HR",
             "BB", "SO", "HBP", "SF"]
_PIT_COLS = ["playerID", "yearID", "G", "GS", "IPouts", "BFP",
             "H", "HR", "BB", "SO", "HBP"]

_bat_raw = pd.read_csv(
    os.path.join(LAHMAN_DIR, "Batting.csv"),
    usecols=lambda c: c in _BAT_COLS + ["stint", "teamID"],
)
_pit_raw = pd.read_csv(
    os.path.join(LAHMAN_DIR, "Pitching.csv"),
    usecols=lambda c: c in _PIT_COLS + ["stint", "teamID"],
)
_people = pd.read_csv(
    os.path.join(LAHMAN_DIR, "People.csv"),
    usecols=["playerID", "nameFirst", "nameLast", "bats", "throws"],
)

adapter  = LahmanAdapter()
_lg_cache: dict = {}


def _get_lg(year: int) -> dict:
    if year not in _lg_cache:
        _lg_cache[year] = adapter.load_league_context(int(year))
    return _lg_cache[year]


# ── Aggregate multi-stint seasons ─────────────────────────────────────────────

def _agg_batting() -> pd.DataFrame:
    b = _bat_raw.copy()
    for col in ["HBP", "SF", "2B", "3B", "HR", "BB", "SO"]:
        b[col] = b[col].fillna(0)
    agg = (
        b.groupby(["playerID", "yearID"])
        [["G", "AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SF"]]
        .sum(min_count=0)
        .fillna(0)
        .reset_index()
    )
    agg["PA"]  = agg["AB"] + agg["BB"] + agg["HBP"] + agg["SF"]
    agg["avg"] = agg["H"] / agg["AB"].replace(0, float("nan"))
    return agg[(agg["yearID"] >= 1900) & (agg["PA"] >= MIN_PA)].copy()


def _agg_pitching() -> pd.DataFrame:
    p = _pit_raw.copy()
    for col in ["HBP", "BB", "SO", "HR", "H"]:
        p[col] = p[col].fillna(0)
    agg = (
        p.groupby(["playerID", "yearID"])
        [["G", "GS", "IPouts", "BFP", "H", "HR", "BB", "SO", "HBP"]]
        .sum(min_count=0)
        .fillna(0)
        .reset_index()
    )
    agg["IP"]        = agg["IPouts"] / 3.0
    agg["bb_per_ip"] = agg["BB"] / agg["IP"].replace(0, float("nan"))
    return agg[(agg["yearID"] >= 1900) & (agg["BFP"] >= MIN_BFP)].copy()


# ── Pool selection ────────────────────────────────────────────────────────────

def _select_pool(
    df: pd.DataFrame,
    sort_col: str,
    ascending: bool,
    pool_name: str,
    extra_filter=None,
    cap: int = POOL_CAP,
) -> pd.DataFrame:
    """
    Filter → sort → one-season-per-player dedup → cap.
    Returns a DataFrame with a 'pool' column added.
    """
    sub = df.copy()
    if extra_filter is not None:
        sub = sub[extra_filter(sub)]
    sub = sub.sort_values(sort_col, ascending=ascending)
    sub = sub.drop_duplicates(subset="playerID", keep="first")
    top = sub.head(cap).copy()
    top["pool"] = pool_name
    return top


def select_batters(bat: pd.DataFrame) -> pd.DataFrame:
    moonshot = _select_pool(bat, "HR",  False, "Moonshot",
                            extra_filter=lambda d: d["HR"] > 40)
    gap_king = _select_pool(bat, "2B",  False, "Gap-King",
                            extra_filter=lambda d: d["2B"] > 50)
    deadball = _select_pool(bat, "avg", False, "Deadball",
                            extra_filter=lambda d:
                                (d["yearID"] >= 1900) &
                                (d["yearID"] <= 1915) &
                                (d["PA"] >= 300))
    combined = pd.concat([moonshot, gap_king, deadball], ignore_index=True)
    # Cross-pool dedup by (playerID, yearID): first occurrence wins
    combined = combined.drop_duplicates(subset=["playerID", "yearID"]).reset_index(drop=True)
    return combined


def select_pitchers(pit: pd.DataFrame) -> pd.DataFrame:
    k_kings  = _select_pool(pit, "SO",        False, "K-Kings",
                            extra_filter=lambda d: d["SO"] > 250)
    control  = _select_pool(pit, "bb_per_ip", True,  "Control",
                            extra_filter=lambda d: (d["IP"] > 150) & (d["BB"] < 20),
                            cap=50)   # keep all qualifiers (historically <15 total)
    wild_flm = _select_pool(pit, "BB",        False, "Wild-Flames",
                            extra_filter=lambda d: d["BB"] > 100)
    combined = pd.concat([k_kings, control, wild_flm], ignore_index=True)
    combined = combined.drop_duplicates(subset=["playerID", "yearID"]).reset_index(drop=True)
    return combined


# ── Card building ─────────────────────────────────────────────────────────────

def _attach_people(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(_people, on="playerID", how="left")
    df["name"]   = (df["nameFirst"].fillna("") + " " + df["nameLast"].fillna("")).str.strip()
    df["bats"]   = df["bats"].fillna("R")
    df["throws"] = df["throws"].fillna("R")
    return df


def build_batter_cards(df: pd.DataFrame) -> list[dict]:
    df = _attach_people(df)
    pool_map = {(str(r["playerID"]), int(r["yearID"])): str(r["pool"])
                for _, r in df.iterrows()}
    entries = []
    for year, group in df.groupby("yearID"):
        lg    = _get_lg(int(year))
        normed = normalize_player_stats(group.copy(), lg)
        for _, row in normed.iterrows():
            pool = pool_map.get((str(row["playerID"]), int(row["yearID"])), "?")
            card = build_hitter_card(row, row["name"], str(row["bats"]), str(row["throws"]))
            entries.append(_card_to_dict(card, pool))
    return entries


def build_pitcher_cards(df: pd.DataFrame) -> list[dict]:
    df = _attach_people(df)
    pool_map = {(str(r["playerID"]), int(r["yearID"])): str(r["pool"])
                for _, r in df.iterrows()}
    entries = []
    for year, group in df.groupby("yearID"):
        lg    = _get_lg(int(year))
        normed = normalize_pitcher_stats(group.copy(), lg)
        for _, row in normed.iterrows():
            pool = pool_map.get((str(row["playerID"]), int(row["yearID"])), "?")
            card = build_pitcher_card(row, row["name"], str(row["bats"]), str(row["throws"]))
            entries.append(_card_to_dict(card, pool))
    return entries


# ── Matchup runner ────────────────────────────────────────────────────────────

def run_matchup(batter: dict, pitcher: dict, n: int) -> dict:
    bid   = batter["card_id"]
    pid   = pitcher["card_id"]
    mseed = sha32(f"{bid}|{pid}|{ERA_TAG}")

    counts = {
        "K": 0, "BB": 0, "HBP": 0, "HR": 0,
        "Single": 0, "Double": 0, "Triple": 0, "Out": 0, "Error": 0,
    }
    for i in range(n):
        outcome = resolve_pa_seeded(batter, pitcher, seed=pa_seed(mseed, i))["outcome"]
        counts[outcome] = counts.get(outcome, 0) + 1

    total   = sum(counts.values())
    hits    = counts["Single"] + counts["Double"] + counts["Triple"] + counts["HR"]
    ab      = total - counts["BB"] - counts["HBP"]
    bip_den = ab - counts["K"] - counts["HR"]
    slg_num = (counts["Single"] + 2 * counts["Double"] +
               3 * counts["Triple"] + 4 * counts["HR"])

    k_pct  = round(counts["K"]  / total, 4) if total   else 0.0
    bb_pct = round(counts["BB"] / total, 4) if total   else 0.0
    hr_pct = round(counts["HR"] / total, 4) if total   else 0.0
    avg    = round(hits / ab,           3)  if ab      else 0.0
    obp    = round((hits + counts["BB"] + counts["HBP"]) / total, 3) if total else 0.0
    slg    = round(slg_num / ab,        3)  if ab      else 0.0
    babip  = round((hits - counts["HR"]) / bip_den, 3) if bip_den else 0.0

    return {
        "batter_pool":    batter["pool"],
        "batter_name":    batter["name"],
        "batter_id":      batter["player_id"],
        "batter_season":  batter["season"],
        "pitcher_pool":   pitcher["pool"],
        "pitcher_name":   pitcher["name"],
        "pitcher_id":     pitcher["player_id"],
        "pitcher_season": pitcher["season"],
        "n_pa":           total,
        "k_pct":          k_pct,
        "bb_pct":         bb_pct,
        "hr_pct":         hr_pct,
        "avg":            avg,
        "obp":            obp,
        "slg":            slg,
        "babip":          babip,
        "matchup_seed":   mseed,
    }


# ── Red flag detection ────────────────────────────────────────────────────────

def _flags(row: dict) -> dict:
    f = {
        "k_flag":    row["k_pct"]  > FLAG_K_CEIL,
        "bb_flag":   row["bb_pct"] > FLAG_BB_CEIL,
        "hr_flag":   row["hr_pct"] > FLAG_HR_CEIL,
        "babip_flag": (row["babip"] < FLAG_BABIP_LO or row["babip"] > FLAG_BABIP_HI),
        "avg_flag":  row["avg"]    > FLAG_AVG_HI,
    }
    f["any_flag"] = any(f.values())
    return f


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n: int = 10_000) -> None:
    os.makedirs(_SUMMARY_DIR, exist_ok=True)
    tracemalloc.start()
    t0 = time.perf_counter()

    # ── Pool selection ─────────────────────────────────────────────────────────
    print("[gst] Aggregating Lahman stints …")
    bat_agg = _agg_batting()
    pit_agg = _agg_pitching()

    bat_df = select_batters(bat_agg)
    pit_df = select_pitchers(pit_agg)

    print(f"\n[gst] Batter pool  — {len(bat_df)} unique player-seasons")
    for pool, grp in bat_df.groupby("pool"):
        label = {"Moonshot": "HR", "Gap-King": "2B", "Deadball": "AVG"}.get(pool, "?")
        print(f"       {pool:<12}: {len(grp):>3} players  (sorted by {label})")

    print(f"\n[gst] Pitcher pool — {len(pit_df)} unique player-seasons")
    for pool, grp in pit_df.groupby("pool"):
        print(f"       {pool:<12}: {len(grp):>3} players")

    # ── Build cards ────────────────────────────────────────────────────────────
    print("\n[gst] Building player cards …")
    batters  = build_batter_cards(bat_df)
    pitchers = build_pitcher_cards(pit_df)

    total_matchups = len(batters) * len(pitchers)
    total_pas      = total_matchups * n
    print(f"\n[gst] {len(batters)} batters × {len(pitchers)} pitchers "
          f"= {total_matchups:,} matchups × {n:,} PAs = {total_pas:,} total PAs")

    # ── Card roster ────────────────────────────────────────────────────────────
    div = "─" * 78
    print(f"\n{div}")
    print(f"  {'Pool':<12} {'Name':<26} {'Yr':>4}   CON  POW  EYE   AK  GAP")
    print(div)
    for b in sorted(batters, key=lambda c: (c["pool"], c["name"])):
        t = b["traits"]
        print(f"  {b['pool']:<12} {b['name']:<26} {b['season']:>4}   "
              f"{t.get('CON',0):>3}  {t.get('POW',0):>3}  {t.get('EYE',0):>3}  "
              f"{t.get('AK',0):>3}  {t.get('GAP',0):>3}")

    print(f"\n{div}")
    print(f"  {'Pool':<12} {'Name':<26} {'Yr':>4}   STF  CTL  CMD  STA")
    print(div)
    for p in sorted(pitchers, key=lambda c: (c["pool"], c["name"])):
        t = p["traits"]
        print(f"  {p['pool']:<12} {p['name']:<26} {p['season']:>4}   "
              f"{t.get('STF',0):>3}  {t.get('CTL',0):>3}  {t.get('CMD',0):>3}  "
              f"{t.get('STA',0):>3}")
    print()

    # ── Simulate ──────────────────────────────────────────────────────────────
    rows         = []
    completed    = 0
    report_every = max(1, total_matchups // 20)

    for batter in batters:
        for pitcher in pitchers:
            row = run_matchup(batter, pitcher, n)
            rows.append(row)
            completed += 1
            if completed % report_every == 0 or completed == total_matchups:
                elapsed = time.perf_counter() - t0
                eta     = (elapsed / completed) * (total_matchups - completed)
                print(f"  [{completed:>4}/{total_matchups}] "
                      f"{batter['name'][:20]:<20} vs {pitcher['name'][:20]:<20}  "
                      f"K%={row['k_pct']:.3f}  BB%={row['bb_pct']:.3f}  "
                      f"HR%={row['hr_pct']:.3f}  BABIP={row['babip']:.3f}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # ── Write full summary CSV ─────────────────────────────────────────────────
    fieldnames = [
        "batter_pool", "batter_name", "batter_id", "batter_season",
        "pitcher_pool", "pitcher_name", "pitcher_id", "pitcher_season",
        "n_pa", "k_pct", "bb_pct", "hr_pct", "avg", "obp", "slg", "babip",
        "matchup_seed",
    ]
    with open(_FULL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Write red-flag CSV ─────────────────────────────────────────────────────
    flag_rows = []
    for row in rows:
        flags = _flags(row)
        if flags["any_flag"]:
            flag_rows.append({**row, **flags})

    flag_fieldnames = fieldnames + [
        "k_flag", "bb_flag", "hr_flag", "babip_flag", "avg_flag", "any_flag",
    ]
    with open(_FLAG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flag_fieldnames)
        writer.writeheader()
        writer.writerows(flag_rows)

    elapsed = time.perf_counter() - t0
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Final report ──────────────────────────────────────────────────────────
    n_flags = len(flag_rows)
    div2    = "═" * 90

    print(f"\n[gst] Done. {total_pas:,} PAs in {elapsed:.1f}s "
          f"({total_pas/elapsed:,.0f} PA/s)  peak mem {peak_kb/1024:.1f} MB")
    print(f"[gst] Full CSV  → {_FULL_CSV}")
    print(f"[gst] Flags CSV → {_FLAG_CSV}")
    print(f"[gst] Red flags: {n_flags}/{total_matchups} matchups "
          f"({100 * n_flags / max(total_matchups, 1):.1f}%)\n")

    # ── Top 5 HR% ─────────────────────────────────────────────────────────────
    sorted_hr = sorted(rows, key=lambda r: r["hr_pct"], reverse=True)
    sorted_k  = sorted(rows, key=lambda r: r["k_pct"],  reverse=True)

    print(div2)
    print("  TOP 5 SINGLE-MATCHUP  HR%")
    print(div2)
    print(f"  {'#':<3} {'Batter':<24} {'B-Pool':<11} {'Pitcher':<24} {'P-Pool':<12} "
          f"{'HR%':>6}  {'AVG':>5}  {'SLG':>5}")
    print("  " + "─" * 86)
    for i, r in enumerate(sorted_hr[:5], 1):
        print(f"  {i:<3} {r['batter_name']:<24} {r['batter_pool']:<11} "
              f"{r['pitcher_name']:<24} {r['pitcher_pool']:<12} "
              f"{r['hr_pct']:>6.3f}  {r['avg']:>5.3f}  {r['slg']:>5.3f}")

    print(f"\n{div2}")
    print("  TOP 5 SINGLE-MATCHUP  K%")
    print(div2)
    print(f"  {'#':<3} {'Batter':<24} {'B-Pool':<11} {'Pitcher':<24} {'P-Pool':<12} "
          f"{'K%':>6}  {'BB%':>6}  {'BABIP':>6}")
    print("  " + "─" * 86)
    for i, r in enumerate(sorted_k[:5], 1):
        print(f"  {i:<3} {r['batter_name']:<24} {r['batter_pool']:<11} "
              f"{r['pitcher_name']:<24} {r['pitcher_pool']:<12} "
              f"{r['k_pct']:>6.3f}  {r['bb_pct']:>6.3f}  {r['babip']:>6.3f}")

    # ── Mission verdict ────────────────────────────────────────────────────────
    print()
    if n_flags == 0:
        print("  *** NO RED FLAGS — Engine passes Global Stress Test.  MISSION COMPLETE. ***")
    else:
        print(f"  *** {n_flags} matchup(s) flagged — review {_FLAG_CSV} ***")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global PA engine stress test.")
    parser.add_argument("--n", type=int, default=10_000,
                        help="PAs per matchup (default: 10000)")
    args = parser.parse_args()
    run(n=args.n)
