"""
debug_hitter_values.py — Hitter Value Diagnostic

For every historical hitter in the DB, runs N_SIMS PAs and tracks the full
outcome distribution so we can compare:

  • Raw runs/PA      — player production before any replacement subtraction
  • RAR_bat          — runs above UTIL replacement over a 600-PA season
  • WAR              — RAR_bat / RPW

Alongside each player we show:
  • Their ingested traits (CON, POW, EYE, AK, GAP)
  • Simulated HR rate and K rate from the PA engine
  • Actual HR count + HR/PA and K/PA from the Lahman 1927 batting file

This isolates whether a player loses value at the PA engine level (raw rpa)
or only at the replacement-level adjustment stage.

Usage:
    python debug_hitter_values.py [--season YEAR] [--n_sims N]
"""

import argparse
import json
import os
import random
from collections import Counter

import pandas as pd

from database import supabase
from pa_engine import (
    resolve_duel,
    resolve_contact,
    map_bip_outcome,
    resolve_defense,
    derive_game_seed,
    derive_pa_seed,
)
from calculate_values import (
    AVG_PITCHER_TRAITS,
    AVG_DEFENSE,
    NEUTRAL_HANDEDNESS,
    VALUATION_SEED,
    LW,
    RPW,
    SEASON_PA_HIT,
    extract_hitter_traits,
    build_replacement_baselines,
    REP_HITTERS,
)

LAHMAN_DIR = "lahman_1871-2025_csv"


# ─── Simulation with full outcome tracking ────────────────────────────────────

def simulate_outcomes(
    batter_traits:  dict,
    pitcher_traits: dict,
    fielder_traits: dict,
    constants:      dict,
    seed_key:       str,
    n_sims:         int,
) -> tuple[float, Counter]:
    """
    Returns (runs_per_pa, outcome_counts).
    Runs n_sims PAs and tracks every outcome so we can compute HR%, K%, etc.
    """
    game_seed = derive_game_seed(VALUATION_SEED, seed_key)
    counts    = Counter()
    total_lw  = 0.0

    for pa_index in range(n_sims):
        pa_seed = derive_pa_seed(game_seed, pa_index)
        rng     = random.Random(pa_seed)

        duel    = resolve_duel(batter_traits, pitcher_traits, NEUTRAL_HANDEDNESS, constants, rng)
        outcome = duel["outcome"]

        if outcome == "BIP":
            contact = resolve_contact(batter_traits, pitcher_traits, constants, rng)
            bip_map = map_bip_outcome(
                contact_quality = contact["contact_quality"],
                contact_score   = contact["contact_score"],
                spray_vector    = contact["spray_vector"],
                effective_pow   = contact["effective_pow"],
                batter          = batter_traits,
                pitcher         = pitcher_traits,
                constants       = constants,
                rng             = rng,
            )
            defense = resolve_defense(
                bip_outcome     = bip_map["bip_outcome"],
                spray_vector    = contact["spray_vector"],
                contact_quality = contact["contact_quality"],
                fielder         = fielder_traits,
                constants       = constants,
                rng             = rng,
            )
            outcome = defense["final_outcome"]

        counts[outcome] += 1
        total_lw += LW.get(outcome, 0.0)

    return total_lw / n_sims, counts


# ─── Lahman lookup ────────────────────────────────────────────────────────────

def load_lahman_batting(season: int) -> dict[str, dict]:
    """Returns {playerID: {HR, PA, hr_pct, k_pct, bb_pct}} for the given season."""
    bat = pd.read_csv(os.path.join(LAHMAN_DIR, "Batting.csv"))
    s   = bat[bat["yearID"] == season].copy()

    s["HBP"] = s["HBP"].fillna(0)
    s["SF"]  = s["SF"].fillna(0)
    s["SO"]  = s["SO"].fillna(0)
    s["BB"]  = s["BB"].fillna(0)
    s["HR"]  = s["HR"].fillna(0)

    agg = s.groupby("playerID", as_index=False).agg(
        AB=("AB", "sum"), H=("H", "sum"),
        HR=("HR", "sum"), SO=("SO", "sum"),
        BB=("BB", "sum"), HBP=("HBP", "sum"), SF=("SF", "sum"),
    )
    agg["PA"]     = agg["AB"] + agg["BB"] + agg["HBP"] + agg["SF"]
    safe_pa       = agg["PA"].replace(0, float("nan"))
    agg["hr_pct"] = agg["HR"] / safe_pa
    agg["k_pct"]  = agg["SO"] / safe_pa
    agg["bb_pct"] = agg["BB"] / safe_pa

    result = {}
    for _, row in agg.iterrows():
        result[row["playerID"]] = {
            "HR":     int(row["HR"]),
            "PA":     int(row["PA"]),
            "hr_pct": round(float(row["hr_pct"]) if pd.notna(row["hr_pct"]) else 0.0, 4),
            "k_pct":  round(float(row["k_pct"])  if pd.notna(row["k_pct"])  else 0.0, 4),
            "bb_pct": round(float(row["bb_pct"]) if pd.notna(row["bb_pct"]) else 0.0, 4),
        }
    return result


# ─── Formatting helpers ───────────────────────────────────────────────────────

def pct(v: float) -> str:
    return f"{v:.1%}"

def sgn(v: float) -> str:
    return f"{v:+.4f}"


def print_table(
    title: str,
    rows:  list[dict],
    sort_key: str,
    n: int = 10,
) -> None:
    sorted_rows = sorted(rows, key=lambda r: -r[sort_key])[:n]

    hdr = (
        f"  {'#':>2}  {'Player':<24}  {'CON':>3} {'POW':>3} {'EYE':>3}  "
        f"{'raw_rpa':>8}  {'sim_HR%':>7}  {'sim_K%':>6}  "
        f"{'act_HR':>6}  {'act_HR%':>7}  {'act_K%':>6}  "
        f"{'RAR_bat':>7}  {'WAR':>5}"
    )
    rule = "  " + "─" * (len(hdr) - 2)

    print(f"\n{'═' * len(hdr)}")
    print(f"  {title}")
    print(f"  (sort: {sort_key})")
    print(f"{'═' * len(hdr)}")
    print(hdr)
    print(rule)

    for i, r in enumerate(sorted_rows, start=1):
        t = r["traits"]
        print(
            f"  {i:>2}  {r['player_name']:<24}  "
            f"{t['CON']:>3} {t['POW']:>3} {t['EYE']:>3}  "
            f"{r['raw_rpa']:>+8.4f}  "
            f"{pct(r['sim_hr_rate']):>7}  "
            f"{pct(r['sim_k_rate']):>6}  "
            f"{r['act_HR']:>6}  "
            f"{pct(r['act_hr_pct']):>7}  "
            f"{pct(r['act_k_pct']):>6}  "
            f"{r['rar_bat']:>+7.2f}  "
            f"{r['war']:>+5.2f}"
        )
    print(rule)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=1927,
                        help="Season to diagnose (default: 1927)")
    parser.add_argument("--n_sims", type=int, default=2000,
                        help="PA simulations per player (default: 2000)")
    args = parser.parse_args()

    with open("sim_constants.json") as f:
        constants = json.load(f)

    # ── Replacement baselines ─────────────────────────────────────────────────
    print(f"Pre-computing UTIL replacement baseline ({args.n_sims} sims)...")
    rep_hitter_rates, _ = build_replacement_baselines(constants)
    util_rep_rpa        = rep_hitter_rates["UTIL"]
    print(f"  UTIL replacement level: {util_rep_rpa:+.5f} runs/PA")

    # ── Lahman ground-truth stats ─────────────────────────────────────────────
    print(f"\nLoading Lahman batting stats for {args.season}...")
    lahman = load_lahman_batting(args.season)
    print(f"  → {len(lahman)} player-seasons in Lahman")

    # ── Fetch hitters from Supabase ───────────────────────────────────────────
    print(f"\nFetching historical hitters (season={args.season}) from Supabase...")
    all_rows = (
        supabase.table("players")
        .select("*")
        .not_.is_("player_id", "null")
        .is_("stuff", "null")          # hitters only: stuff IS NULL
        .eq("season_year", args.season)
        .execute()
        .data
    )
    print(f"  → {len(all_rows)} hitter rows")

    # ── Simulate every hitter ─────────────────────────────────────────────────
    print(f"\nSimulating {len(all_rows)} hitters × {args.n_sims} PAs ...")
    records: list[dict] = []

    for row in all_rows:
        traits = extract_hitter_traits(row)
        if traits is None:
            continue

        pid      = row["player_id"]
        name     = row["player_name"]
        seed_key = f"hitter_{pid}_{args.season}"

        raw_rpa, counts = simulate_outcomes(
            traits, AVG_PITCHER_TRAITS, AVG_DEFENSE, constants, seed_key, args.n_sims
        )

        total    = sum(counts.values())
        sim_hr   = counts.get("HR", 0)     / total
        sim_k    = counts.get("K", 0)      / total
        sim_bb   = counts.get("BB", 0)     / total
        sim_bip  = (total - counts.get("K", 0) - counts.get("BB", 0) - counts.get("HBP", 0)) / total

        lah        = lahman.get(pid, {})
        act_hr     = lah.get("HR", 0)
        act_hr_pct = lah.get("hr_pct", 0.0)
        act_k_pct  = lah.get("k_pct", 0.0)
        act_bb_pct = lah.get("bb_pct", 0.0)
        act_pa     = lah.get("PA", 0)

        rar_bat = (raw_rpa - util_rep_rpa) * SEASON_PA_HIT
        war     = rar_bat / RPW

        records.append({
            "player_id":    pid,
            "player_name":  name,
            "traits":       traits,
            "raw_rpa":      raw_rpa,
            "sim_hr_rate":  sim_hr,
            "sim_k_rate":   sim_k,
            "sim_bb_rate":  sim_bb,
            "sim_bip_rate": sim_bip,
            "act_HR":       act_hr,
            "act_hr_pct":   act_hr_pct,
            "act_k_pct":    act_k_pct,
            "act_bb_pct":   act_bb_pct,
            "act_pa":       act_pa,
            "rar_bat":      rar_bat,
            "war":          war,
            "outcome_counts": dict(counts),
        })

    print(f"  Done — {len(records)} hitters valued")

    # ── Print the three diagnostic tables ─────────────────────────────────────
    print_table("TOP 10 BY RAW RUNS / PA  (pre-replacement)",
                records, "raw_rpa", n=10)
    print_table("TOP 10 BY RAR_bat  (Runs Above UTIL Replacement, 600-PA season)",
                records, "rar_bat", n=10)
    print_table("TOP 10 BY WAR  (RAR_bat / 10.0)",
                records, "war",     n=10)

    # ── Ruth-specific spotlight ───────────────────────────────────────────────
    ruth = next((r for r in records if r["player_id"] == "ruthba01"), None)
    if ruth:
        t  = ruth["traits"]
        oc = ruth["outcome_counts"]
        total = sum(oc.values())
        print(f"\n{'═'*70}")
        print(f"  RUTH SPOTLIGHT  ({args.season})")
        print(f"{'═'*70}")
        print(f"  Ingested traits   :  CON={t['CON']}  POW={t['POW']}  EYE={t['EYE']}  "
              f"AK={t['AK']}  GAP={t['GAP']}")
        print(f"  Actual stats      :  "
              f"HR={ruth['act_HR']}  "
              f"HR%={pct(ruth['act_hr_pct'])}  "
              f"K%={pct(ruth['act_k_pct'])}  "
              f"BB%={pct(ruth['act_bb_pct'])}  "
              f"PA={ruth['act_pa']}")
        print(f"  Simulated rates   :  "
              f"HR%={pct(ruth['sim_hr_rate'])}  "
              f"K%={pct(ruth['sim_k_rate'])}  "
              f"BB%={pct(ruth['sim_bb_rate'])}  "
              f"BIP%={pct(ruth['sim_bip_rate'])}")
        print(f"  Full outcome dist :")
        for outcome, cnt in sorted(oc.items(), key=lambda x: -x[1]):
            print(f"      {outcome:<8}  {cnt:>5}  ({cnt/total:.1%})")
        print(f"  raw_rpa           :  {ruth['raw_rpa']:+.5f} runs/PA")
        print(f"  UTIL rep level    :  {util_rep_rpa:+.5f} runs/PA")
        print(f"  RAR_bat           :  {ruth['rar_bat']:+.2f}  "
              f"({'ABOVE' if ruth['rar_bat'] > 0 else 'BELOW'} replacement)")
        print(f"  WAR               :  {ruth['war']:+.2f}")
        print(f"{'═'*70}")

        # Diagnosis
        print(f"\n  DIAGNOSIS:")
        hr_gap = ruth["sim_hr_rate"] - ruth["act_hr_pct"]
        k_gap  = ruth["sim_k_rate"]  - ruth["act_k_pct"]
        if k_gap > 0.05:
            print(f"  ⚠  Simulated K% ({pct(ruth['sim_k_rate'])}) >> Actual K% "
                  f"({pct(ruth['act_k_pct'])}) by {pct(k_gap)}.")
            print(f"     Root cause: CON={t['CON']} is too low.")
            print(f"     The BABIP-based CON ingestion doesn't capture Ruth's elite")
            print(f"     strikeout resistance. Fix: blend K-rate into CON derivation.")
        if abs(hr_gap) > 0.02:
            direction = "under" if hr_gap < 0 else "over"
            print(f"  ⚠  Simulated HR% ({pct(ruth['sim_hr_rate'])}) {direction}shoots "
                  f"actual HR% ({pct(ruth['act_hr_pct'])}) by {pct(abs(hr_gap))}.")


if __name__ == "__main__":
    main()
