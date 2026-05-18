"""
core_stress_test.py — PA Engine Stress Test v0

Modes:
  --audit     Invariant audit: sample N random matchups, verify probability constraints.
  --matrix    Matrix runner: 10,000 PA each for 6 predefined matchups.
  --all       Run both (default).

Output:
  core_stress_report.md written to the working directory.

Usage:
  python3 core_stress_test.py [--audit] [--matrix] [--all] [--n 10000] [--audit_n 500]

Rules:
  - No fatigue, SB, or catcher-arm logic.
  - Deterministic: seeded from matchup identity + PA index.
  - Uses exact repo schema/identifiers from ingest_historical.py and the existing sim runner.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import sys
from datetime import datetime
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import LahmanAdapter, normalize_pitcher_stats, normalize_player_stats
from pa_engine import resolve_duel, PROB_FLOOR, K_CEILING, BB_CEILING, HBP_CEILING
from pa_wrapper import resolve_pa_seeded
from ratings import build_hitter_card, build_pitcher_card

LAHMAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lahman_1871-2025_csv")


# ---------------------------------------------------------------------------
# Card builders (mirrors pedro_vs_gehrig.py — no changes to schema)
# ---------------------------------------------------------------------------

def _load_pitching_row(player_id: str, year: int) -> pd.Series:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "Pitching.csv"))
    df = df[(df["playerID"] == player_id) & (df["yearID"] == year)].copy()
    agg_cols = [c for c in ["G","GS","H","HR","BB","SO","HBP","IPouts","BFP","SF","SH","ER"]
                if c in df.columns]
    df = df.groupby(["playerID","yearID"], as_index=False)[agg_cols].sum()
    return df.iloc[0]


def _load_batting_row(player_id: str, year: int) -> pd.Series:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "Batting.csv"))
    df = df[(df["playerID"] == player_id) & (df["yearID"] == year)].copy()
    agg_cols = [c for c in ["AB","H","2B","3B","HR","BB","SO","HBP","SF","SH","SB","CS"]
                if c in df.columns]
    df = df.groupby(["playerID","yearID"], as_index=False)[agg_cols].sum()
    df["SF"]  = df["SF"].fillna(0)
    df["HBP"] = df["HBP"].fillna(0)
    df["PA"]  = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    df["BIP"] = df["AB"] - df["SO"] - df["HR"] + df["SF"]
    return df.iloc[0]


def _load_people(player_id: str) -> dict:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "People.csv"),
                     usecols=["playerID","nameFirst","nameLast","bats","throws"])
    row = df[df["playerID"] == player_id].iloc[0]
    return {
        "name":   f"{row['nameFirst']} {row['nameLast']}".strip(),
        "bats":   str(row["bats"]),
        "throws": str(row["throws"]),
    }


def build_card_pitcher(player_id: str, year: int, adapter: LahmanAdapter) -> dict:
    p_row  = pd.DataFrame([_load_pitching_row(player_id, year)])
    lg     = adapter.load_league_context(year)
    normed = normalize_pitcher_stats(p_row, lg).iloc[0]
    people = _load_people(player_id)
    card   = build_pitcher_card(normed, people["name"], people["bats"], people["throws"])
    return {
        "player_id":    card.player_id,
        "name":         card.name,
        "season":       card.season,
        "bats":         card.bats,
        "throws":       card.throws,
        "primary_role": "Pitcher",
        "pitcher_role": card.pitcher_role,
        "traits": {"STF": card.STF, "CTL": card.CTL, "CMD": card.CMD, "STA": card.STA},
    }


def build_card_hitter(player_id: str, year: int, adapter: LahmanAdapter) -> dict:
    b_row  = pd.DataFrame([_load_batting_row(player_id, year)])
    lg     = adapter.load_league_context(year)
    normed = normalize_player_stats(b_row, lg).iloc[0]
    people = _load_people(player_id)
    card   = build_hitter_card(normed, people["name"], people["bats"], people["throws"])
    return {
        "player_id":    card.player_id,
        "name":         card.name,
        "season":       card.season,
        "bats":         card.bats,
        "throws":       card.throws,
        "primary_role": "Hitter",
        "traits": {"CON": card.CON, "GAP": card.GAP, "POW": card.POW,
                   "EYE": card.EYE, "AK":  card.AK},
    }


def _avg_pitcher(label: str = "avg_pitcher") -> dict:
    return {
        "player_id": label, "name": "Average Pitcher", "season": 0,
        "bats": "R", "throws": "R", "primary_role": "Pitcher", "pitcher_role": "SP",
        "traits": {"STF": 50, "CTL": 50, "CMD": 50, "STA": 50},
    }


def _avg_hitter(label: str = "avg_hitter") -> dict:
    return {
        "player_id": label, "name": "Average Hitter", "season": 0,
        "bats": "R", "throws": "R", "primary_role": "Hitter",
        "traits": {"CON": 50, "POW": 50, "EYE": 50, "AK": 50, "GAP": 50},
    }


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

def _sha32(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 32)


def simulate_matchup(
    batter: dict,
    pitcher: dict,
    n_pa: int,
    fielder: dict | None = None,
) -> dict[str, int]:
    """
    Simulate N plate appearances, returning raw outcome counts.
    Seeded deterministically from player IDs + PA index.

    Args:
        fielder: Optional ``{RNG, HND, ARM}`` dict. If omitted, ``pa_wrapper``
                 uses its internal default of 50/50/50.
    """
    base    = _sha32(f"{batter['player_id']}|{pitcher['player_id']}")
    context = {"fielder": fielder} if fielder else None
    counts: dict[str, int] = {}
    for i in range(n_pa):
        seed   = _sha32(f"{base}:{i}")
        result = resolve_pa_seeded(batter, pitcher, context=context, seed=seed)
        o      = result["outcome"]
        counts[o] = counts.get(o, 0) + 1
    return counts


def aggregate_counts(counts: dict[str, int], n_pa: int) -> dict[str, Any]:
    k      = counts.get("K",      0)
    bb     = counts.get("BB",     0)
    hbp    = counts.get("HBP",    0)
    hr     = counts.get("HR",     0)
    triple = counts.get("Triple", 0)
    double = counts.get("Double", 0)
    single = counts.get("Single", 0)
    error  = counts.get("Error",  0)

    ab  = n_pa - bb - hbp
    bip = ab - k - hr          # balls in play (no HR, no K)
    hip = single + double + triple
    bip_outs = bip - hip - error

    return {
        "PA": n_pa, "K": k, "BB": bb, "HBP": hbp, "HR": hr,
        "1B": single, "2B": double, "3B": triple, "Error": error,
        "BIP": bip, "BIP_outs": bip_outs, "HIP": hip,
        "K%":    k      / n_pa,
        "BB%":   bb     / n_pa,
        "HBP%":  hbp    / n_pa,
        "HR%":   hr     / n_pa,
        "1B%":   single / n_pa,
        "2B%":   double / n_pa,
        "3B%":   triple / n_pa,
        "BABIP": hip / bip if bip > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Invariant audit
# ---------------------------------------------------------------------------

def run_invariant_audit(n_samples: int = 500) -> dict[str, Any]:
    """
    Sample N random trait combinations, resolve one PA each, and verify:
      1. pK + pBB + pHBP + pBIP == 1.0  (Stage 1 probability closure)
      2. All Stage 1 probabilities are within [PROB_FLOOR, ceiling].
      3. Outcome is one of the known set.

    Returns a summary dict with pass/fail counts and any flagged violations.
    """
    import json
    constants_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_constants.json")
    with open(constants_path) as f:
        constants = json.load(f)

    known_outcomes = {"K", "BB", "HBP", "HR", "Single", "Double", "Triple", "Error", "Out"}
    violations: list[str] = []
    total = 0

    rng_seed = random.Random(42)

    for i in range(n_samples):
        # Random traits in [20, 99]
        batter_traits = {
            "CON": rng_seed.randint(20, 99),
            "EYE": rng_seed.randint(20, 99),
            "AK":  rng_seed.randint(20, 99),
            "POW": rng_seed.randint(20, 99),
            "GAP": rng_seed.randint(20, 99),
        }
        pitcher_traits = {
            "STF": rng_seed.randint(20, 99),
            "CTL": rng_seed.randint(20, 99),
            "CMD": rng_seed.randint(20, 99),
        }
        handedness = {"batter": "R", "pitcher": "R"}
        rng_pa = random.Random(i)

        duel = resolve_duel(batter_traits, pitcher_traits, handedness, constants, rng_pa)
        probs = duel["probabilities"]
        total += 1

        # Check 1: closure — tolerance is 5e-5 because each probability is
        # rounded to 4 decimal places (round(x, 4)) before being returned;
        # four independently-rounded values that sum to 1.0 in full precision
        # can produce a rounded sum up to ±4 × 0.5e-4 = ±2e-4 off from 1.0.
        s = sum(probs.values())
        if abs(s - 1.0) > 2e-4:
            violations.append(
                f"[{i}] pSum={s:.10f} ≠ 1.0  traits={batter_traits}|{pitcher_traits}"
            )

        # Check 2: floor/ceiling bounds
        if probs["K"] < PROB_FLOOR - 1e-12 or probs["K"] > K_CEILING + 1e-12:
            violations.append(f"[{i}] K={probs['K']:.4f} out of range [{PROB_FLOOR},{K_CEILING}]")
        if probs["BB"] < PROB_FLOOR - 1e-12 or probs["BB"] > BB_CEILING + 1e-12:
            violations.append(f"[{i}] BB={probs['BB']:.4f} out of range [{PROB_FLOOR},{BB_CEILING}]")
        if probs["HBP"] < PROB_FLOOR - 1e-12 or probs["HBP"] > HBP_CEILING + 1e-12:
            violations.append(f"[{i}] HBP={probs['HBP']:.4f} out of range [{PROB_FLOOR},{HBP_CEILING}]")
        if probs["BIP"] < PROB_FLOOR - 1e-12:
            violations.append(f"[{i}] BIP={probs['BIP']:.4f} below floor")

        # Check 3: outcome validity (full PA)
        batter_card  = {"player_id": f"audit_{i}", "name": "", "season": 0,
                        "bats": "R", "throws": "R", "primary_role": "Hitter",
                        "traits": batter_traits}
        pitcher_card = {"player_id": f"audit_p_{i}", "name": "", "season": 0,
                        "bats": "R", "throws": "R", "primary_role": "Pitcher",
                        "traits": {**pitcher_traits, "STA": 50}}
        result = resolve_pa_seeded(batter_card, pitcher_card, seed=i)
        if result["outcome"] not in known_outcomes:
            violations.append(f"[{i}] Unknown outcome: {result['outcome']!r}")

    return {
        "n_samples":   total,
        "n_pass":      total - len(violations),
        "n_fail":      len(violations),
        "violations":  violations,
    }


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _trait_str_pitcher(card: dict) -> str:
    t = card["traits"]
    return f"STF {t['STF']}  CTL {t['CTL']}  CMD {t['CMD']}"


def _trait_str_hitter(card: dict) -> str:
    t = card["traits"]
    return f"CON {t['CON']}  POW {t['POW']}  EYE {t['EYE']}  AK {t['AK']}  GAP {t['GAP']}"


def _matchup_md_table(
    label: str,
    pitcher: dict,
    hitter: dict,
    stats: dict[str, Any],
) -> str:
    p_season = f" {pitcher['season']}" if pitcher["season"] else ""
    h_season = f" {hitter['season']}"  if hitter["season"]  else ""
    lines = [
        f"### {label}",
        f"",
        f"**Pitcher:** {pitcher['name']}{p_season} ({pitcher['throws']}HP) — {_trait_str_pitcher(pitcher)}  ",
        f"**Hitter:**  {hitter['name']}{h_season} (bats {hitter['bats']}) — {_trait_str_hitter(hitter)}",
        f"",
        f"| Outcome | Count | Rate |",
        f"|---------|------:|-----:|",
        f"| PA      | {stats['PA']:,} | — |",
        f"| K       | {stats['K']:,} | {stats['K%']:.3f} |",
        f"| BB      | {stats['BB']:,} | {stats['BB%']:.3f} |",
        f"| HBP     | {stats['HBP']:,} | {stats['HBP%']:.3f} |",
        f"| HR      | {stats['HR']:,} | {stats['HR%']:.3f} |",
        f"| 1B      | {stats['1B']:,} | {stats['1B%']:.3f} |",
        f"| 2B      | {stats['2B']:,} | {stats['2B%']:.3f} |",
        f"| 3B      | {stats['3B']:,} | {stats['3B%']:.3f} |",
        f"| BIP outs| {stats['BIP_outs']:,} | {stats['BIP_outs']/stats['PA']:.3f} |",
        f"| **BABIP**| — | **{stats['BABIP']:.3f}** |",
        f"",
    ]
    return "\n".join(lines)


def build_report(
    matchup_results: list[tuple[str, dict, dict, dict]],
    audit_result:    dict[str, Any] | None,
    n_pa:            int,
) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    sections = [
        f"# Chin Music Engine — Core Stress Test Report",
        f"",
        f"Generated: {ts}  ",
        f"PA Engine version: no fatigue / no SB / no catcher arm  ",
        f"PAs per matchup: {n_pa:,}  ",
        f"",
    ]

    # ── Invariant audit ──────────────────────────────────────────────────────
    if audit_result is not None:
        a = audit_result
        pass_str = "✅ PASS" if a["n_fail"] == 0 else f"❌ FAIL ({a['n_fail']} violations)"
        sections += [
            f"---",
            f"",
            f"## Invariant Audit  {pass_str}",
            f"",
            f"| Check | Result |",
            f"|-------|--------|",
            f"| Samples tested | {a['n_samples']:,} |",
            f"| Passed | {a['n_pass']:,} |",
            f"| Failed | {a['n_fail']} |",
            f"",
        ]
        if a["violations"]:
            sections += [
                f"**Violations:**",
                f"```",
            ] + a["violations"][:20] + [
                f"```",
                f"",
            ]
        else:
            sections.append("All checks passed — pK+pBB+pHBP+pBIP=1.0 for every sample; "
                            "all probabilities within bounds; all outcomes valid.\n")

    # ── Matchup tables ───────────────────────────────────────────────────────
    sections += [
        f"---",
        f"",
        f"## Matchup Results",
        f"",
    ]
    for label, pitcher, hitter, stats in matchup_results:
        sections.append(_matchup_md_table(label, pitcher, hitter, stats))

    # ── Summary comparison table ─────────────────────────────────────────────
    sections += [
        f"---",
        f"",
        f"## Summary — All Matchups",
        f"",
        f"| Matchup | K% | BB% | HR% | BABIP |",
        f"|---------|---:|----:|----:|------:|",
    ]
    for label, _, _, stats in matchup_results:
        short = label.replace("Matchup ", "M")
        sections.append(
            f"| {short} | {stats['K%']:.3f} | {stats['BB%']:.3f} "
            f"| {stats['HR%']:.3f} | {stats['BABIP']:.3f} |"
        )

    sections += [
        f"",
        f"---",
        f"",
        f"## Engine Constants (sim_constants.json snapshot)",
        f"",
        f"| Constant | Value |",
        f"|----------|-------|",
        f"| league_avg_k_pct | 0.130 |",
        f"| league_avg_bb_pct | 0.085 |",
        f"| K_STF_ABSOLUTE | 0.25 |",
        f"| K_STF_CON_COEFF | 0.20 |",
        f"| _HR_POW_SCALE | 0.13 |",
        f"| CMD_HR_DAMPENER_EXP | 2.0 |",
        f"| GLOBAL_BABIP_ADJUST | 0.82 |",
        f"",
    ]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------

MATRIX: list[tuple[str, str | None, int | None, str | None, str | None, int | None]] = [
    # (label, pitcher_id, pitcher_yr, hitter_id, hitter_yr)
    # None = use average card
    ("Avg Hitter vs Avg Pitcher",         None,         None, None,         None),
    ("Pedro 2000 vs Gehrig 1927",         "martipe02",  2000, "gehrilo01",  1927),
    ("Pedro 2000 vs Avg Hitter",          "martipe02",  2000, None,         None),
    ("Avg Pitcher vs Gehrig 1927",        None,         None, "gehrilo01",  1927),
    ("Gibson 1968 vs Gehrig 1927",        "gibsobo01",  1968, "gehrilo01",  1927),
    ("deGrom 2018 vs Gehrig 1927",        "degroja01",  2018, "gehrilo01",  1927),
]


def run_matrix(n_pa: int, adapter: LahmanAdapter) -> list[tuple[str, dict, dict, dict]]:
    results = []
    for label, pit_id, pit_yr, bat_id, bat_yr in MATRIX:
        print(f"  [{label}] building cards ...")
        pitcher = (_avg_pitcher(label) if pit_id is None
                   else build_card_pitcher(pit_id, pit_yr, adapter))
        hitter  = (_avg_hitter(label) if bat_id is None
                   else build_card_hitter(bat_id, bat_yr, adapter))

        pt = pitcher["traits"]
        ht = hitter["traits"]
        if pit_id:
            print(f"    ⚾  {pitcher['name']} {pitcher['season']}  "
                  f"STF={pt['STF']} CTL={pt['CTL']} CMD={pt['CMD']}")
        else:
            print(f"    ⚾  Average Pitcher  STF=50 CTL=50 CMD=50")
        if bat_id:
            print(f"    🏏  {hitter['name']} {hitter['season']}  "
                  f"CON={ht['CON']} POW={ht['POW']} EYE={ht['EYE']} "
                  f"AK={ht['AK']} GAP={ht['GAP']}")
        else:
            print(f"    🏏  Average Hitter   CON=50 POW=50 EYE=50 AK=50 GAP=50")

        print(f"    → simulating {n_pa:,} PAs ...")
        counts = simulate_matchup(hitter, pitcher, n_pa)
        stats  = aggregate_counts(counts, n_pa)
        results.append((label, pitcher, hitter, stats))
        print(f"    ✓  K%={stats['K%']:.3f}  BB%={stats['BB%']:.3f}  "
              f"HR%={stats['HR%']:.3f}  BABIP={stats['BABIP']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Core PA Engine Stress Test v0")
    parser.add_argument("--audit",    action="store_true", help="Run invariant audit only")
    parser.add_argument("--matrix",   action="store_true", help="Run matrix matchups only")
    parser.add_argument("--all",      action="store_true", help="Run both (default)")
    parser.add_argument("--n",        type=int, default=10_000, help="PAs per matchup")
    parser.add_argument("--audit_n",  type=int, default=500,    help="Samples for invariant audit")
    parser.add_argument("--out",      default="core_stress_report.md", help="Output markdown file")
    args = parser.parse_args()

    run_audit  = args.audit  or args.all or (not args.matrix and not args.audit)
    run_matrix_flag = args.matrix or args.all or (not args.matrix and not args.audit)

    print(f"\n{'═'*60}")
    print(f"  Chin Music Engine — Core Stress Test v0")
    print(f"  PAs per matchup : {args.n:,}")
    print(f"  Audit samples   : {args.audit_n:,}")
    print(f"{'═'*60}\n")

    audit_result    = None
    matchup_results: list[tuple[str, dict, dict, dict]] = []

    # ── Invariant audit ──────────────────────────────────────────────────────
    if run_audit:
        print(f"Running invariant audit ({args.audit_n:,} random matchups) ...")
        audit_result = run_invariant_audit(args.audit_n)
        a = audit_result
        status = "✅ PASS" if a["n_fail"] == 0 else f"❌ FAIL — {a['n_fail']} violations"
        print(f"  {status}  ({a['n_pass']}/{a['n_samples']} passed)\n")
        for v in a["violations"][:5]:
            print(f"  VIOLATION: {v}")
        if a["violations"]:
            print()

    # ── Matrix runner ────────────────────────────────────────────────────────
    if run_matrix_flag:
        print("Running matchup matrix ...")
        adapter = LahmanAdapter()
        matchup_results = run_matrix(args.n, adapter)
        print()

    # ── Write report ─────────────────────────────────────────────────────────
    report = build_report(matchup_results, audit_result, args.n)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report written → {out_path}\n")

    # ── Print summary to stdout ───────────────────────────────────────────────
    if matchup_results:
        print(f"{'═'*60}")
        print(f"  SUMMARY — {args.n:,} PAs per matchup")
        print(f"{'─'*60}")
        print(f"  {'Matchup':<38}  {'K%':>5}  {'BB%':>5}  {'HR%':>5}  {'BABIP':>6}")
        print(f"  {'─'*38}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
        for label, _, _, stats in matchup_results:
            print(f"  {label:<38}  {stats['K%']:>5.3f}  {stats['BB%']:>5.3f}"
                  f"  {stats['HR%']:>5.3f}  {stats['BABIP']:>6.3f}")
        print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
