"""
engine_validation.py — Pedro Martinez 2000 vs 1927 Yankees (1,000 PA)

Builds player cards using the new global z-score calibration system,
then runs 1,000 plate appearances for each 1927 Yankee hitter against
Pedro Martinez 2000, aggregating K%, BB%, HR%, and an ERA proxy.

The goal: verify that a +3σ pitcher dominates a +2σ lineup the way
physics demands — not just "better than average" but historically crushing.

Usage:
    python engine_validation.py [--n 1000]
"""

import argparse
import hashlib
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from global_calibration import get_global_stats, z_score_trait
from ingestion import LahmanAdapter, normalize_pitcher_stats, normalize_player_stats
from ratings import (
    build_hitter_card,
    build_pitcher_card,
    _map_to_trait_legacy,
    _AVG_IP_PER_START,
)
from pa_wrapper import resolve_pa_seeded

LAHMAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lahman_1871-2025_csv")

# ── Linear weights (FanGraphs / custom calibrated) ───────────────────────────
# Used to estimate ERA-proxy from raw outcome counts.
_LW = {
    "HR":     1.40,
    "Triple": 1.05,
    "Double": 0.77,
    "Single": 0.47,
    "BB":     0.30,
    "HBP":    0.30,
    "Out":    -0.10,
    "Error":   0.05,
    "K":      -0.10,
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_batting_season(year: int, team: str = None) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "Batting.csv"))
    df = df[df["yearID"] == year]
    if team:
        df = df[df["teamID"] == team]
    agg = [c for c in ["AB","H","2B","3B","HR","BB","SO","HBP","SF","SH","SB","CS"]
           if c in df.columns]
    df = df.groupby(["playerID","yearID"], as_index=False)[agg].sum()
    df["SF"]  = df["SF"].fillna(0)  if "SF"  in df.columns else 0
    df["HBP"] = df["HBP"].fillna(0) if "HBP" in df.columns else 0
    df["PA"]  = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    df["BIP"] = df["AB"] - df["SO"] - df["HR"] + df["SF"]
    return df


def load_pitching_player(player_id: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "Pitching.csv"))
    df = df[(df["playerID"] == player_id) & (df["yearID"] == year)]
    agg = [c for c in ["G","GS","H","HR","BB","SO","HBP","IPouts","BFP","SF","SH"]
           if c in df.columns]
    df = df.groupby(["playerID","yearID"], as_index=False)[agg].sum()
    if "BFP" in df.columns and df["BFP"].notna().any():
        df["BF"] = df["BFP"].fillna(0)
    else:
        outs = df["IPouts"].fillna(0) if "IPouts" in df.columns else 0
        hbp  = df["HBP"].fillna(0)   if "HBP"   in df.columns else 0
        df["BF"] = outs + df["H"].fillna(0) + df["BB"].fillna(0) + hbp
    return df


def load_people() -> pd.DataFrame:
    return pd.read_csv(os.path.join(LAHMAN_DIR, "People.csv"),
                       usecols=["playerID","nameFirst","nameLast","bats","throws"])


# ── Card builders ─────────────────────────────────────────────────────────────

def build_pedro_card(people: pd.DataFrame, adapter: LahmanAdapter) -> dict:
    pit_raw = load_pitching_player("martipe02", 2000)
    if pit_raw.empty:
        raise RuntimeError("Pedro Martinez 2000 not found in Lahman Pitching.csv")

    lg = adapter.load_league_context(2000)
    pit_normed = normalize_pitcher_stats(pit_raw, lg)

    row = pit_normed.iloc[0]
    p   = people[people["playerID"] == "martipe02"].iloc[0]
    card = build_pitcher_card(row, "Pedro Martinez", str(p["bats"]), str(p["throws"]),
                              team_id="BOS")
    return {
        "player_id":   card.player_id,
        "name":        card.name,
        "season":      card.season,
        "team_id":     card.team_id,
        "bats":        card.bats,
        "throws":      card.throws,
        "primary_role":"Pitcher",
        "pitcher_role": card.pitcher_role,
        "traits": {
            "STF": card.STF, "CTL": card.CTL, "CMD": card.CMD, "STA": card.STA,
            # Engine also needs CON/POW/GAP/EYE/AK from pitcher side (unused but harmless)
        },
    }


def build_yankees_lineup(people: pd.DataFrame, adapter: LahmanAdapter) -> list[dict]:
    """Build hitter cards for the 1927 NYA core lineup (min 200 PA)."""
    bat_raw = load_batting_season(1927, team="NYA")
    if bat_raw.empty:
        raise RuntimeError("1927 NYA batting not found")

    lg = adapter.load_league_context(1927)
    bat_normed = normalize_player_stats(bat_raw, lg)

    cards = []
    for _, row in bat_normed.iterrows():
        if row["PA"] < 200:
            continue
        pid = row["playerID"]
        p_row = people[people["playerID"] == pid]
        if p_row.empty:
            name   = pid
            bats   = "R"
            throws = "R"
        else:
            p_row = p_row.iloc[0]
            name   = f"{p_row['nameFirst']} {p_row['nameLast']}".strip()
            bats   = str(p_row["bats"])
            throws = str(p_row["throws"])

        card = build_hitter_card(row, name, bats, throws, team_id="NYA")
        cards.append({
            "player_id":    card.player_id,
            "name":         card.name,
            "season":       card.season,
            "team_id":      card.team_id,
            "bats":         card.bats,
            "throws":       card.throws,
            "primary_role": "Hitter",
            "traits": {
                "CON": card.CON, "GAP": card.GAP, "POW": card.POW,
                "EYE": card.EYE, "AK":  card.AK,
            },
            "_pa": int(row["PA"]),
        })
    # Sort by PA descending so the core starters appear first
    return sorted(cards, key=lambda c: -c["_pa"])


# ── Simulation ────────────────────────────────────────────────────────────────

def _sha32(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 32)


def run_matchup(batter: dict, pitcher: dict, n_pa: int) -> dict:
    mseed = _sha32(f"{batter['player_id']}|{pitcher['player_id']}|pedro-validation")
    counts: dict[str, int] = {}
    for i in range(n_pa):
        seed  = _sha32(f"{mseed}:{i}")
        result = resolve_pa_seeded(batter, pitcher, seed=seed)
        out = result["outcome"]
        counts[out] = counts.get(out, 0) + 1

    total  = sum(counts.values())
    k      = counts.get("K", 0)
    bb     = counts.get("BB", 0)
    hbp    = counts.get("HBP", 0)
    hr     = counts.get("HR", 0)
    triple = counts.get("Triple", 0)
    double = counts.get("Double", 0)
    single = counts.get("Single", 0)
    out    = counts.get("Out", 0) + counts.get("Error", 0)

    hits = single + double + triple + hr
    ab   = total - bb - hbp
    bip  = ab - k - hr
    slg_num = single + 2*double + 3*triple + 4*hr

    # Linear-weight runs (approximate)
    lw_runs = (hr     * _LW["HR"]     +
               triple * _LW["Triple"] +
               double * _LW["Double"] +
               single * _LW["Single"] +
               bb     * _LW["BB"]     +
               hbp    * _LW["HBP"]    +
               out    * _LW["Out"]    +
               k      * _LW["K"])

    # ERA proxy = LW runs per 27 outs × 9 innings
    outs_rec  = k + out + (counts.get("Error", 0) * 0)  # errors still on base, don't count as out
    # Use conventional outs: K + fielded outs (non-error non-hit BIP results)
    outs_rec  = k + counts.get("Out", 0)
    era_proxy = (lw_runs / outs_rec * 27) if outs_rec > 0 else float("nan")

    return {
        "n_pa":    total,
        "k_pct":   round(k  / total, 4),
        "bb_pct":  round(bb / total, 4),
        "hr_pct":  round(hr / total, 4),
        "avg":     round(hits / ab, 3) if ab > 0 else 0.0,
        "obp":     round((hits + bb + hbp) / total, 3),
        "slg":     round(slg_num / ab, 3) if ab > 0 else 0.0,
        "babip":   round((hits - hr) / bip, 3) if bip > 0 else 0.0,
        "lw_runs_per_pa": round(lw_runs / total, 4),
        "era_proxy": round(era_proxy, 2) if math.isfinite(era_proxy) else None,
        "counts":  counts,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(pedro: dict, lineup: list[dict], results: list[dict], n_pa: int):
    pt = pedro["traits"]
    w  = 72

    print(f"\n{'═' * w}")
    print(f"  ENGINE VALIDATION: Pedro Martinez 2000  vs  1927 New York Yankees")
    print(f"  {n_pa:,} PA per batter  |  Global Z-Score Calibration (PRD 02 Fix)")
    print(f"{'─' * w}")
    print(f"  Pedro Martinez 2000 (BOS)  —  SP")
    print(f"    STF {pt['STF']}   CTL {pt['CTL']}   CMD {pt['CMD']}   STA {pt['STA']}")
    print(f"{'─' * w}")
    print(f"  {'Batter':<22}  PA  {'POW':>3} {'EYE':>3} {'CON':>3}  "
          f"{'K%':>6}  {'BB%':>6}  {'HR%':>6}  {'AVG':>5}  {'OPS':>5}  BABIP")
    print(f"{'─' * w}")

    agg_k = agg_bb = agg_hr = agg_hit = agg_ab = agg_slg_n = agg_pa = 0
    agg_obp_n = 0

    for batter, res in zip(lineup, results):
        bt = batter["traits"]
        t  = res["n_pa"]
        cnt = res["counts"]
        hits   = cnt.get("Single",0)+cnt.get("Double",0)+cnt.get("Triple",0)+cnt.get("HR",0)
        ab_b   = t - cnt.get("BB",0) - cnt.get("HBP",0)
        slg_n  = cnt.get("Single",0)+2*cnt.get("Double",0)+3*cnt.get("Triple",0)+4*cnt.get("HR",0)

        agg_k    += cnt.get("K", 0)
        agg_bb   += cnt.get("BB", 0)
        agg_hr   += cnt.get("HR", 0)
        agg_hit  += hits
        agg_ab   += ab_b
        agg_slg_n+= slg_n
        agg_pa   += t
        agg_obp_n+= hits + cnt.get("BB",0) + cnt.get("HBP",0)

        ops = res["obp"] + res["slg"]
        print(f"  {batter['name']:<22}  {t:>3}  "
              f"{bt['POW']:>3} {bt['EYE']:>3} {bt['CON']:>3}  "
              f"{res['k_pct']:>6.3f}  {res['bb_pct']:>6.3f}  {res['hr_pct']:>6.3f}  "
              f"{res['avg']:>5.3f}  {ops:>5.3f}  {res['babip']:>.3f}")

    print(f"{'─' * w}")
    tot_k_pct  = agg_k  / agg_pa
    tot_bb_pct = agg_bb / agg_pa
    tot_hr_pct = agg_hr / agg_pa
    tot_avg    = agg_hit / agg_ab if agg_ab else 0
    tot_obp    = agg_obp_n / agg_pa if agg_pa else 0
    tot_slg    = agg_slg_n / agg_ab if agg_ab else 0
    tot_ops    = tot_obp + tot_slg

    # Aggregate ERA proxy
    agg_lw = sum(r["lw_runs_per_pa"] * r["n_pa"] for r in results)
    agg_outs= agg_k + sum(r["counts"].get("Out",0) for r in results)
    era_agg = (agg_lw / agg_outs * 27) if agg_outs > 0 else float("nan")

    print(f"  {'LINEUP TOTAL':<22}  {agg_pa:>3}  "
          f"{'---':>3} {'---':>3} {'---':>3}  "
          f"{tot_k_pct:>6.3f}  {tot_bb_pct:>6.3f}  {tot_hr_pct:>6.3f}  "
          f"{tot_avg:>5.3f}  {tot_ops:>5.3f}")
    print(f"{'═' * w}")

    print(f"\n  HEADLINE NUMBERS")
    print(f"    Pedro simulated K%   = {tot_k_pct:.1%}")
    print(f"    Pedro simulated BB%  = {tot_bb_pct:.1%}")
    print(f"    Pedro simulated HR%  = {tot_hr_pct:.1%}")
    print(f"    Lineup AVG           = {tot_avg:.3f}")
    print(f"    Lineup OPS           = {tot_ops:.3f}")
    if math.isfinite(era_agg):
        print(f"    ERA proxy            = {era_agg:.2f}")
    else:
        print(f"    ERA proxy            = n/a")

    gs = get_global_stats()
    gp = gs["pitcher"]
    bf_pedro = 855  # actual 2000 BFP
    k_rate_pedro = float(284) / bf_pedro
    sigma_k = (k_rate_pedro - gp["k_rate"]["mean"]) / gp["k_rate"]["std"]
    print(f"\n  Pedro 2000 K% sigma vs global mean:  {sigma_k:+.2f}σ  (STF={pt['STF']})")
    print(f"  A {sigma_k:+.1f}σ pitcher should strike out "
          f"~{int(round(tot_k_pct*100))}% of a +2σ lineup's plate appearances.")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of PAs per batter (default: 1000)")
    args = parser.parse_args()

    adapter = LahmanAdapter()
    people  = load_people()

    print("\n  Building Pedro Martinez 2000 card ...")
    pedro = build_pedro_card(people, adapter)
    pt = pedro["traits"]
    print(f"    STF={pt['STF']}  CTL={pt['CTL']}  CMD={pt['CMD']}  STA={pt['STA']}")

    print("  Building 1927 Yankees lineup ...")
    lineup = build_yankees_lineup(people, adapter)
    print(f"    {len(lineup)} hitters found with PA ≥ 200")

    print(f"  Simulating {args.n:,} PAs per batter ...")
    results = []
    for batter in lineup:
        res = run_matchup(batter, pedro, args.n)
        results.append(res)
        bt = batter["traits"]
        print(f"    {batter['name']:<22}  "
              f"POW={bt['POW']} EYE={bt['EYE']} CON={bt['CON']}  "
              f"K%={res['k_pct']:.3f}  BB%={res['bb_pct']:.3f}  "
              f"HR%={res['hr_pct']:.3f}  AVG={res['avg']:.3f}")

    print_report(pedro, lineup, results, args.n)


if __name__ == "__main__":
    main()
