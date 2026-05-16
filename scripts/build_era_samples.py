"""
scripts/build_era_samples.py — Build PlayerCard era pools from Lahman CSVs.

For each of four target seasons, aggregates all stints, normalises stats
against that season's league averages, builds PlayerCard objects using the
repo's canonical ratings pipeline, and saves to data/era_pools/<year>.json.

Usage:
    python scripts/build_era_samples.py

Output:
    data/era_pools/1906.json
    data/era_pools/1927.json
    data/era_pools/1968.json
    data/era_pools/1999.json
"""

import os
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from ingestion import (
    LahmanAdapter,
    normalize_player_stats,
    normalize_pitcher_stats,
    LAHMAN_DIR,
)
from ratings import build_hitter_card, build_pitcher_card, export_to_json

# ── Config ────────────────────────────────────────────────────────────────────
YEARS        = [1906, 1927, 1968, 1999]
TOP_BATTERS  = 30
TOP_PITCHERS = 20
MIN_PA       = 100
MIN_BFP      = 30
OUT_DIR      = os.path.join(_REPO, "data", "era_pools")

# ── Load full Lahman CSVs once ────────────────────────────────────────────────
_BAT_COLS = ["playerID", "yearID", "G", "AB", "H", "2B", "3B", "HR",
             "BB", "SO", "HBP", "SF"]
_PIT_COLS = ["playerID", "yearID", "G", "GS", "IPouts", "BFP",
             "H", "HR", "BB", "SO", "HBP"]

print("[build] Loading Lahman CSVs …")
_bat_raw  = pd.read_csv(os.path.join(LAHMAN_DIR, "Batting.csv"),
                        usecols=lambda c: c in _BAT_COLS + ["stint", "teamID"])
_pit_raw  = pd.read_csv(os.path.join(LAHMAN_DIR, "Pitching.csv"),
                        usecols=lambda c: c in _PIT_COLS + ["stint", "teamID"])
_people   = pd.read_csv(os.path.join(LAHMAN_DIR, "People.csv"),
                        usecols=["playerID", "nameFirst", "nameLast", "bats", "throws"])

adapter   = LahmanAdapter()
os.makedirs(OUT_DIR, exist_ok=True)


def _dominant_team(sub: pd.DataFrame, ab_col: str = "AB") -> str:
    """Return teamID with highest counting stat (best proxy for primary team)."""
    if "teamID" not in sub.columns or sub.empty:
        return "UNK"
    return sub.groupby("teamID")[ab_col].sum().idxmax()


def build_year(year: int) -> list:
    print(f"\n[build] ── {year} ──────────────────────────────────────────")
    lg = adapter.load_league_context(year)
    print(f"         lg K%={lg['lg_avg_k_rate']:.3f}  BB%={lg['lg_avg_bb_rate']:.3f}  "
          f"HR%={lg['lg_avg_hr_rate']:.3f}  BABIP={lg['lg_avg_babip']:.3f}")

    # ── Batters ───────────────────────────────────────────────────────────────
    bat_yr = _bat_raw[_bat_raw["yearID"] == year].copy()

    # Primary team per player (most AB)
    primary_team = (
        bat_yr.groupby("playerID")
        .apply(lambda g: _dominant_team(g, "AB"), include_groups=False)
        .rename("primary_team")
    )

    # Aggregate stints → season totals
    agg_bat = (
        bat_yr.groupby(["playerID"])[["AB", "H", "2B", "3B", "HR", "BB",
                                      "SO", "HBP", "SF"]]
        .sum(min_count=0)
        .fillna(0)
        .reset_index()
    )
    agg_bat["yearID"]  = year
    agg_bat["HBP"]     = agg_bat["HBP"].fillna(0)
    agg_bat["SF"]      = agg_bat["SF"].fillna(0)
    agg_bat["PA"]      = agg_bat["AB"] + agg_bat["BB"] + agg_bat["HBP"] + agg_bat["SF"]
    agg_bat = agg_bat.join(primary_team, on="playerID")

    # Merge people (bats/throws/name)
    agg_bat = agg_bat.merge(_people, on="playerID", how="left")
    agg_bat["name"]   = (agg_bat["nameFirst"].fillna("") + " " +
                          agg_bat["nameLast"].fillna("")).str.strip()
    agg_bat["bats"]   = agg_bat["bats"].fillna("R")
    agg_bat["throws"] = agg_bat["throws"].fillna("R")

    # Filter & sort
    agg_bat = agg_bat[agg_bat["PA"] >= MIN_PA].sort_values("PA", ascending=False)
    top_bat = agg_bat.head(TOP_BATTERS).reset_index(drop=True)

    normed_bat = normalize_player_stats(top_bat, lg)

    hitter_cards = []
    for _, row in normed_bat.iterrows():
        card = build_hitter_card(
            row, row["name"], str(row["bats"]), str(row["throws"]),
            team_id=str(row.get("primary_team", "UNK")),
        )
        hitter_cards.append(card)

    # ── Pitchers ──────────────────────────────────────────────────────────────
    pit_yr = _pit_raw[_pit_raw["yearID"] == year].copy()

    primary_pit_team = (
        pit_yr.groupby("playerID")
        .apply(lambda g: _dominant_team(g, "IPouts"), include_groups=False)
        .rename("primary_team")
    )

    agg_pit = (
        pit_yr.groupby(["playerID"])[["G", "GS", "IPouts", "BFP",
                                      "H", "HR", "BB", "SO", "HBP"]]
        .sum(min_count=0)
        .fillna(0)
        .reset_index()
    )
    agg_pit["yearID"] = year
    agg_pit["BFP"]    = agg_pit["BFP"].fillna(0)
    agg_pit["HBP"]    = agg_pit["HBP"].fillna(0)
    agg_pit = agg_pit.join(primary_pit_team, on="playerID")

    agg_pit = agg_pit.merge(_people, on="playerID", how="left")
    agg_pit["name"]   = (agg_pit["nameFirst"].fillna("") + " " +
                          agg_pit["nameLast"].fillna("")).str.strip()
    agg_pit["bats"]   = agg_pit["bats"].fillna("R")
    agg_pit["throws"] = agg_pit["throws"].fillna("R")

    agg_pit = agg_pit[agg_pit["BFP"] >= MIN_BFP].sort_values("BFP", ascending=False)
    top_pit = agg_pit.head(TOP_PITCHERS).reset_index(drop=True)

    normed_pit = normalize_pitcher_stats(top_pit, lg)

    pitcher_cards = []
    for _, row in normed_pit.iterrows():
        card = build_pitcher_card(
            row, row["name"], str(row["bats"]), str(row["throws"]),
            team_id=str(row.get("primary_team", "UNK")),
        )
        pitcher_cards.append(card)

    all_cards = hitter_cards + pitcher_cards

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"         Hitters : {len(hitter_cards):>3}  "
          f"(PA range {int(normed_bat['PA'].max())}–{int(normed_bat['PA'].min())})")
    print(f"         Pitchers: {len(pitcher_cards):>3}  "
          f"(BFP range {int(top_pit['BFP'].max())}–{int(top_pit['BFP'].min())})")

    # Print top 5 hitters / pitchers for verification
    print(f"\n         Top 5 Hitters by PA:")
    print(f"         {'Name':<24} PA   CON  POW  EYE   AK  GAP")
    for c in hitter_cards[:5]:
        row = normed_bat[normed_bat["playerID"] == c.player_id]
        pa  = int(row["PA"].values[0]) if not row.empty else 0
        print(f"         {c.name:<24} {pa:>4}  "
              f"{c.CON:>3}  {c.POW:>3}  {c.EYE:>3}  {c.AK:>3}  {c.GAP:>3}")

    print(f"\n         Top 5 Pitchers by BFP:")
    print(f"         {'Name':<24} BFP  STF  CTL  CMD  STA  Role")
    for c in pitcher_cards[:5]:
        row = top_pit[top_pit["playerID"] == c.player_id]
        bfp = int(row["BFP"].values[0]) if not row.empty else 0
        print(f"         {c.name:<24} {bfp:>4}  "
              f"{c.STF:>3}  {c.CTL:>3}  {c.CMD:>3}  {c.STA:>3}  {c.pitcher_role}")

    return all_cards


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for year in YEARS:
        cards = build_year(year)
        out_path = os.path.join(OUT_DIR, f"{year}.json")
        export_to_json(cards, out_path)

    print(f"\n[build] Done. Pools written to {OUT_DIR}/")
