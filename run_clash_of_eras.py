"""
run_clash_of_eras.py — Clash of Eras stress test.

Matchup A: 1913 Philadelphia Athletics pitchers vs 2019 Washington Nationals batters
Matchup B: 2019 Washington Nationals pitchers vs 1913 Philadelphia Athletics batters

Rules:
  - Top 5 pitchers by IPouts per team/year.
  - Top 9 batters by PA per team/year, pitchers excluded.
  - 50,000 total PAs per matchup, distributed proportionally:
      n_pa(batter_i, pitcher_j) = 50000 × (PA_i / ΣPA) × (IPouts_j / ΣIPouts)
  - Frozen PA engine — no calibration constants touched.
  - Neutral fielder (50/50/50) — cross-era fielding cards not available.

Usage:
    python run_clash_of_eras.py [--n 50000]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import LahmanAdapter, normalize_player_stats, normalize_pitcher_stats
from ratings import build_hitter_card, build_pitcher_card
from pa_wrapper import resolve_pa_seeded

LAHMAN_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lahman_1871-2025_csv")
DEF_FIELDER = {"RNG": 50, "HND": 50, "ARM": 50}
N_BATTERS   = 9
N_PITCHERS  = 5


# ── Seeding ───────────────────────────────────────────────────────────────────

def _sha32(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 32)


# ── Roster selection ──────────────────────────────────────────────────────────

def _people_lookup(adapter: LahmanAdapter) -> dict[str, dict]:
    """Returns {playerID: {name, bats, throws}} for quick reference."""
    df = adapter.load_people()
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        out[row["playerID"]] = {
            "name":   f"{row['nameFirst']} {row['nameLast']}".strip(),
            "bats":   str(row.get("bats",   "R")),
            "throws": str(row.get("throws", "R")),
        }
    return out


def _top_pitchers(
    adapter: LahmanAdapter,
    team_id: str,
    year: int,
    n: int,
) -> pd.DataFrame:
    """Top N pitchers by IPouts for team/year. Returns rows with playerID, IPouts."""
    df = adapter.load_pitching(team_id, year)
    if df.empty:
        raise ValueError(f"No pitching data for {team_id} {year}")
    df["IPouts"] = pd.to_numeric(df["IPouts"], errors="coerce").fillna(0)
    # Aggregate stints if a player appears multiple times for this team
    df = df.groupby("playerID", as_index=False)["IPouts"].sum()
    df = df.sort_values("IPouts", ascending=False).head(n).reset_index(drop=True)
    return df


def _top_batters(
    adapter: LahmanAdapter,
    team_id: str,
    year: int,
    pitcher_ids: set[str],
    n: int,
) -> pd.DataFrame:
    """
    Top N batters by PA for team/year, excluding playerIDs in pitcher_ids.
    PA = AB + BB + HBP + SF (SF filled to 0 for dead-ball eras).
    Returns rows with playerID, PA.
    """
    df = adapter.load_batting(team_id, year)
    if df.empty:
        raise ValueError(f"No batting data for {team_id} {year}")
    for col in ("AB", "BB", "HBP", "SF"):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["PA"] = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    # Aggregate stints, then filter
    df = df.groupby("playerID", as_index=False)["PA"].sum()
    df = df[~df["playerID"].isin(pitcher_ids)]
    df = df[df["PA"] > 0]
    df = df.sort_values("PA", ascending=False).head(n).reset_index(drop=True)
    return df


# ── Card builders (wrapping core_stress_test helpers) ────────────────────────

def _build_pitcher_card(pid: str, year: int, adapter: LahmanAdapter, people: dict) -> dict:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "Pitching.csv"))
    df = df[(df["playerID"] == pid) & (df["yearID"] == year)].copy()
    agg_cols = [c for c in ["G","GS","H","HR","BB","SO","HBP","IPouts","BFP","SF","SH","ER"]
                if c in df.columns]
    df = df.groupby(["playerID","yearID"], as_index=False)[agg_cols].sum()
    if df.empty:
        raise ValueError(f"No pitching data for {pid} {year}")
    lg = adapter.load_league_context(year)
    normed = normalize_pitcher_stats(df, lg).iloc[0]
    p = people.get(pid, {})
    card = build_pitcher_card(normed, p.get("name", pid), p.get("bats", "R"), p.get("throws", "R"))
    return {
        "player_id":    card.player_id,
        "name":         card.name,
        "season":       card.season,
        "bats":         card.bats,
        "throws":       card.throws,
        "primary_role": "Pitcher",
        "pitcher_role": card.pitcher_role,
        "traits":       {"STF": card.STF, "CTL": card.CTL, "CMD": card.CMD, "STA": card.STA},
    }


def _build_hitter_card(pid: str, year: int, adapter: LahmanAdapter, people: dict) -> dict:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "Batting.csv"))
    df = df[(df["playerID"] == pid) & (df["yearID"] == year)].copy()
    agg_cols = [c for c in ["AB","H","2B","3B","HR","BB","SO","HBP","SF","SH","SB","CS"]
                if c in df.columns]
    df = df.groupby(["playerID","yearID"], as_index=False)[agg_cols].sum()
    for col in ("SF","HBP"):
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["PA"]  = df.get("AB", 0) + df.get("BB", 0) + df.get("HBP", 0) + df.get("SF", 0)
    df["BIP"] = df.get("AB", 0) - df.get("SO", 0) - df.get("HR", 0) + df.get("SF", 0)
    if df.empty:
        raise ValueError(f"No batting data for {pid} {year}")
    lg = adapter.load_league_context(year)
    normed = normalize_player_stats(df, lg).iloc[0]
    p = people.get(pid, {})
    card = build_hitter_card(normed, p.get("name", pid), p.get("bats", "R"), p.get("throws", "R"))
    return {
        "player_id":    card.player_id,
        "name":         card.name,
        "season":       card.season,
        "bats":         card.bats,
        "throws":       card.throws,
        "primary_role": "Hitter",
        "traits":       {"CON": card.CON, "GAP": card.GAP, "POW": card.POW,
                         "EYE": card.EYE, "AK": card.AK},
    }


# ── PA simulation ─────────────────────────────────────────────────────────────

@dataclass
class SlashStats:
    pa:      int = 0
    k:       int = 0
    bb:      int = 0
    hr:      int = 0
    single:  int = 0
    double:  int = 0
    triple:  int = 0
    error:   int = 0
    bip:     int = 0
    hip:     int = 0  # H on BIP (excl. HR)

    def record(self, outcome: str, is_bip: bool) -> None:
        self.pa += 1
        if outcome == "K":
            self.k += 1
        elif outcome == "BB":
            self.bb += 1
        elif outcome == "HR":
            self.hr += 1
            self.bip += 1
        elif outcome in ("Single", "Double", "Triple", "Error", "Out"):
            self.bip += 1
            if outcome == "Single":
                self.single += 1; self.hip += 1
            elif outcome == "Double":
                self.double += 1; self.hip += 1
            elif outcome == "Triple":
                self.triple += 1; self.hip += 1

    def add(self, other: "SlashStats") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @property
    def k_pct(self)    -> float: return self.k  / self.pa if self.pa else 0.0
    @property
    def bb_pct(self)   -> float: return self.bb / self.pa if self.pa else 0.0
    @property
    def hr_pa(self)    -> float: return self.hr / self.pa if self.pa else 0.0
    @property
    def babip(self)    -> float:
        d = self.bip - self.hr
        return self.hip / d if d > 0 else 0.0


def simulate_pair(
    batter:  dict,
    pitcher: dict,
    n_pa:    int,
    tag:     str,
) -> SlashStats:
    if n_pa == 0:
        return SlashStats()
    base    = _sha32(f"{tag}|{batter['player_id']}|{pitcher['player_id']}")
    context = {"fielder": DEF_FIELDER}
    stats   = SlashStats()
    for i in range(n_pa):
        seed   = _sha32(f"{base}:{i}")
        result = resolve_pa_seeded(batter, pitcher, context=context, seed=seed)
        is_bip = result["contact"] is not None
        stats.record(result["outcome"], is_bip)
    return stats


# ── Matchup runner ────────────────────────────────────────────────────────────

def run_matchup(
    label:        str,
    batter_cards: list[dict],
    batter_pas:   list[float],
    pitcher_cards: list[dict],
    pitcher_ipouts: list[float],
    total_n:      int,
) -> None:
    total_pa     = sum(batter_pas)
    total_ipouts = sum(pitcher_ipouts)

    # per-pitcher aggregate stats
    pit_stats: dict[str, SlashStats] = {p["player_id"]: SlashStats() for p in pitcher_cards}
    team_stats = SlashStats()

    print(f"\n  {'─'*60}")
    print(f"  Simulating: {label}")

    for b_card, b_pa in zip(batter_cards, batter_pas):
        b_weight = b_pa / total_pa
        for p_card, p_ipouts in zip(pitcher_cards, pitcher_ipouts):
            p_weight = p_ipouts / total_ipouts
            n_pa = max(1, round(total_n * b_weight * p_weight))
            tag  = label.replace(" ", "_")
            s = simulate_pair(b_card, p_card, n_pa, tag)
            pit_stats[p_card["player_id"]].add(s)
            team_stats.add(s)

    # ── Per-pitcher lines ─────────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  {'Pitcher':<26} {'IP%':>5}  {'K%':>5}  {'BB%':>5}  {'HR/PA':>6}  {'BABIP':>5}  {'PA':>6}")
    print(f"  {'─'*26}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*6}")

    for p_card, p_ipouts in zip(pitcher_cards, pitcher_ipouts):
        pid  = p_card["player_id"]
        ip_pct = p_ipouts / total_ipouts * 100
        s    = pit_stats[pid]
        t    = p_card["traits"]
        print(
            f"  {p_card['name']:<26} {ip_pct:>4.1f}%"
            f"  {s.k_pct:>5.3f}  {s.bb_pct:>5.3f}"
            f"  {s.hr_pa:>6.4f}  {s.babip:>5.3f}  {s.pa:>6,}"
        )
        print(
            f"    traits: STF={t['STF']} CTL={t['CTL']} CMD={t['CMD']} STA={t['STA']}"
        )

    # ── Team aggregate ────────────────────────────────────────────────────────
    s = team_stats
    print(f"  {'─'*60}")
    print(
        f"  TEAM TOTAL  PA={s.pa:,}"
        f"  K%={s.k_pct:.3f}  BB%={s.bb_pct:.3f}"
        f"  HR/PA={s.hr_pa:.4f}  BABIP={s.babip:.3f}"
    )
    print(f"  {'─'*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Clash of Eras — 1913 PHA vs 2019 WAS")
    parser.add_argument("--n", type=int, default=50_000, help="Total PAs per matchup")
    args = parser.parse_args()

    adapter = LahmanAdapter()
    people  = _people_lookup(adapter)

    print(f"\n{'═'*64}")
    print(f"  CLASH OF ERAS — {args.n:,} PAs per matchup")
    print(f"  1913 Philadelphia Athletics  vs  2019 Washington Nationals")
    print(f"{'═'*64}")

    # ── Build rosters ─────────────────────────────────────────────────────────

    print("\n  Building 1913 PHA roster...")
    pha_pit_df = _top_pitchers(adapter, "PHA", 1913, N_PITCHERS)
    pha_pit_ids = set(pha_pit_df["playerID"])
    pha_bat_df  = _top_batters(adapter, "PHA", 1913, pha_pit_ids, N_BATTERS)

    print("\n  Building 2019 WAS roster...")
    was_pit_df = _top_pitchers(adapter, "WAS", 2019, N_PITCHERS)
    was_pit_ids = set(was_pit_df["playerID"])
    was_bat_df  = _top_batters(adapter, "WAS", 2019, was_pit_ids, N_BATTERS)

    # ── Print selected rosters ────────────────────────────────────────────────

    print("\n  ┌─ 1913 Philadelphia Athletics — Pitchers (top 5 by IP)")
    for _, row in pha_pit_df.iterrows():
        ip = row["IPouts"] / 3.0
        print(f"  │  {people.get(row['playerID'], {}).get('name', row['playerID']):<28} {ip:>6.1f} IP")

    print("  │")
    print("  ├─ 1913 Philadelphia Athletics — Batters (top 9 by PA, excl. pitchers)")
    for _, row in pha_bat_df.iterrows():
        print(f"  │  {people.get(row['playerID'], {}).get('name', row['playerID']):<28} {int(row['PA']):>4} PA")

    print("  │")
    print("  ├─ 2019 Washington Nationals — Pitchers (top 5 by IP)")
    for _, row in was_pit_df.iterrows():
        ip = row["IPouts"] / 3.0
        print(f"  │  {people.get(row['playerID'], {}).get('name', row['playerID']):<28} {ip:>6.1f} IP")

    print("  │")
    print("  ├─ 2019 Washington Nationals — Batters (top 9 by PA, excl. pitchers)")
    for _, row in was_bat_df.iterrows():
        print(f"  │  {people.get(row['playerID'], {}).get('name', row['playerID']):<28} {int(row['PA']):>4} PA")
    print("  └" + "─" * 60)

    # ── Build trait cards ─────────────────────────────────────────────────────

    print("\n  Building pitcher cards...")

    pha_pit_cards, pha_pit_ipouts = [], []
    for _, row in pha_pit_df.iterrows():
        pid = row["playerID"]
        try:
            card = _build_pitcher_card(pid, 1913, adapter, people)
            pha_pit_cards.append(card)
            pha_pit_ipouts.append(float(row["IPouts"]))
            t = card["traits"]
            print(f"    PHA {card['name']:<26} STF={t['STF']} CTL={t['CTL']} CMD={t['CMD']}")
        except Exception as e:
            print(f"    ⚠  PHA pitcher {pid} skipped: {e}")

    was_pit_cards, was_pit_ipouts = [], []
    for _, row in was_pit_df.iterrows():
        pid = row["playerID"]
        try:
            card = _build_pitcher_card(pid, 2019, adapter, people)
            was_pit_cards.append(card)
            was_pit_ipouts.append(float(row["IPouts"]))
            t = card["traits"]
            print(f"    WAS {card['name']:<26} STF={t['STF']} CTL={t['CTL']} CMD={t['CMD']}")
        except Exception as e:
            print(f"    ⚠  WAS pitcher {pid} skipped: {e}")

    print("\n  Building batter cards...")

    pha_bat_cards, pha_bat_pas = [], []
    for _, row in pha_bat_df.iterrows():
        pid = row["playerID"]
        try:
            card = _build_hitter_card(pid, 1913, adapter, people)
            pha_bat_cards.append(card)
            pha_bat_pas.append(float(row["PA"]))
            t = card["traits"]
            print(f"    PHA {card['name']:<26} CON={t['CON']} POW={t['POW']} EYE={t['EYE']}")
        except Exception as e:
            print(f"    ⚠  PHA batter {pid} skipped: {e}")

    was_bat_cards, was_bat_pas = [], []
    for _, row in was_bat_df.iterrows():
        pid = row["playerID"]
        try:
            card = _build_hitter_card(pid, 2019, adapter, people)
            was_bat_cards.append(card)
            was_bat_pas.append(float(row["PA"]))
            t = card["traits"]
            print(f"    WAS {card['name']:<26} CON={t['CON']} POW={t['POW']} EYE={t['EYE']}")
        except Exception as e:
            print(f"    ⚠  WAS batter {pid} skipped: {e}")

    # ── Run matchups ──────────────────────────────────────────────────────────

    print(f"\n\n{'═'*64}")
    print(f"  MATCHUP A: 1913 PHA pitching  vs  2019 WAS batting")
    print(f"{'═'*64}")
    run_matchup(
        "1913 PHA pitching vs 2019 WAS batting",
        was_bat_cards, was_bat_pas,
        pha_pit_cards, pha_pit_ipouts,
        args.n,
    )

    print(f"\n\n{'═'*64}")
    print(f"  MATCHUP B: 2019 WAS pitching  vs  1913 PHA batting")
    print(f"{'═'*64}")
    run_matchup(
        "2019 WAS pitching vs 1913 PHA batting",
        pha_bat_cards, pha_bat_pas,
        was_pit_cards, was_pit_ipouts,
        args.n,
    )

    print(f"\n{'═'*64}\n")


if __name__ == "__main__":
    main()
