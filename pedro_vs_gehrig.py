"""
pedro_vs_gehrig.py — Pedro Martinez 2000 vs Lou Gehrig 1927

10,000 simulated plate appearances.

Reports:
  PA count
  K%      BB%      HR%      BABIP (hits-in-play / balls-in-play)
  1B%     2B%      3B%      HBP%

Usage:
  python3 pedro_vs_gehrig.py [--n 10000]
"""

import argparse
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAHMAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lahman_1871-2025_csv")

from ingestion import LahmanAdapter, normalize_pitcher_stats, normalize_player_stats
from ratings import build_hitter_card, build_pitcher_card
from pa_wrapper import resolve_pa_seeded


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_pitching_row(player_id: str, year: int) -> pd.Series:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "Pitching.csv"))
    df = df[(df["playerID"] == player_id) & (df["yearID"] == year)].copy()
    agg = [c for c in ["G","GS","H","HR","BB","SO","HBP","IPouts","BFP","SF","SH","ER"]
           if c in df.columns]
    df = df.groupby(["playerID","yearID"], as_index=False)[agg].sum()
    return df.iloc[0]


def _load_batting_row(player_id: str, year: int) -> pd.Series:
    df = pd.read_csv(os.path.join(LAHMAN_DIR, "Batting.csv"))
    df = df[(df["playerID"] == player_id) & (df["yearID"] == year)].copy()
    agg = [c for c in ["AB","H","2B","3B","HR","BB","SO","HBP","SF","SH","SB","CS"]
           if c in df.columns]
    df = df.groupby(["playerID","yearID"], as_index=False)[agg].sum()
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


# ── Card builders ──────────────────────────────────────────────────────────────

def build_card_pitcher(player_id: str, year: int, adapter: LahmanAdapter) -> dict:
    p_row  = pd.DataFrame([_load_pitching_row(player_id, year)])
    lg     = adapter.load_league_context(year)
    normed = normalize_pitcher_stats(p_row, lg).iloc[0]
    people = _load_people(player_id)
    card   = build_pitcher_card(normed, people["name"], people["bats"],
                                people["throws"])
    return {
        "player_id":    card.player_id,
        "name":         card.name,
        "season":       card.season,
        "bats":         card.bats,
        "throws":       card.throws,
        "primary_role": "Pitcher",
        "pitcher_role": card.pitcher_role,
        "traits": {"STF": card.STF, "CTL": card.CTL,
                   "CMD": card.CMD, "STA": card.STA},
    }


def build_card_hitter(player_id: str, year: int, adapter: LahmanAdapter) -> dict:
    b_row  = pd.DataFrame([_load_batting_row(player_id, year)])
    lg     = adapter.load_league_context(year)
    normed = normalize_player_stats(b_row, lg).iloc[0]
    people = _load_people(player_id)
    card   = build_hitter_card(normed, people["name"], people["bats"],
                               people["throws"])
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


# ── Simulation ────────────────────────────────────────────────────────────────

def _sha32(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 32)


def simulate(batter: dict, pitcher: dict, n_pa: int, label: str = "") -> dict:
    base = _sha32(f"{batter['player_id']}|{pitcher['player_id']}|{label}")
    counts: dict[str, int] = {}
    for i in range(n_pa):
        seed   = _sha32(f"{base}:{i}")
        result = resolve_pa_seeded(batter, pitcher, seed=seed)
        out    = result["outcome"]
        counts[out] = counts.get(out, 0) + 1
    return counts


def aggregate(counts: dict, n_pa: int) -> dict:
    k      = counts.get("K",      0)
    bb     = counts.get("BB",     0)
    hbp    = counts.get("HBP",    0)
    hr     = counts.get("HR",     0)
    triple = counts.get("Triple", 0)
    double = counts.get("Double", 0)
    single = counts.get("Single", 0)

    ab  = n_pa - bb - hbp
    bip = ab - k - hr
    hits_in_play = single + double + triple   # hits on balls in play (no HR)

    return {
        "PA":    n_pa,
        "K":     k,   "BB":  bb,  "HBP":    hbp,
        "HR":    hr,  "3B":  triple, "2B":  double, "1B":  single,
        "AB":    ab,  "BIP": bip,
        "hits_in_play": hits_in_play,
        # rates
        "K%":   k      / n_pa,
        "BB%":  bb     / n_pa,
        "HBP%": hbp    / n_pa,
        "HR%":  hr     / n_pa,
        "3B%":  triple / n_pa,
        "2B%":  double / n_pa,
        "1B%":  single / n_pa,
        "BABIP": hits_in_play / bip if bip > 0 else 0.0,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 30) -> str:
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)


def print_report(pitcher: dict, hitter: dict, stats: dict, n_pa: int):
    pt = pitcher["traits"]
    ht = hitter["traits"]
    sep = "═" * 70

    print(f"\n{sep}")
    print(f"  MATCHUP AUDIT — {n_pa:,} Plate Appearances")
    print(f"{'─'*70}")
    print(f"  PITCHER  {pitcher['name']} {pitcher['season']}"
          f"  ({pitcher['throws']}HP)")
    print(f"           STF {pt['STF']}  CTL {pt['CTL']}  CMD {pt['CMD']}  STA {pt['STA']}")
    print(f"  BATTER   {hitter['name']} {hitter['season']}"
          f"  (bats {hitter['bats']})")
    print(f"           CON {ht['CON']}  POW {ht['POW']}  EYE {ht['EYE']}"
          f"  AK {ht['AK']}  GAP {ht['GAP']}")
    print(f"{'─'*70}")
    print()

    rows = [
        ("PA",    stats["PA"],    None,          None),
        ("K",     stats["K"],     stats["K%"],   "K%"),
        ("BB",    stats["BB"],    stats["BB%"],  "BB%"),
        ("HBP",   stats["HBP"],   stats["HBP%"], "HBP%"),
        ("HR",    stats["HR"],    stats["HR%"],  "HR%"),
        ("1B",    stats["1B"],    stats["1B%"],  "1B%"),
        ("2B",    stats["2B"],    stats["2B%"],  "2B%"),
        ("3B",    stats["3B"],    stats["3B%"],  "3B%"),
        ("BABIP", None,           stats["BABIP"],"BABIP"),
    ]

    print(f"  {'Outcome':<8}  {'Count':>6}  {'Rate':>7}  Bar (1 block = ~3.3%)")
    print(f"  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*30}")
    for label, count, rate, rate_lbl in rows:
        count_str = f"{count:>6}" if count is not None else " " * 6
        rate_str  = f"{rate:>7.3f}" if rate  is not None else " " * 7
        bar_str   = _bar(rate, 30) if rate is not None else ""
        print(f"  {label:<8}  {count_str}  {rate_str}  {bar_str}")

    print()
    print(f"  {'BIP (balls in play)':>30}  {stats['BIP']:>6}")
    print(f"  {'Hits in play (1B+2B+3B)':>30}  {stats['hits_in_play']:>6}")
    bip_out = stats["BIP"] - stats["hits_in_play"]
    print(f"  {'BIP outs':>30}  {bip_out:>6}")
    print()

    # Real-world reference
    print(f"{'─'*70}")
    print(f"  REAL-WORLD REFERENCE  (Pedro 2000 actual season vs MLB batters)")
    real = {
        "K%":    0.348,  "BB%":  0.039,  "HBP%": 0.007,
        "HR%":   0.021,  "BABIP":0.237,
    }
    for lbl, val in real.items():
        sim_val = stats.get(lbl, 0.0)
        delta   = sim_val - val
        print(f"  {lbl:<8}  sim={sim_val:.3f}  real={val:.3f}  "
              f"delta={delta:+.3f}  {'▲ higher' if delta > 0 else '▼ lower'}")

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000,
                        help="Number of PAs to simulate (default: 10,000)")
    args = parser.parse_args()

    adapter = LahmanAdapter()

    print(f"\n  Building Pedro Martinez 2000 ...")
    pedro = build_card_pitcher("martipe02", 2000, adapter)
    pt    = pedro["traits"]
    print(f"    STF={pt['STF']}  CTL={pt['CTL']}  CMD={pt['CMD']}  STA={pt['STA']}")

    print(f"  Building Lou Gehrig 1927 ...")
    gehrig = build_card_hitter("gehrilo01", 1927, adapter)
    ht     = gehrig["traits"]
    print(f"    CON={ht['CON']}  POW={ht['POW']}  EYE={ht['EYE']}  "
          f"AK={ht['AK']}  GAP={ht['GAP']}")

    print(f"\n  Simulating {args.n:,} plate appearances ...")
    counts = simulate(gehrig, pedro, args.n, label="pedro-gehrig-audit")
    stats  = aggregate(counts, args.n)

    print_report(pedro, gehrig, stats, args.n)


if __name__ == "__main__":
    main()
