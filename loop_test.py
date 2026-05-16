"""
Simulates 1,000 plate appearances between two players stored in Supabase
and prints an outcome summary + slash line.
"""

import random
from collections import Counter

from database import supabase
from pa_engine import (
    resolve_duel,
    resolve_contact,
    map_bip_outcome,
    resolve_defense,
    derive_game_seed,
    derive_pa_seed,
)

import json

with open("sim_constants.json") as f:
    SIM_CONSTANTS = json.load(f)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATTER_NAME   = "Babe Ruth"
BATTER_SEASON = 1927
PITCHER_NAME  = "Pedro Martinez"
PITCHER_SEASON = 2000
HANDEDNESS    = {"batter": "L", "pitcher": "R"}
FIELDER       = {"RNG": 50.0, "HND": 50.0, "ARM": 50.0}

SIM_SEED  = "loop-test-ruth-vs-pedro"
GAME_ID   = "game-1"
NUM_PA    = 1_000


# ---------------------------------------------------------------------------
# Player lookup + trait mapping
# ---------------------------------------------------------------------------

def fetch_player(name: str, season: int) -> dict:
    response = (
        supabase.table("players")
        .select("*")
        .eq("player_name", name)
        .eq("season_year", season)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise ValueError(f"Player not found: '{name}' ({season})")
    return response.data[0]


def map_batter(row: dict) -> dict:
    return {
        "CON": float(row["contact"]),
        "GAP": float(row["power"]),
        "POW": float(row["power"]),
        "EYE": float(row["eye"]),
        "AK":  float(row["contact"]),
        "BNT": 50.0,
    }


def map_pitcher(row: dict) -> dict:
    return {
        "STF": float(row["stuff"]),
        "CTL": float(row["control"]),
        "CMD": float(row["movement"]),
        "STA": 80.0,
    }


# ---------------------------------------------------------------------------
# Single PA
# ---------------------------------------------------------------------------

def run_pa(batter: dict, pitcher: dict, game_seed: int, pa_index: int) -> str:
    pa_seed = derive_pa_seed(game_seed, pa_index)
    rng = random.Random(pa_seed)

    duel = resolve_duel(
        batter=batter,
        pitcher=pitcher,
        handedness=HANDEDNESS,
        constants=SIM_CONSTANTS,
        rng=rng,
    )

    if duel["outcome"] != "BIP":
        return duel["outcome"]  # K, BB, or HBP

    contact = resolve_contact(
        batter=batter, pitcher=pitcher, constants=SIM_CONSTANTS, rng=rng
    )
    bip_mapping = map_bip_outcome(
        contact_quality=contact["contact_quality"],
        contact_score=contact["contact_score"],
        spray_vector=contact["spray_vector"],
        effective_pow=contact["effective_pow"],
        batter=batter,
        pitcher=pitcher,
        constants=SIM_CONSTANTS,
        rng=rng,
    )
    defense = resolve_defense(
        bip_outcome=bip_mapping["bip_outcome"],
        spray_vector=contact["spray_vector"],
        contact_quality=contact["contact_quality"],
        fielder=FIELDER,
        constants=SIM_CONSTANTS,
        rng=rng,
    )
    return defense["final_outcome"]


# ---------------------------------------------------------------------------
# Slash line math
# ---------------------------------------------------------------------------

def slash_line(counts: Counter) -> tuple[float, float, float]:
    singles  = counts["Single"]
    doubles  = counts["Double"]
    triples  = counts["Triple"]
    hrs      = counts["HR"]
    bbs      = counts["BB"]
    hbps     = counts["HBP"]
    outs     = counts["Out"]

    hits = singles + doubles + triples + hrs
    ab   = hits + outs                      # BB and HBP excluded from AB
    pa   = ab + bbs + hbps

    avg  = hits / ab if ab else 0.0
    obp  = (hits + bbs + hbps) / pa if pa else 0.0
    slg  = (singles + 2*doubles + 3*triples + 4*hrs) / ab if ab else 0.0

    return avg, obp, slg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Fetching players from Supabase...")
    batter_row  = fetch_player(BATTER_NAME,  BATTER_SEASON)
    pitcher_row = fetch_player(PITCHER_NAME, PITCHER_SEASON)

    batter  = map_batter(batter_row)
    pitcher = map_pitcher(pitcher_row)

    print(f"Simulating {NUM_PA:,} PAs: {BATTER_NAME} ({BATTER_SEASON}) vs {PITCHER_NAME} ({PITCHER_SEASON})\n")

    game_seed = derive_game_seed(SIM_SEED, GAME_ID)
    counts: Counter = Counter()

    for i in range(NUM_PA):
        outcome = run_pa(batter, pitcher, game_seed, i)
        counts[outcome] += 1

    # Print outcome table
    order = ["HR", "Triple", "Double", "Single", "BB", "HBP", "Out"]
    print(f"{'Outcome':<10} {'Count':>6}  {'%':>6}")
    print("-" * 26)
    for outcome in order:
        n = counts[outcome]
        print(f"{outcome:<10} {n:>6}  {n / NUM_PA:>6.1%}")
    print("-" * 26)
    print(f"{'TOTAL':<10} {NUM_PA:>6}")

    # Print slash line
    avg, obp, slg = slash_line(counts)
    ops = obp + slg
    print(f"\n  AVG / OBP / SLG / OPS")
    print(f"  .{avg*1000:03.0f} / .{obp*1000:03.0f} / .{slg*1000:03.0f} / .{ops*1000:03.0f}")


if __name__ == "__main__":
    main()
