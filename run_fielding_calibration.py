"""
run_fielding_calibration.py — Avg Regular vs Avg Regular with real defense traits.

Fetches the mean RNG / HND / ARM from the Supabase `players` table (only rows
where all three traits are non-null), uses those as the fielder context, and
runs N plate appearances of the "Avg Regular" archetype (all batting/pitching
traits at 50) vs itself.

Reports K%, BB%, HR/PA, BABIP side-by-side against the 50/50/50 baseline.

Usage:
    python run_fielding_calibration.py [--n 50000]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import supabase
from core_stress_test import _avg_hitter, _avg_pitcher, simulate_matchup, aggregate_counts


# ── Archetype cards ────────────────────────────────────────────────────────────

AVG_BATTER  = _avg_hitter("avg_regular_bat")
AVG_PITCHER = _avg_pitcher("avg_regular_pit")


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_avg_defense() -> dict[str, int]:
    """Query Supabase for the mean RNG / HND / ARM across all fully-populated rows."""
    print("Fetching avg defense traits from Supabase…")
    resp = (
        supabase.table("players")
        .select("rng,hnd,arm")
        .not_.is_("rng", "null")
        .not_.is_("hnd", "null")
        .not_.is_("arm", "null")
        .execute()
    )
    rows = resp.data
    if not rows:
        print("  ⚠  No rows with rng/hnd/arm found — falling back to 50/50/50.")
        return {"RNG": 50, "HND": 50, "ARM": 50}

    rng_mean = sum(r["rng"] for r in rows) / len(rows)
    hnd_mean = sum(r["hnd"] for r in rows) / len(rows)
    arm_mean = sum(r["arm"] for r in rows) / len(rows)

    print(f"  → {len(rows):,} players with full fielding traits")
    print(f"     RNG mean = {rng_mean:.2f}   HND mean = {hnd_mean:.2f}   ARM mean = {arm_mean:.2f}")

    return {
        "RNG": round(rng_mean),
        "HND": round(hnd_mean),
        "ARM": round(arm_mean),
    }


def fmt_row(label: str, stats: dict) -> str:
    return (
        f"  {label:<30}  "
        f"K%={stats['K%']:.3f}  "
        f"BB%={stats['BB%']:.3f}  "
        f"HR/PA={stats['HR%']:.4f}  "
        f"BABIP={stats['BABIP']:.3f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Avg Regular vs Avg Regular — fielding calibration")
    parser.add_argument("--n", type=int, default=50_000, help="PAs to simulate (default: 50,000)")
    args = parser.parse_args()

    n_pa = args.n

    real_defense = fetch_avg_defense()
    baseline     = {"RNG": 50, "HND": 50, "ARM": 50}

    print(f"\nRunning {n_pa:,} PAs — Avg Regular vs Avg Regular")
    print(f"  Baseline fielder  : {baseline}")
    print(f"  Real avg fielder  : {real_defense}")
    print()

    print("  [1/2] Simulating with baseline defense (50/50/50)…")
    counts_base  = simulate_matchup(AVG_BATTER, AVG_PITCHER, n_pa, fielder=baseline)
    stats_base   = aggregate_counts(counts_base, n_pa)

    print("  [2/2] Simulating with real avg defense…")
    counts_real  = simulate_matchup(AVG_BATTER, AVG_PITCHER, n_pa, fielder=real_defense)
    stats_real   = aggregate_counts(counts_real, n_pa)

    print()
    print("=" * 72)
    print(f"  Avg Regular vs Avg Regular — {n_pa:,} PAs")
    print("=" * 72)
    print(fmt_row("Baseline (50/50/50 defense)", stats_base))
    print(fmt_row(
        f"Real avg defense "
        f"({real_defense['RNG']}/{real_defense['HND']}/{real_defense['ARM']})",
        stats_real,
    ))
    print("=" * 72)

    # Deltas
    dk    = stats_real["K%"]    - stats_base["K%"]
    dbb   = stats_real["BB%"]   - stats_base["BB%"]
    dhr   = stats_real["HR%"]   - stats_base["HR%"]
    dbabip= stats_real["BABIP"] - stats_base["BABIP"]
    print(
        f"  {'Delta':<30}  "
        f"ΔK%={dk:+.3f}  "
        f"ΔBB%={dbb:+.3f}  "
        f"ΔHR/PA={dhr:+.4f}  "
        f"ΔBABIP={dbabip:+.3f}"
    )
    print()


if __name__ == "__main__":
    main()
