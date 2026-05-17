"""
salary_report_clean.py — PRD 03 Clean Salary Report

Runs the full valuation + salary pipeline restricted to historically-ingested
player-seasons only. Excludes all seeded / arcade / demo rows via
is_valuation_eligible().

Outputs:
    salary_report_clean.csv  — one row per valued player
    Console reports:
        • Top 25 by salary     (all positions)
        • Top 10 SP by salary
        • Top 10 RP by salary
        • Top 10 hitters by salary
        • Updated salary meta  (DPW, total positive WAR)
"""

import csv
import json
import os

import pandas as pd

from database import supabase
from calculate_values import (
    build_replacement_baselines,
    value_player,
    apply_salary_model,
    _fmt_salary,
    N_SIMS,
    VALUATION_SEED,
    SEASON_PA_HIT,
    SEASON_BF_SP,
    SEASON_BF_RP,
    SEASON_BF_SP,
    N_TEAMS,
    ROSTER_SIZE,
    ROSTER_SLOTS,
    MIN_SALARY,
    TOTAL_BUDGET,
)

# ─── Eligibility Filter ───────────────────────────────────────────────────────

def is_valuation_eligible(row: dict) -> bool:
    """
    True only for historically-ingested player-seasons.

    Seeded / arcade / demo rows have player_id = null (they were inserted via
    seed_players.py without a Lahman ID). All rows produced by ingest_historical.py
    carry an explicit Lahman player_id string ('ruthba01', 'martipe02', etc.).
    """
    return bool(row.get("player_id"))


# ─── Role Inference (historical players have no position column) ──────────────

def infer_player_role(row: dict) -> str:
    """
    Infer 'hitter' or 'pitcher' for historical rows that carry no position.

    Historical hitters:  stuff IS NULL   (column never set by ingest_historical)
    Historical pitchers: stuff IS NOT NULL and stuff > 0
    Two-ways (e.g. pitcher who batted in deadball era): both contact + stuff set;
        we treat them as pitchers since pitcher traits dominate valuation.
    """
    stuff = row.get("stuff")
    return "pitcher" if (stuff is not None and stuff > 0) else "hitter"


def infer_pitcher_role(row: dict) -> str:
    """
    Infer SP or RP for a historical pitcher.

    The players table does not store GS / G, so we have no direct signal.
    Heuristic: any pitcher in the DB passed the min_bf ingestion threshold
    and is assumed to be a starter (SP) unless proven otherwise.
    This is acknowledged in the report header and is the correct conservative
    choice for cross-era valuation (starters are the higher-value benchmark).
    """
    return "SP"


def patch_row(row: dict, role: str) -> dict:
    """
    Returns a shallow copy of the row with `position` set to the inferred role
    so that value_player() picks up the correct scarcity group and season BF.
    """
    out = dict(row)
    if not out.get("position"):
        if role == "pitcher":
            out["position"] = infer_pitcher_role(row)   # "SP" by default
        # hitters stay as position=None → scarcity_group = "UTIL"
    return out


# ─── Lahman BFP lookup ────────────────────────────────────────────────────────

LAHMAN_DIR = "lahman_1871-2025_csv"


def load_lahman_bfp() -> dict[tuple[str, int], int]:
    """
    Returns {(playerID, yearID): total_BFP} aggregated across all stints.
    BFP = Batters Faced by Pitcher, native to Lahman Pitching.csv.
    """
    pit = pd.read_csv(os.path.join(LAHMAN_DIR, "Pitching.csv"))
    agg = (
        pit.groupby(["playerID", "yearID"], as_index=False)["BFP"]
        .sum()
    )
    result: dict[tuple[str, int], int] = {}
    for _, row in agg.iterrows():
        if pd.notna(row["BFP"]) and row["BFP"] > 0:
            result[(row["playerID"], int(row["yearID"]))] = int(row["BFP"])
    return result


# ─── Console Report Helpers ───────────────────────────────────────────────────

_HDR  = (f"  {'Player':<26}  {'Season':>6}  {'Grp':>4}  "
         f"{'WAR':>6}  {'Salary':>10}  {'RAR_bat':>8}  {'RAR_pit':>8}")
_RULE = (f"  {'─'*26}  {'─'*6}  {'─'*4}  "
         f"{'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}")


def _print_rows(rows: list[dict], n: int) -> None:
    print(_HDR)
    print(_RULE)
    for r in rows[:n]:
        print(
            f"  {r['player_name']:<26}  {r['season_year']:>6}  "
            f"{r['scarcity_group']:>4}  "
            f"{r['simWAR']:>+6.2f}  "
            f"{_fmt_salary(r['salary']):>10}  "
            f"{r['hitter_rar']:>8.2f}  "
            f"{r['pitcher_rar']:>8.2f}"
        )


def _section(title: str, rows: list[dict], n: int) -> None:
    bar = "─" * (68 - len(title))
    print(f"\n  ── {title} {bar}")
    if not rows:
        print("  (none)")
        return
    _print_rows(rows, n)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with open("sim_constants.json") as f:
        constants = json.load(f)

    # ── Lahman BFP lookup (reliability shrinkage for pitchers) ───────────────
    print("Loading Lahman BFP table...")
    lahman_bfp = load_lahman_bfp()
    print(f"  → {len(lahman_bfp)} pitcher-seasons with BFP data")

    # ── Fetch + filter ────────────────────────────────────────────────────────
    print("Fetching players from Supabase...")
    all_rows: list[dict] = []
    page_size = 1000
    offset    = 0
    while True:
        page = (
            supabase.table("players")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        all_rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    print(f"  → {len(all_rows)} total rows")

    eligible_raw = [r for r in all_rows if is_valuation_eligible(r)]
    excluded     = len(all_rows) - len(eligible_raw)
    print(f"  → {len(eligible_raw)} eligible (historical)  "
          f"|  {excluded} excluded (seeded / demo)")

    # Patch position field so value_player uses the right scarcity group + BF.
    # For pitchers, also stamp actual_bf from Lahman so value_player can apply
    # reliability shrinkage (BF_eval = min(actual_BF, SP_default)).
    eligible: list[dict] = []
    role_map:  dict[str, str] = {}    # player_id → "hitter" | "pitcher"
    for row in eligible_raw:
        role = infer_player_role(row)
        pid  = row["player_id"]
        role_map[pid] = role
        patched = patch_row(row, role)
        if role == "pitcher":
            season = row.get("season_year") or 0
            actual_bfp = lahman_bfp.get((pid, int(season)))
            if actual_bfp is not None:
                patched["actual_bf"] = actual_bfp
        eligible.append(patched)

    # ── Replacement baselines ─────────────────────────────────────────────────
    print(f"\nPre-computing replacement level baselines ({N_SIMS} PAs each)...")
    rep_hitter_rates, rep_pitcher_rates = build_replacement_baselines(constants)
    print("  Replacement hitter run rates:")
    for g, rate in rep_hitter_rates.items():
        print(f"    {g:<5}  {rate:+.5f} runs/PA")
    print("  Replacement pitcher run rates allowed:")
    for role, rate in rep_pitcher_rates.items():
        print(f"    {role:<5}  {rate:+.5f} runs/PA")

    # ── Valuation loop ────────────────────────────────────────────────────────
    print(f"\nValuating {len(eligible)} historical players ({N_SIMS} sims each)...")
    results: list[dict] = []
    skipped = 0

    for i, row in enumerate(eligible, start=1):
        rec = value_player(row, rep_hitter_rates, rep_pitcher_rates, constants)
        if rec is None:
            skipped += 1
            continue
        # Attach inferred role for downstream report slicing
        rec["_role"] = role_map.get(row.get("player_id", ""), "hitter")
        results.append(rec)
        if i % 100 == 0 or i == len(eligible):
            print(f"  [{i:>4}/{len(eligible)}] {i/len(eligible)*100:.0f}% complete")

    if skipped:
        print(f"  Skipped {skipped} rows with no usable traits.")

    results.sort(key=lambda r: -r["simWAR"])

    # ── Salary model ──────────────────────────────────────────────────────────
    print("\nApplying salary model...")
    salary_meta = apply_salary_model(results)
    dpw         = salary_meta["dollars_per_win"]

    # ── Slice for reports ─────────────────────────────────────────────────────
    by_salary   = sorted(results, key=lambda r: -r["salary"])
    # Top SP: exclude any pitcher whose bf_used fell below 200 (small-sample ghost)
    sp_only     = [r for r in by_salary
                   if r["scarcity_group"] == "SP" and r.get("bf_used", 0) >= 200]
    rp_only     = [r for r in by_salary if r["scarcity_group"] == "RP"]
    hitters     = [r for r in by_salary if r["_role"] == "hitter"]

    # ── Print salary report ───────────────────────────────────────────────────
    eq = "═" * 72
    print(f"\n{eq}")
    print(f"  PRD 03 FINAL SALARY REPORT — HISTORICAL PLAYERS ONLY")
    print(f"  Seed: {VALUATION_SEED}  |  N_sims: {N_SIMS}  |  Players: {len(results)}")
    print(f"  Fixes applied: wOBA weights · BF reliability shrinkage (SP bf≥200)")
    print(f"                 era k_base=0.130 (1927 baseline, 2000 slightly low by ~3.5pp)")
    print(f"  Note: historical pitchers default to SP (no GS data in DB).")
    print(f"        historical hitters default to UTIL (no position in DB).")
    print(eq)

    _section("TOP 25 BY SALARY  (all positions)", by_salary, 25)
    _section("TOP 10 SP BY SALARY  (bf_used ≥ 200)", sp_only, 10)
    _section("TOP 10 RP BY SALARY", rp_only, 10)
    _section("TOP 10 HITTERS BY SALARY", hitters, 10)

    # ── Ruth spotlight ────────────────────────────────────────────────────────
    ruth_rec = next((r for r in results if r.get("player_id") == "ruthba01"), None)
    if ruth_rec:
        # Re-simulate Ruth with outcome tracking for K% and HR% readout
        import random
        from pa_engine import (resolve_duel, resolve_contact, map_bip_outcome,
                                resolve_defense, derive_game_seed, derive_pa_seed)
        from calculate_values import (extract_hitter_traits, AVG_PITCHER_TRAITS,
                                       AVG_DEFENSE, NEUTRAL_HANDEDNESS, VALUATION_SEED as VS)
        from collections import Counter

        ruth_row_raw = next(
            r for r in eligible if r.get("player_id") == "ruthba01"
        )
        ruth_traits = extract_hitter_traits(ruth_row_raw)
        game_seed   = derive_game_seed(VS, f"hitter_ruthba01_1927")
        counts      = Counter()
        for pa_i in range(N_SIMS):
            pa_seed = derive_pa_seed(game_seed, pa_i)
            rng     = random.Random(pa_seed)
            duel    = resolve_duel(ruth_traits, AVG_PITCHER_TRAITS,
                                   NEUTRAL_HANDEDNESS, constants, rng)
            outcome = duel["outcome"]
            if outcome == "BIP":
                contact = resolve_contact(ruth_traits, AVG_PITCHER_TRAITS,
                                          constants, rng)
                bip_map = map_bip_outcome(
                    contact["contact_quality"], contact["contact_score"],
                    contact["spray_vector"], contact["effective_pow"],
                    ruth_traits, AVG_PITCHER_TRAITS, constants, rng)
                defense = resolve_defense(
                    bip_map["bip_outcome"], contact["spray_vector"],
                    contact["contact_quality"], AVG_DEFENSE, constants, rng)
                outcome = defense["final_outcome"]
            counts[outcome] += 1

        total    = sum(counts.values())
        sim_k    = counts["K"] / total
        sim_hr   = counts["HR"] / total
        sim_bb   = counts["BB"] / total

        print(f"\n{eq}")
        print(f"  RUTH 1927 SPOTLIGHT  (N_sims={N_SIMS:,})")
        print(f"  Traits:  CON={ruth_traits['CON']}  POW={ruth_traits['POW']}  "
              f"EYE={ruth_traits['EYE']}  AK={ruth_traits['AK']}  GAP={ruth_traits['GAP']}")
        print(f"  Simulated:  K%={sim_k:.1%}  HR%={sim_hr:.1%}  BB%={sim_bb:.1%}")
        print(f"  Actual 1927: K%=12.9%  HR%=8.7%  BB%=19.8%")
        print(f"  RAR_bat: {ruth_rec['hitter_rar']:+.2f}  |  WAR: {ruth_rec['simWAR']:+.2f}"
              f"  |  Salary: {_fmt_salary(ruth_rec['salary'])}")
        print(eq)

    # ── Era distortion note ───────────────────────────────────────────────────
    print(f"\n  ERA CALIBRATION NOTE (k_base = 0.130)")
    print(f"  ┌─────────────┬──────────────┬─────────────┬──────────────────────────┐")
    print(f"  │ Era         │ Actual lg K% │ Sim avg K%  │ Distortion               │")
    print(f"  ├─────────────┼──────────────┼─────────────┼──────────────────────────┤")
    print(f"  │ 1927        │    5.1%      │  ~13–16%    │ Still 2.5–3× too high    │")
    print(f"  │ 2000        │   16.5%      │  ~13–16%    │ ~3.5pp too low (slight)  │")
    print(f"  │ Game world  │   13.0%      │  ~13%       │ Calibrated (neutral)     │")
    print(f"  └─────────────┴──────────────┴─────────────┴──────────────────────────┘")
    print(f"  Implication: 2000-era hitters get fractionally fewer simulated Ks")
    print(f"  vs their actual history, mildly over-valuing their contact. 1927")
    print(f"  players still see a higher absolute K% than their era, but relative")
    print(f"  ordering within each era is preserved by the era-relative trait pipeline.")

    print(f"\n{eq}")
    print(f"  SALARY META")
    print(f"  {'DPW':<28}: {_fmt_salary(dpw)}")
    print(f"  {'Total Positive WAR':<28}: {salary_meta['total_positive_war']:.2f}")
    print(f"  {'Discretionary Pool':<28}: {_fmt_salary(salary_meta['discretionary_pool'])}")
    print(f"  {'Min Salary Pool':<28}: {_fmt_salary(salary_meta['min_salary_pool'])}")
    print(f"  {'Total Budget':<28}: {_fmt_salary(TOTAL_BUDGET)}")
    print(f"  {'Roster Slots':<28}: {ROSTER_SLOTS}  ({N_TEAMS} teams × {ROSTER_SIZE} spots)")
    print(f"  {'Min Salary':<28}: {_fmt_salary(MIN_SALARY)}")
    print(eq)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_fields = [
        "player_id",
        "player_name",
        "season_year",
        "role",
        "position_group_used",
        "PA_BF",
        "RAR_bat",
        "RAR_pit",
        "WAR",
        "salary",
    ]

    csv_rows: list[dict] = []
    for r in results:
        role  = r["_role"]
        grp   = r["scarcity_group"]
        pa_bf = r.get("bf_used", SEASON_PA_HIT)   # actual BF after shrinkage
        csv_rows.append({
            "player_id":           r["player_id"],
            "player_name":         r["player_name"],
            "season_year":         r["season_year"],
            "role":                role,
            "position_group_used": grp,
            "PA_BF":               pa_bf,
            "RAR_bat":             r["hitter_rar"],
            "RAR_pit":             r["pitcher_rar"],
            "WAR":                 r["simWAR"],
            "salary":              r["salary"],
        })

    out_path = "salary_report_final.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    total_salary = sum(r["salary"] for r in results)
    print(f"\nCSV written → {out_path}  ({len(csv_rows)} rows)")
    print(f"Total salary assigned (all {len(results)} players): {_fmt_salary(total_salary)}")


if __name__ == "__main__":
    main()
