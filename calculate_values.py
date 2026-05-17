"""
calculate_values.py — PRD 03: The Value Engine

Implements SimWAR → Salary:
  1. Fetch every player from the Supabase `players` table.
  2. For each player, run N_SIMS plate appearances vs a Neutral Evaluation Context
     (average-traits archetypes) to compute their expected runs/PA.
  3. Compare against a position-specific Replacement Level archetype to get RAR.
  4. Divide by RPW (Runs Per Win) to get simWAR.
  5. Apply the PRD 03 Salary Model to convert WAR → dollars.
  6. Write results to valuation_test.json.

All simulations are fully deterministic via a fixed VALUATION_SEED.
"""

import json
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


# ─── Valuation Constants ──────────────────────────────────────────────────────

VALUATION_SEED = "chin-music-valuation-v1"   # fixed seed — deterministic per PRD 03 §5A
N_SIMS         = 10_000                        # PA simulations per player — PRD 03 baseline
SEASON_PA_HIT  = 600                           # typical full-season PAs for a hitter
SEASON_BF_SP   = 750                           # typical SP season batters faced
SEASON_BF_RP   = 250                           # typical RP season batters faced
RPW            = 10.0                          # Runs Per Win (PRD 03 §4)

# ─── Salary Model Constants (PRD 03 §7) ──────────────────────────────────────

N_TEAMS        = 12
ROSTER_SIZE    = 26                             # active roster slots per team
ROSTER_SLOTS   = N_TEAMS * ROSTER_SIZE          # 312 total active slots
MIN_SALARY     = 1_500_000                      # $1.5M floor per slot
TOTAL_BUDGET   = N_TEAMS * 260_000_000          # $3.12B league-wide pool

# Linear weights — wOBA-calibrated run values per PA outcome.
# Anchor: Out = 0.00 (all outs are equal, no negative penalty).
# Positive event weights sourced from wOBA linear weights (2024 MLB scale).
# Applied uniformly so cross-era comparisons remain consistent inside the
# neutral evaluation context.
LW: dict[str, float] = {
    "K":       0.00,   # same value as any out
    "Out":     0.00,
    "Error":   0.50,   # batter reaches; ~half-single credit (not in official wOBA)
    "BB":      0.69,
    "HBP":     0.69,
    "Single":  0.89,
    "Double":  1.27,
    "Triple":  1.62,
    "HR":      2.10,
}


# ─── Neutral Evaluation Context — Archetypes (PRD 03 §5A) ────────────────────
# "League Average" card at traits ~70/100 on every dimension.
# Using neutral handedness (R vs R) so no platoon modifier fires.

NEUTRAL_HANDEDNESS = {"batter": "R", "pitcher": "R"}

AVG_HITTER_TRAITS: dict[str, int] = {
    "CON": 70, "POW": 70, "EYE": 70,
    "AK":  70, "GAP": 70, "BNT": 50,
}
AVG_PITCHER_TRAITS: dict[str, int] = {
    "STF": 70, "CTL": 70, "CMD": 70, "STA": 70,
}
AVG_DEFENSE: dict[str, int] = {
    "RNG": 65, "HND": 65, "ARM": 65,
}


# ─── Replacement Level Archetypes (PRD 03 §5B–5C) ────────────────────────────
# One replacement archetype per hitter scarcity group and pitcher role.
# Traits hover around the sim_constants `replacement_level_skillscore` (50).
# Positional scarcity is encoded via modest adjustments:
#   C < IF ≈ OF < 1B ≈ UTIL   (harder to replace = lower replacement level traits)
#   SP ≈ RP

REP_HITTERS: dict[str, dict] = {
    "C":    {"CON": 47, "POW": 43, "EYE": 46, "AK": 46, "GAP": 43, "BNT": 50},
    "IF":   {"CON": 50, "POW": 46, "EYE": 48, "AK": 50, "GAP": 47, "BNT": 50},
    "1B":   {"CON": 53, "POW": 54, "EYE": 50, "AK": 50, "GAP": 52, "BNT": 50},
    "OF":   {"CON": 50, "POW": 50, "EYE": 48, "AK": 50, "GAP": 50, "BNT": 50},
    "UTIL": {"CON": 52, "POW": 50, "EYE": 50, "AK": 50, "GAP": 50, "BNT": 50},
}

REP_PITCHERS: dict[str, dict] = {
    "SP": {"STF": 50, "CTL": 52, "CMD": 50, "STA": 50},
    "RP": {"STF": 52, "CTL": 52, "CMD": 52, "STA": 50},
}


# ─── Position → Scarcity Group ────────────────────────────────────────────────

_PITCHER_POSITIONS = {"SP", "RP"}
_C_POSITIONS       = {"C"}
_IF_POSITIONS      = {"2B", "SS", "3B"}
_1B_POSITIONS      = {"1B"}
_OF_POSITIONS      = {"OF", "LF", "CF", "RF"}


def get_scarcity_group(position: str | None) -> str:
    """Map a position string to one of: C, IF, 1B, OF, UTIL, SP, RP."""
    if not position:
        return "UTIL"
    pos = position.upper().strip()
    if pos in _PITCHER_POSITIONS:
        return pos          # "SP" or "RP"
    if pos in _C_POSITIONS:
        return "C"
    if pos in _IF_POSITIONS:
        return "IF"
    if pos in _1B_POSITIONS:
        return "1B"
    if pos in _OF_POSITIONS:
        return "OF"
    return "UTIL"           # DH, LF in some schemas, bench, unknown


# ─── Player Classification ────────────────────────────────────────────────────

def classify_player(row: dict) -> str:
    """Return 'hitter', 'pitcher', or 'unknown' based on position and traits."""
    position = (row.get("position") or "").upper().strip()
    contact  = row.get("contact") or 0
    power    = row.get("power")   or 0
    stuff    = row.get("stuff")   or 0

    if position in _PITCHER_POSITIONS:
        return "pitcher"
    if position and position not in _PITCHER_POSITIONS:
        return "hitter"

    # No position set — infer from which trait set is populated.
    if stuff > 0 and (contact + power) == 0:
        return "pitcher"
    if (contact + power) > 0:
        return "hitter"

    return "unknown"


# ─── Trait Extraction ─────────────────────────────────────────────────────────

def extract_hitter_traits(row: dict) -> dict | None:
    """Map Supabase hitter columns → PA Engine hitter trait schema."""
    contact = row.get("contact") or 0
    power   = row.get("power")   or 0
    eye     = row.get("eye")     or 0
    speed   = row.get("speed")   or 0

    if contact == 0 and power == 0 and eye == 0:
        return None

    return {
        "CON": contact,
        "POW": power,
        "EYE": eye,
        "AK":  speed,                    # speed proxy for anti-K awareness
        "GAP": (contact + power) // 2,   # derived gap power (blended)
        "BNT": 50,
    }


def extract_pitcher_traits(row: dict) -> dict | None:
    """Map Supabase pitcher columns → PA Engine pitcher trait schema."""
    stuff    = row.get("stuff")    or 0
    control  = row.get("control")  or 0
    movement = row.get("movement") or 0

    if stuff == 0 and control == 0 and movement == 0:
        return None

    position = (row.get("position") or "").upper().strip()
    sta = 68 if position == "SP" else 55   # starters need more stamina

    return {
        "STF": stuff,
        "CTL": control,
        "CMD": movement,
        "STA": sta,
    }


# ─── Core PA Simulation ───────────────────────────────────────────────────────

def _resolve_pa(
    batter_traits:  dict,
    pitcher_traits: dict,
    fielder_traits: dict,
    constants:      dict,
    rng:            random.Random,
) -> str:
    """Runs a single PA through all three engine stages. Returns the final outcome."""
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

    return outcome


def simulate_run_rate(
    batter_traits:  dict,
    pitcher_traits: dict,
    fielder_traits: dict,
    constants:      dict,
    seed_key:       str,
) -> float:
    """
    Simulates N_SIMS plate appearances and returns expected runs per PA.

    Determinism: VALUATION_SEED + seed_key are hashed into a game seed;
    each PA index then derives its own pa_seed via the chain in PRD 01 §7.
    The same seed_key always produces identical outcomes.
    """
    game_seed = derive_game_seed(VALUATION_SEED, seed_key)
    total_lw  = 0.0

    for pa_index in range(N_SIMS):
        pa_seed = derive_pa_seed(game_seed, pa_index)
        rng     = random.Random(pa_seed)
        outcome = _resolve_pa(batter_traits, pitcher_traits, fielder_traits, constants, rng)
        total_lw += LW.get(outcome, 0.0)

    return total_lw / N_SIMS


# ─── Pre-compute Replacement Level Baselines ─────────────────────────────────

def build_replacement_baselines(constants: dict) -> tuple[dict[str, float], dict[str, float]]:
    """
    Simulate every replacement archetype vs the neutral baseline.

    Returns:
        rep_hitter_rates   — {group: runs_per_PA} for each hitter scarcity group
        rep_pitcher_rates  — {role:  runs_allowed_per_PA} for SP and RP
    """
    rep_hitter_rates: dict[str, float] = {}
    for group, traits in REP_HITTERS.items():
        seed_key = f"__rep_hitter_{group}__"
        rep_hitter_rates[group] = simulate_run_rate(
            traits, AVG_PITCHER_TRAITS, AVG_DEFENSE, constants, seed_key
        )

    rep_pitcher_rates: dict[str, float] = {}
    for role, traits in REP_PITCHERS.items():
        seed_key = f"__rep_pitcher_{role}__"
        rep_pitcher_rates[role] = simulate_run_rate(
            AVG_HITTER_TRAITS, traits, AVG_DEFENSE, constants, seed_key
        )

    return rep_hitter_rates, rep_pitcher_rates


# ─── Per-Player Valuation ─────────────────────────────────────────────────────

def value_player(
    row:                dict,
    rep_hitter_rates:   dict[str, float],
    rep_pitcher_rates:  dict[str, float],
    constants:          dict,
) -> dict | None:
    """
    Computes simWAR for a single Supabase player row.

    Returns a value record dict, or None if the row has no usable traits.
    """
    player_id   = row.get("player_id") or str(row.get("id", "unknown"))
    player_name = row.get("player_name") or "Unknown"
    season      = row.get("season_year") or 0
    position    = row.get("position")
    role        = classify_player(row)

    if role == "unknown":
        return None

    scarcity_group = get_scarcity_group(position)

    # When position is null, enforce consistency with trait-inferred role.
    if not position:
        scarcity_group = "SP" if role == "pitcher" else "UTIL"

    is_pitcher = (role == "pitcher")

    hitter_traits  = None if is_pitcher else extract_hitter_traits(row)
    pitcher_traits = extract_pitcher_traits(row) if is_pitcher else None

    if hitter_traits is None and pitcher_traits is None:
        return None

    # ── Hitter RAR (PRD 03 §5B) ───────────────────────────────────────────────
    hitter_rar     = 0.0
    hitter_rpa     = None
    rep_hitter_rpa = None

    if hitter_traits:
        seed_key   = f"hitter_{player_id}_{season}"
        player_rpa = simulate_run_rate(
            hitter_traits, AVG_PITCHER_TRAITS, AVG_DEFENSE, constants, seed_key
        )
        rep_rpa    = rep_hitter_rates[scarcity_group]
        hitter_rar = (player_rpa - rep_rpa) * SEASON_PA_HIT
        hitter_rpa     = round(player_rpa, 5)
        rep_hitter_rpa = round(rep_rpa, 5)

    # ── Pitcher RAR (PRD 03 §5C) ──────────────────────────────────────────────
    pitcher_rar     = 0.0
    pitcher_rpa     = None
    rep_pitcher_rpa = None

    if pitcher_traits:
        role_key        = scarcity_group if scarcity_group in REP_PITCHERS else "RP"
        default_bf      = SEASON_BF_SP if role_key == "SP" else SEASON_BF_RP
        # Reliability shrinkage: cap to actual historical workload so a pitcher
        # who threw 44 real BF is never extrapolated over a full SP season (750 BF).
        actual_bf_raw   = row.get("actual_bf")
        season_bf       = (min(int(actual_bf_raw), default_bf)
                           if actual_bf_raw is not None and actual_bf_raw > 0
                           else default_bf)
        seed_key        = f"pitcher_{player_id}_{season}"
        player_rpa_a    = simulate_run_rate(
            AVG_HITTER_TRAITS, pitcher_traits, AVG_DEFENSE, constants, seed_key
        )
        rep_rpa_a       = rep_pitcher_rates[role_key]
        # Pitchers gain RAR by allowing *fewer* runs than replacement.
        pitcher_rar     = (rep_rpa_a - player_rpa_a) * season_bf
        pitcher_rpa     = round(player_rpa_a, 5)
        rep_pitcher_rpa = round(rep_rpa_a, 5)

    # ── Defense RAR (PRD 03 §5D) ─────────────────────────────────────────────
    # Deferred: Supabase players table does not yet carry defensive traits
    # (RNG, HND, ARM). Defense value is 0.0 until defensive cards are built.
    # The 10% calibration target will be applied once those traits are available.
    defense_rar = 0.0

    # ── WAR (PRD 03 §6) ───────────────────────────────────────────────────────
    total_rar = hitter_rar + pitcher_rar + defense_rar
    sim_war   = round(total_rar / RPW, 2)

    return {
        "player_id":        player_id,
        "player_name":      player_name,
        "season_year":      season,
        "position":         position,
        "scarcity_group":   scarcity_group,
        "simWAR":           sim_war,
        "total_rar":        round(total_rar, 2),
        "hitter_rar":       round(hitter_rar, 2),
        "pitcher_rar":      round(pitcher_rar, 2),
        "defense_rar":      defense_rar,
        # BF/PA actually used when computing RAR (reflects reliability shrinkage)
        "bf_used":          season_bf if pitcher_traits else SEASON_PA_HIT,
        # Per-PA rates — useful for debugging and sanity checks
        "hitter_rpa":       hitter_rpa,
        "rep_hitter_rpa":   rep_hitter_rpa,
        "pitcher_rpa":      pitcher_rpa,
        "rep_pitcher_rpa":  rep_pitcher_rpa,
        # Traits actually used in the sims
        "traits_used": {
            "hitter":  hitter_traits,
            "pitcher": pitcher_traits,
        },
    }


# ─── Salary Model (PRD 03 §7) ────────────────────────────────────────────────

def apply_salary_model(results: list[dict]) -> dict:
    """
    Applies the PRD 03 salary formula to a valued player list.

    Formula:
        DPW    = (Total Budget  −  Roster Slots × Min Salary) / Total Positive WAR
        Salary = Min Salary + max(0, WAR) × DPW

    Every player receives at least the minimum salary regardless of WAR.
    Players with WAR ≤ 0 receive exactly the minimum salary.

    Returns a salary_meta dict containing DPW and pool summary stats.
    Mutates each record in `results` in-place, adding a `salary` key.
    """
    total_positive_war = sum(r["simWAR"] for r in results if r["simWAR"] > 0)
    min_salary_pool    = ROSTER_SLOTS * MIN_SALARY
    discretionary_pool = TOTAL_BUDGET - min_salary_pool

    if total_positive_war <= 0:
        raise ValueError("No positive WAR in the pool — cannot compute DPW.")

    dpw = discretionary_pool / total_positive_war

    for r in results:
        marginal = max(0.0, r["simWAR"])
        r["salary"] = round(MIN_SALARY + marginal * dpw)

    salary_meta = {
        "total_budget":        TOTAL_BUDGET,
        "roster_slots":        ROSTER_SLOTS,
        "min_salary":          MIN_SALARY,
        "min_salary_pool":     min_salary_pool,
        "discretionary_pool":  discretionary_pool,
        "total_positive_war":  round(total_positive_war, 2),
        "dollars_per_win":     round(dpw),
    }
    return salary_meta


def _fmt_salary(dollars: int) -> str:
    """Formats an integer dollar amount as e.g. '$12.34M' or '$1.50M'."""
    return f"${dollars / 1_000_000:.2f}M"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with open("sim_constants.json") as f:
        constants = json.load(f)

    # ── Fetch players ─────────────────────────────────────────────────────────
    print("Fetching players from Supabase...")
    response = supabase.table("players").select("*").execute()
    players  = response.data
    print(f"  → {len(players)} rows fetched")

    # ── Pre-compute replacement baselines ─────────────────────────────────────
    total_rep_sims = (len(REP_HITTERS) + len(REP_PITCHERS)) * N_SIMS
    print(f"\nPre-computing replacement level baselines "
          f"({len(REP_HITTERS) + len(REP_PITCHERS)} archetypes × {N_SIMS} PAs = {total_rep_sims:,} sims)...")

    rep_hitter_rates, rep_pitcher_rates = build_replacement_baselines(constants)

    print("  Replacement hitter run rates (vs avg pitcher):")
    for group, rate in rep_hitter_rates.items():
        print(f"    {group:<5}  {rate:+.5f} runs/PA")

    print("  Replacement pitcher run rates allowed (vs avg hitter):")
    for role, rate in rep_pitcher_rates.items():
        print(f"    {role:<5}  {rate:+.5f} runs/PA")

    # ── Value every player ────────────────────────────────────────────────────
    print(f"\nValuating {len(players)} players ({N_SIMS} sims each) ...")
    results: list[dict] = []
    skipped = 0

    for i, row in enumerate(players, start=1):
        rec = value_player(row, rep_hitter_rates, rep_pitcher_rates, constants)
        if rec is None:
            skipped += 1
            continue
        results.append(rec)
        if i % 5 == 0 or i == len(players):
            name = row.get("player_name", "?")
            war  = rec["simWAR"]
            print(f"  [{i:>3}/{len(players)}] {name:<28}  simWAR = {war:+.2f}")

    results.sort(key=lambda r: -r["simWAR"])

    # ── Apply salary model ────────────────────────────────────────────────────
    print("\nApplying salary model...")
    salary_meta = apply_salary_model(results)
    dpw = salary_meta["dollars_per_win"]
    print(f"  Total positive WAR : {salary_meta['total_positive_war']:>8.2f}")
    print(f"  Discretionary pool : {_fmt_salary(salary_meta['discretionary_pool'])}")
    print(f"  Dollars Per Win    : {_fmt_salary(dpw)}")

    # Sort by salary descending for the report
    by_salary = sorted(results, key=lambda r: -r["salary"])

    # ── Print salary report ───────────────────────────────────────────────────
    sep  = "─" * 72
    hdr  = f"  {'Player':<28}  {'Pos':>5}  {'WAR':>6}  {'Salary':>10}"
    rule = f"  {'─'*28}  {'─'*5}  {'─'*6}  {'─'*10}"

    print(f"\n{'═' * 72}")
    print(f"  SALARY REPORT  —  DPW = {_fmt_salary(dpw)}  |  "
          f"Min = {_fmt_salary(MIN_SALARY)}  |  Pool = {_fmt_salary(TOTAL_BUDGET)}")
    print(f"{'═' * 72}")

    print(f"\n  ── TOP 20 EARNERS ──────────────────────────────────────────────")
    print(hdr)
    print(rule)
    for r in by_salary[:20]:
        pos = r["position"] or "?"
        print(
            f"  {r['player_name']:<28}  {pos:>5}  "
            f"{r['simWAR']:>+6.2f}  "
            f"{_fmt_salary(r['salary']):>10}"
        )

    print(f"\n  ── BOTTOM 20 EARNERS ───────────────────────────────────────────")
    print(hdr)
    print(rule)
    for r in by_salary[-20:]:
        pos = r["position"] or "?"
        print(
            f"  {r['player_name']:<28}  {pos:>5}  "
            f"{r['simWAR']:>+6.2f}  "
            f"{_fmt_salary(r['salary']):>10}"
        )

    print(f"\n{sep}")
    print(f"  Total valued: {len(results)}   Skipped (no traits): {skipped}")
    total_assigned = sum(r["salary"] for r in results)
    print(f"  Total salary assigned  : {_fmt_salary(total_assigned)}")
    print(f"  League budget          : {_fmt_salary(TOTAL_BUDGET)}")
    print(sep)

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "valuation_seed":     VALUATION_SEED,
            "n_sims":             N_SIMS,
            "rpw":                RPW,
            "season_pa_hitter":   SEASON_PA_HIT,
            "season_bf_sp":       SEASON_BF_SP,
            "season_bf_rp":       SEASON_BF_RP,
            "linear_weights":     LW,
            "neutral_archetypes": {
                "avg_hitter":  AVG_HITTER_TRAITS,
                "avg_pitcher": AVG_PITCHER_TRAITS,
                "avg_defense": AVG_DEFENSE,
            },
            "replacement_level_archetypes": {
                "hitters":  REP_HITTERS,
                "pitchers": REP_PITCHERS,
            },
            "replacement_run_rates": {
                "hitters":  {k: round(v, 5) for k, v in rep_hitter_rates.items()},
                "pitchers": {k: round(v, 5) for k, v in rep_pitcher_rates.items()},
            },
            "salary_model":  salary_meta,
            "defense_note": (
                "Defense RAR is 0.0 for all players in this run. "
                "The Supabase `players` table does not yet carry RNG/HND/ARM traits. "
                "The ~10% defense calibration target (PRD 03 §5D) will be applied "
                "once defensive player cards are available."
            ),
        },
        "players": results,
    }

    out_path = "valuation_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to → {out_path}")


if __name__ == "__main__":
    main()
