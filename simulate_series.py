"""
simulate_series.py — Multi-PA outcome distribution test.
Simulates N plate appearances between two players and prints outcome rates.
"""
import json
import random
from collections import Counter

from pa_engine import (
    resolve_duel, resolve_contact, map_bip_outcome,
    resolve_defense, derive_game_seed, derive_pa_seed,
)

PILOT_FILE      = "pilot_1927_nya.json"
DEFAULT_FIELDER = {"RNG": 65, "HND": 65, "ARM": 65}
SIM_SEED        = "series-test-2026"
GAME_ID         = "game-001"


def load_roster(path):
    with open(path) as f:
        return json.load(f)


def find_player(roster, name, role):
    for card in roster:
        if name.lower() in card["name"].lower() and card["primary_role"] == role:
            return card
    return None


def simulate_pa(batter, pitcher, constants, pa_index, game_index=1):
    # Spread PAs across multiple game seeds for better RNG diversity.
    # pa_index resets each game; game_index drives a unique game seed.
    game_id   = f"game-{game_index:04d}"
    game_seed = derive_game_seed(SIM_SEED, game_id)
    pa_seed   = derive_pa_seed(game_seed, pa_index)
    rng       = random.Random(pa_seed)

    batter_traits  = batter["traits"]
    pitcher_traits = pitcher["traits"]
    handedness     = {"batter": batter["bats"], "pitcher": pitcher["throws"]}

    duel = resolve_duel(batter_traits, pitcher_traits, handedness, constants, rng)

    if duel["outcome"] != "BIP":
        return {"final": duel["outcome"], "bip": False, "contact_quality": None, "pre_defense": None}

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
        fielder         = DEFAULT_FIELDER,
        constants       = constants,
        rng             = rng,
    )
    return {
        "final":           defense["final_outcome"],
        "bip":             True,
        "contact_quality": contact["contact_quality"],
        "pre_defense":     bip_map["bip_outcome"],
    }


def _make_player(name, role, traits, bats="R", throws="R"):
    """Build a minimal player object compatible with simulate_pa."""
    return {"name": name, "primary_role": role, "traits": traits, "bats": bats, "throws": throws}


def _print_series(batter, pitcher, counts, bip_total, contact_counts, tier_out, hard_hr, n):
    """Print the full results block for a completed series."""
    order = ["HR", "Triple", "Double", "Single", "BB", "HBP", "Out", "Error", "K"]
    total = sum(counts.values())

    print(f"{'Outcome':<10} {'Count':>6}  {'Rate':>7}  {'Bar'}")
    print("─" * 50)
    for outcome in order:
        if counts[outcome] == 0:
            continue
        count = counts[outcome]
        pct   = count / total
        bar   = "█" * int(pct * 40)
        print(f"  {outcome:<8} {count:>6}  {pct:>6.1%}  {bar}")
    print("─" * 50)
    print(f"  {'TOTAL':<8} {total:>6}")

    hits    = counts["HR"] + counts["Triple"] + counts["Double"] + counts["Single"]
    ab      = total - counts["BB"] - counts["HBP"]
    slg_num = counts["Single"] + 2*counts["Double"] + 3*counts["Triple"] + 4*counts["HR"]
    ba      = hits / ab if ab else 0
    obp     = (hits + counts["BB"] + counts["HBP"]) / total
    slg     = slg_num / ab if ab else 0
    babip_den = ab - counts["K"] - counts["HR"]
    babip   = (hits - counts["HR"]) / babip_den if babip_den else 0

    print(f"\n  BA:  {ba:.3f}   OBP: {obp:.3f}   SLG: {slg:.3f}   OPS: {ba+slg:.3f}")
    print(f"  BABIP: {babip:.3f}")
    print(f"  HR%: {counts['HR']/total:.1%}   K%: {counts['K']/total:.1%}   BB%: {counts['BB']/total:.1%}")

    print(f"\n  Contact%: {bip_total}/{total} = {bip_total/total:.1%}")
    print(f"\n  {'Tier':<8} {'BIPs':>5}  {'BIP%':>6}  {'Outs':>5}  {'Out%':>6}  Bar")
    print("  " + "─" * 44)
    for tier in ("Hard", "Medium", "Weak"):
        cnt  = contact_counts[tier]
        outs = tier_out[tier]
        bpct = cnt  / bip_total if bip_total else 0
        opct = outs / cnt       if cnt       else 0
        bar  = "█" * int(bpct * 25)
        print(f"  {tier:<8} {cnt:>5}  {bpct:>6.1%}  {outs:>5}  {opct:>6.1%}  {bar}")

    hard_total  = contact_counts["Hard"]
    hard_pct    = hard_total / bip_total if bip_total else 0
    hr_per_hard = hard_hr / hard_total if hard_total else 0
    print(f"\n  Hard%: {hard_pct:.1%}  |  HR/PA: {counts['HR']/total:.1%}  |  HR/Hard: {hr_per_hard:.1%}")
    print()


def _run(batter, pitcher, constants, n):
    """Simulate n PAs and collect stats; returns nothing — prints inline."""
    t = batter["traits"]
    p = pitcher["traits"]
    print(f"\n{'═'*58}")
    print(f"  {batter['name']}  vs  {pitcher['name']}  ({n:,} PAs)")
    print(f"  Batter:  POW {t.get('POW','-'):>3}  EYE {t.get('EYE','-'):>3}  "
          f"CON {t.get('CON','-'):>3}  AK {t.get('AK','-'):>3}  GAP {t.get('GAP','-'):>3}")
    print(f"  Pitcher: STF {p.get('STF','-'):>3}  CTL {p.get('CTL','-'):>3}  "
          f"CMD {p.get('CMD','-'):>3}  STA {p.get('STA','-'):>3}")
    print(f"{'═'*58}\n")

    PAS_PER_GAME   = 9
    counts         = Counter()
    bip_total      = 0
    contact_counts = Counter()
    tier_out       = Counter()
    hard_hr        = 0

    for i in range(n):
        game_index = (i // PAS_PER_GAME) + 1
        pa_in_game = (i %  PAS_PER_GAME) + 1
        result = simulate_pa(batter, pitcher, constants,
                             pa_index=pa_in_game, game_index=game_index)
        counts[result["final"]] += 1
        if result["bip"]:
            bip_total += 1
            cq = result["contact_quality"]
            contact_counts[cq] += 1
            if result["final"] in ("Out", "Error"):
                tier_out[cq] += 1
            if result["final"] == "HR" and cq == "Hard":
                hard_hr += 1

    _print_series(batter, pitcher, counts, bip_total, contact_counts, tier_out, hard_hr, n)


def run_series(batter_name, pitcher_name, n=100):
    with open("sim_constants.json") as f:
        constants = json.load(f)
    roster  = load_roster(PILOT_FILE)
    batter  = find_player(roster, batter_name,  "Hitter")
    pitcher = find_player(roster, pitcher_name, "Pitcher")
    if not batter or not pitcher:
        print(f"Could not find '{batter_name}' or '{pitcher_name}' in roster.")
        return
    _run(batter, pitcher, constants, n)


def run_matrix(n=5000):
    with open("sim_constants.json") as f:
        constants = json.load(f)

    roster  = load_roster(PILOT_FILE)
    ruth    = find_player(roster, "Babe Ruth",  "Hitter")
    hoyt    = find_player(roster, "Waite Hoyt", "Pitcher")

    avg_pitcher = _make_player("Avg Pitcher", "Pitcher",
                               {"STF": 50, "CTL": 50, "CMD": 50, "STA": 50})
    elite_pitcher = _make_player("Elite Pitcher", "Pitcher",
                                 {"STF": 80, "CTL": 80, "CMD": 80, "STA": 75})
    avg_hitter = _make_player("Avg Hitter", "Hitter",
                              {"POW": 50, "EYE": 50, "CON": 55, "AK": 50, "GAP": 50})
    contact_hitter = _make_player("Contact Hitter", "Hitter",
                                  {"POW": 45, "EYE": 60, "CON": 80, "AK": 75, "GAP": 55})

    matchups = [
        (ruth,          avg_pitcher,   "Ruth vs Average Pitcher"),
        (ruth,          elite_pitcher, "Ruth vs Elite Pitcher"),
        (avg_hitter,    hoyt,          "Average Hitter vs Hoyt"),
        (contact_hitter, hoyt,         "Contact Hitter vs Hoyt"),
    ]

    print(f"\nSeed: {SIM_SEED}  |  {n:,} PAs per matchup")
    for batter, pitcher, _ in matchups:
        _run(batter, pitcher, constants, n)


if __name__ == "__main__":
    with open("sim_constants.json") as f:
        _c = json.load(f)
    roster = load_roster(PILOT_FILE)
    ruth   = find_player(roster, "Babe Ruth",  "Hitter")
    avg_p  = _make_player("Avg Pitcher",  "Pitcher", {"STF": 50, "CTL": 50, "CMD": 50, "STA": 50})
    avg_h  = _make_player("Avg Hitter",   "Hitter",  {"POW": 50, "EYE": 50, "CON": 55, "AK": 50, "GAP": 50})

    _run(ruth,  avg_p, _c, 5000)
    _run(avg_h, avg_p, _c, 5000)
