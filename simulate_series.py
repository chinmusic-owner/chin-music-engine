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


def run_series(batter_name, pitcher_name, n=100):
    with open("sim_constants.json") as f:
        constants = json.load(f)

    roster  = load_roster(PILOT_FILE)
    batter  = find_player(roster, batter_name,  "Hitter")
    pitcher = find_player(roster, pitcher_name, "Pitcher")

    if not batter or not pitcher:
        print(f"Could not find '{batter_name}' or '{pitcher_name}' in roster.")
        return

    print(f"\nSimulating {n} PAs: {batter['name']} vs {pitcher['name']}")
    print(f"Batter:  POW {batter['traits']['POW']}  EYE {batter['traits']['EYE']}  "
          f"CON {batter['traits']['CON']}  AK {batter['traits']['AK']}")
    print(f"Pitcher: STF {pitcher['traits']['STF']}  CTL {pitcher['traits']['CTL']}  "
          f"CMD {pitcher['traits']['CMD']}  STA {pitcher['traits']['STA']}")
    print(f"Seed: {SIM_SEED}\n")

    # Distribute PAs across games of ~9 PAs each (realistic PA-per-game count).
    # This ensures each PA draws from a unique game seed, not just sequential
    # pa_index increments within a single game.
    PAS_PER_GAME = 9
    counts         = Counter()
    bip_total      = 0
    contact_counts = Counter()   # Hard / Medium / Weak
    hard_hr        = 0           # HRs that came off Hard contact

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
            if result["final"] == "HR" and cq == "Hard":
                hard_hr += 1

    # Ordered display
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

    # Quick sanity check vs real baseball expectations
    hits   = counts["HR"] + counts["Triple"] + counts["Double"] + counts["Single"]
    obp    = (hits + counts["BB"] + counts["HBP"]) / total
    slg_num = (counts["Single"] + 2*counts["Double"] + 3*counts["Triple"] + 4*counts["HR"])
    ab      = total - counts["BB"] - counts["HBP"]
    slg     = slg_num / ab if ab > 0 else 0
    ba      = hits / ab if ab > 0 else 0

    babip_num = hits - counts["HR"]
    babip_den = ab - counts["K"] - counts["HR"]
    babip = babip_num / babip_den if babip_den > 0 else 0

    print(f"\n  BA:  {ba:.3f}   OBP: {obp:.3f}   SLG: {slg:.3f}   OPS: {ba+slg:.3f}")
    print(f"  BABIP: {babip:.3f}")
    print(f"  HR%: {counts['HR']/total:.1%}   K%: {counts['K']/total:.1%}   BB%: {counts['BB']/total:.1%}")

    # ── Contact breakdown ──────────────────────────────────────────────────
    print(f"\n  Contact% (BIP / total PA):  {bip_total}/{total} = {bip_total/total:.1%}")
    print(f"\n  Contact Quality (of {bip_total} BIPs):")
    for tier in ("Hard", "Medium", "Weak"):
        cnt = contact_counts[tier]
        pct = cnt / bip_total if bip_total else 0
        bar = "█" * int(pct * 30)
        print(f"    {tier:<7} {cnt:>4}  {pct:>6.1%}  {bar}")

    hard_total = contact_counts["Hard"]
    hr_per_hard = hard_hr / hard_total if hard_total else 0
    print(f"\n  HR / PA:            {counts['HR']}/{total} = {counts['HR']/total:.1%}")
    print(f"  HR / Hard contact:  {hard_hr} HR on {hard_total} hard-hit balls = {hr_per_hard:.1%}")
    print()


if __name__ == "__main__":
    run_series("Babe Ruth", "Waite Hoyt", n=5000)
