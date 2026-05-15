"""
main_app.py — Chin Music Game Runner
Loads a pilot roster JSON and simulates a plate appearance between real players.
"""
import json
import random

from pa_engine import (
    resolve_duel,
    resolve_contact,
    map_bip_outcome,
    resolve_defense,
    derive_game_seed,
    derive_pa_seed,
)

PILOT_FILE     = "pilot_1927_nya.json"
DEFAULT_FIELDER = {"RNG": 65, "HND": 65, "ARM": 65}  # average fielder until defensive cards are built


# ─── Roster helpers ──────────────────────────────────────────────────────────

def load_roster(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def find_player(roster: list[dict], name_or_id: str, role: str = None) -> dict | None:
    """Case-insensitive search by name fragment or exact playerID."""
    query = name_or_id.lower()
    for card in roster:
        match = (query in card["name"].lower() or query == card["player_id"])
        if match and (role is None or card["primary_role"] == role):
            return card
    return None


def list_roster(roster: list[dict]) -> None:
    hitters  = [c for c in roster if c["primary_role"] == "Hitter"]
    pitchers = [c for c in roster if c["primary_role"] == "Pitcher"]

    print("\n── Hitters ──────────────────────────────────────")
    print(f"  {'Name':<22}  POW  EYE   AK  CON  GAP")
    for c in sorted(hitters, key=lambda x: -x["traits"]["POW"]):
        t = c["traits"]
        print(f"  {c['name']:<22}  {t['POW']:>3}  {t['EYE']:>3}  "
              f"{t['AK']:>3}  {t['CON']:>3}  {t['GAP']:>3}")

    print("\n── Pitchers ─────────────────────────────────────")
    print(f"  {'Name':<22} Role  STF  CTL  CMD  STA")
    for c in sorted(pitchers, key=lambda x: -x["traits"]["STF"]):
        t = c["traits"]
        print(f"  {c['name']:<22}  {c['pitcher_role']:<2}   "
              f"{t['STF']:>3}  {t['CTL']:>3}  {t['CMD']:>3}  {t['STA']:>3}")
    print()


# ─── PA simulation ───────────────────────────────────────────────────────────

def simulate_pa(
    batter: dict,
    pitcher: dict,
    fielder: dict = None,
    sim_seed: str = "pilot-game-1",
    game_id: str  = "game-001",
    pa_index: int = 1,
) -> dict:
    """Runs a full PA through all four stages using real player card traits."""
    with open("sim_constants.json") as f:
        constants = json.load(f)

    fielder = fielder or DEFAULT_FIELDER

    batter_traits  = batter["traits"]
    pitcher_traits = pitcher["traits"]
    handedness     = {"batter": batter["bats"], "pitcher": pitcher["throws"]}

    game_seed_int = derive_game_seed(sim_seed, game_id)
    pa_seed       = derive_pa_seed(game_seed_int, pa_index)
    rng           = random.Random(pa_seed)

    duel = resolve_duel(batter_traits, pitcher_traits, handedness, constants, rng)

    contact     = None
    bip_mapping = None
    defense     = None
    final       = duel["outcome"]

    if duel["outcome"] == "BIP":
        contact = resolve_contact(batter_traits, pitcher_traits, constants, rng)
        bip_mapping = map_bip_outcome(
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
            bip_outcome     = bip_mapping["bip_outcome"],
            spray_vector    = contact["spray_vector"],
            contact_quality = contact["contact_quality"],
            fielder         = fielder,
            constants       = constants,
            rng             = rng,
        )
        final = defense["final_outcome"]

    return {
        "final_outcome": final,
        "pa_seed":       pa_seed,
        "duel":          duel,
        "contact":       contact,
        "bip_mapping":   bip_mapping,
        "defense":       defense,
    }


def print_pa_result(batter: dict, pitcher: dict, result: dict) -> None:
    w = 54
    print(f"\n{'═' * w}")
    print(f"  {batter['name']}  vs.  {pitcher['name']}")
    print(f"  {batter['season']} {batter['team_id']} — Plate Appearance")
    print(f"{'─' * w}")

    bt, pt = batter["traits"], pitcher["traits"]
    print(f"  Batter:  {batter['name']:<20} Bats: {batter['bats']}")
    print(f"           POW {bt['POW']}  EYE {bt['EYE']}  CON {bt['CON']}  AK {bt['AK']}")
    print(f"  Pitcher: {pitcher['name']:<20} Throws: {pitcher['throws']}")
    print(f"           STF {pt['STF']}  CTL {pt['CTL']}  CMD {pt['CMD']}  STA {pt['STA']}")
    print(f"{'─' * w}")

    d = result["duel"]
    p = d["probabilities"]
    print(f"  Stage 1 — Duel")
    print(f"    Score: {d['duel_score']:>7.2f}  |  Batter advantage: {d['p_batter_advantage']:.1%}")
    print(f"    K {p['K']:.1%}  BB {p['BB']:.1%}  HBP {p['HBP']:.1%}  BIP {p['BIP']:.1%}")

    if result["contact"]:
        c = result["contact"]
        print(f"\n  Stage 2 — Contact Quality")
        print(f"    {c['contact_quality']} contact to {c['spray_vector']}  "
              f"(score: {c['contact_score']:.1f}  eff_pow: {c['effective_pow']:.1f})")

    if result["bip_mapping"]:
        bm = result["bip_mapping"]
        bp = bm["bip_probabilities"]
        print(f"\n  Stage 2.5 — Outcome Mapping  (pre-defense: {bm['bip_outcome']})")
        print(f"    HR {bp['HR']:.1%}  2B {bp['Double']:.1%}  1B {bp['Single']:.1%}  "
              f"3B {bp['Triple']:.1%}  Out {bp['Out']:.1%}")

    if result["defense"]:
        dr = result["defense"]["defense_resolution"]
        print(f"\n  Stage 3 — Defense")
        print(f"    RNG: {dr['RNG_check']:<12} HND: {dr['HND_check']:<12} ARM: {dr['ARM_check']}")

    print(f"{'═' * w}")
    print(f"  FINAL OUTCOME:  {result['final_outcome']}")
    print(f"{'═' * w}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    roster = load_roster(PILOT_FILE)

    list_roster(roster)

    batter  = find_player(roster, "Lou Gehrig",    role="Hitter")
    pitcher = find_player(roster, "Herb Pennock",  role="Pitcher")

    if not batter or not pitcher:
        print("ERROR: Could not find batter or pitcher in roster.")
        raise SystemExit(1)

    # Run the same PA multiple times with different pa_index to show variance
    print(f"Running 5 plate appearances: {batter['name']} vs. {pitcher['name']}\n")
    for i in range(1, 6):
        result = simulate_pa(batter, pitcher, pa_index=i)
        outcome = result["final_outcome"]
        d = result["duel"]
        contact_str = ""
        if result["contact"]:
            c = result["contact"]
            contact_str = f" → {c['contact_quality']} to {c['contact_str'] if 'contact_str' in c else c['spray_vector']}"
        print(f"  PA #{i}  seed={result['pa_seed']}  |  {outcome:<8} "
              f" BIP%={d['probabilities']['BIP']:.0%}{contact_str}")

    # Full breakdown for PA #1
    result = simulate_pa(batter, pitcher, pa_index=1)
    print_pa_result(batter, pitcher, result)
