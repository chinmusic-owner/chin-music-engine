"""
pa_wrapper.py — Deterministic PA resolution wrapper around pa_engine.py stages.

Exposes:
    resolve_pa_seeded(batter, pitcher, context=None, seed=0) -> dict

Caller contract:
    batter  — player card dict with keys: "traits" (flat trait dict), "bats", "throws"
    pitcher — player card dict with keys: "traits", "bats", "throws"
    context — optional dict; supports "fielder" (RNG/HND/ARM) and "constants" override
    seed    — integer seed passed directly to random.Random for full determinism

Returns a standardized dict:
    {
        "outcome":      final resolved outcome string,
        "seed":         seed used,
        "duel":         full duel stage result,
        "contact":      contact stage result (None if outcome not BIP),
        "bip_map":      BIP outcome mapping result (None if outcome not BIP),
        "defense":      defense stage result (None if outcome not BIP),
    }
"""

import json
import os
import random
import warnings

from pa_engine import (
    resolve_duel,
    resolve_contact,
    map_bip_outcome,
    resolve_defense,
)

_CONSTANTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_constants.json")
_DEFAULT_FIELDER = {"RNG": 50, "HND": 50, "ARM": 50}

# (db_column, engine_key) pairs for fielding traits
_FIELDER_TRAIT_MAP = (("rng", "RNG"), ("hnd", "HND"), ("arm", "ARM"))


def fielder_from_row(row: dict) -> dict:
    """Convert a Supabase ``players`` row to a ``{RNG, HND, ARM}`` fielder dict.

    Each trait falls back to 50 (league average) if the column is absent or
    null, and emits a ``UserWarning`` so callers know a default was applied.

    Args:
        row: A dict representing one row from the ``players`` table — typically
             the result of a Supabase ``select("*")`` call.

    Returns:
        ``{"RNG": int, "HND": int, "ARM": int}`` ready to pass as
        ``context["fielder"]`` in ``resolve_pa_seeded``.
    """
    label = row.get("player_name") or row.get("player_id") or "unknown"
    out: dict[str, int] = {}
    for db_key, eng_key in _FIELDER_TRAIT_MAP:
        val = row.get(db_key)
        if val is None:
            warnings.warn(
                f"fielder_from_row: '{db_key}' is null for player '{label}' "
                f"— defaulting to 50",
                UserWarning,
                stacklevel=2,
            )
            val = 50
        out[eng_key] = int(val)
    return out

# Load constants once at import time; callers can override via context["constants"].
with open(_CONSTANTS_PATH) as _f:
    _CONSTANTS = json.load(_f)


def resolve_pa_seeded(
    batter: dict,
    pitcher: dict,
    context: dict | None = None,
    seed: int = 0,
) -> dict:
    """
    Resolves a single plate appearance deterministically.

    Args:
        batter:  Player card dict (must have "traits" dict + "bats" str).
        pitcher: Player card dict (must have "traits" dict + "throws" str).
        context: Optional runtime overrides:
                     "fielder"   — {RNG, HND, ARM} (defaults to league-avg 50/50/50)
                     "constants" — full sim_constants dict (defaults to sim_constants.json)
        seed:    Integer seed for random.Random — same seed always produces same outcome.

    Returns:
        Standardized dict with outcome + full stage metadata.
    """
    ctx       = context or {}
    constants = ctx.get("constants", _CONSTANTS)
    fielder   = ctx.get("fielder", _DEFAULT_FIELDER)

    rng = random.Random(seed)

    batter_traits  = batter["traits"]
    pitcher_traits = pitcher["traits"]
    handedness     = {
        "batter":  batter.get("bats",   "R"),
        "pitcher": pitcher.get("throws", "R"),
    }

    # Stage 1: Duel — resolves K / BB / HBP / BIP
    duel = resolve_duel(batter_traits, pitcher_traits, handedness, constants, rng)

    if duel["outcome"] != "BIP":
        return {
            "outcome": duel["outcome"],
            "seed":    seed,
            "duel":    duel,
            "contact": None,
            "bip_map": None,
            "defense": None,
        }

    # Stage 2: Contact Quality — Weak / Medium / Hard + spray
    contact = resolve_contact(batter_traits, pitcher_traits, constants, rng)

    # Stage 2.5: BIP Outcome Mapping — HR gate + non-HR multinomial
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

    # Stage 3: Defense — RNG / HND / ARM checks
    defense = resolve_defense(
        bip_outcome     = bip_map["bip_outcome"],
        spray_vector    = contact["spray_vector"],
        contact_quality = contact["contact_quality"],
        fielder         = fielder,
        constants       = constants,
        rng             = rng,
    )

    return {
        "outcome": defense["final_outcome"],
        "seed":    seed,
        "duel":    duel,
        "contact": contact,
        "bip_map": bip_map,
        "defense": defense,
    }
