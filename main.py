from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv
import os
import json
import random

from pa_engine import resolve_duel, resolve_contact, map_bip_outcome, resolve_defense, derive_game_seed, derive_pa_seed

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Connection successful!")

with open("sim_constants.json") as f:
    sim_constants = json.load(f)
print(f"Simulation Constants Loaded (version: {sim_constants['constants_version']})")

app = FastAPI()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BatterTraits(BaseModel):
    CON: float
    GAP: float
    POW: float
    EYE: float
    AK: float
    BNT: float = 50.0

class PitcherTraits(BaseModel):
    STF: float
    CTL: float
    CMD: float
    STA: float

class Handedness(BaseModel):
    batter: str   # L | R | S
    pitcher: str  # L | R

class DuelRequest(BaseModel):
    sim_seed: str
    game_id: str
    pa_index: int
    batter_traits: BatterTraits
    pitcher_traits: PitcherTraits
    handedness: Handedness

class FielderTraits(BaseModel):
    RNG: float
    HND: float
    ARM: float

class PARequest(BaseModel):
    sim_seed: str
    game_id: str
    pa_index: int
    batter_traits: BatterTraits
    pitcher_traits: PitcherTraits
    handedness: Handedness
    fielder_traits: FielderTraits


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Hello Chin Music"}


@app.post("/simulate/duel")
def simulate_duel(req: DuelRequest):
    game_seed = derive_game_seed(req.sim_seed, req.game_id)
    pa_seed   = derive_pa_seed(game_seed, req.pa_index)
    rng       = random.Random(pa_seed)

    result = resolve_duel(
        batter=req.batter_traits.model_dump(),
        pitcher=req.pitcher_traits.model_dump(),
        handedness=req.handedness.model_dump(),
        constants=sim_constants,
        rng=rng,
    )

    return {
        "sim_seed": req.sim_seed,
        "game_id":  req.game_id,
        "pa_index": req.pa_index,
        "pa_seed":  pa_seed,
        **result,
    }


@app.post("/simulate/pa")
def simulate_pa(req: PARequest):
    """
    Full PA resolution: Stage 1 → Stage 2 → Stage 2.5 → Stage 3.
    The same seeded RNG instance advances through every stage so the entire
    PA is deterministic from a single pa_seed.
    """
    game_seed = derive_game_seed(req.sim_seed, req.game_id)
    pa_seed   = derive_pa_seed(game_seed, req.pa_index)
    rng       = random.Random(pa_seed)

    batter  = req.batter_traits.model_dump()
    pitcher = req.pitcher_traits.model_dump()
    fielder = req.fielder_traits.model_dump()

    duel = resolve_duel(
        batter=batter,
        pitcher=pitcher,
        handedness=req.handedness.model_dump(),
        constants=sim_constants,
        rng=rng,
    )

    contact      = None
    bip_mapping  = None
    defense      = None
    final_outcome = duel["outcome"]   # K / BB / HBP pass straight through

    if duel["outcome"] == "BIP":
        contact = resolve_contact(
            batter=batter,
            pitcher=pitcher,
            constants=sim_constants,
            rng=rng,
        )
        bip_mapping = map_bip_outcome(
            contact_quality=contact["contact_quality"],
            contact_score=contact["contact_score"],
            spray_vector=contact["spray_vector"],
            effective_pow=contact["effective_pow"],
            batter=batter,
            pitcher=pitcher,
            constants=sim_constants,
            rng=rng,
        )
        defense = resolve_defense(
            bip_outcome=bip_mapping["bip_outcome"],
            spray_vector=contact["spray_vector"],
            contact_quality=contact["contact_quality"],
            fielder=fielder,
            constants=sim_constants,
            rng=rng,
        )
        final_outcome = defense["final_outcome"]

    return {
        "final_outcome":            final_outcome,
        "sim_seed":                 req.sim_seed,
        "game_id":                  req.game_id,
        "pa_index":                 req.pa_index,
        "pa_seed":                  pa_seed,
        "stage_1_duel":             duel,
        "stage_2_contact":          contact,
        "stage_25_outcome_mapping": bip_mapping,
        "stage_3_defense":          defense,
    }
