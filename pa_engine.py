"""
PA Engine — Core Simulation Brain (PRD 01)
Stage 1: Duel (K / BB / HBP / BIP)
Stage 2: Contact Quality (Weak / Medium / Hard + spray)
"""

import math
import random
import hashlib


# Normalizes raw duel score into logistic input range.
# Raw D spans roughly -100 to +100; dividing by 15 keeps P(batter_advantage)
# well away from 0 and 1 for all realistic trait matchups.
DUEL_SCALE = 15.0

# Clamp bounds applied to scaled D before the logistic function.
# Prevents float saturation at extreme trait mismatches while still
# allowing meaningful probability differences across the full trait range.
DUEL_LOGISTIC_CLAMP = 10.0

PROB_FLOOR = 0.005   # No outcome can ever be 0% (per PRD constraints)
K_CEILING  = 0.55
BB_CEILING = 0.25
HBP_CEILING = 0.030


# ---------------------------------------------------------------------------
# Seeding (Section 7 — Deterministic Seeding)
# ---------------------------------------------------------------------------

def _sha_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % (2 ** 32)

def derive_game_seed(sim_seed: str, game_id: str) -> int:
    return _sha_int(f"{sim_seed}:{game_id}")

def derive_pa_seed(game_seed: int, pa_index: int) -> int:
    return _sha_int(f"{game_seed}:{pa_index}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def _platoon_modifier(batter_hand: str, pitcher_hand: str, delta: float) -> float:
    """
    Applies handedness advantage to the raw duel score (Section 6).
    Switch hitters always face the favorable side (opposite of pitcher).
    Same-hand = pitcher advantage (-delta); opposite-hand = batter advantage (+delta).
    """
    effective = "R" if pitcher_hand == "L" else "L" if batter_hand == "S" else batter_hand
    return delta if effective != pitcher_hand else -delta


# ---------------------------------------------------------------------------
# Stage 1 — Duel
# ---------------------------------------------------------------------------

def resolve_duel(
    batter: dict,
    pitcher: dict,
    handedness: dict,
    constants: dict,
    rng: random.Random,
) -> dict:
    """
    Resolves a single plate appearance through Stage 1 (Duel).

    Args:
        batter:     trait dict — must contain CON, EYE, AK
        pitcher:    trait dict — must contain STF, CTL
        handedness: {"batter": "L|R|S", "pitcher": "L|R"}
        constants:  loaded sim_constants.json
        rng:        seeded random.Random (provides determinism)

    Returns:
        {
            "outcome": "K" | "BB" | "HBP" | "BIP",
            "duel_score": float,
            "p_batter_advantage": float,
            "probabilities": {"K": float, "BB": float, "HBP": float, "BIP": float}
        }
    """
    w1 = constants["duel_weights"]["w1_con_stf"]
    w2 = constants["duel_weights"]["w2_eye_ctl"]
    platoon_delta = constants.get("platoon_advantage_delta", 4.0)
    k_base  = constants["league_avg_k_pct"]
    bb_base = constants["league_avg_bb_pct"]

    CON = batter["CON"]
    EYE = batter["EYE"]
    AK  = batter["AK"]
    STF = pitcher["STF"]
    CTL = pitcher["CTL"]

    platoon_mod = _platoon_modifier(
        handedness["batter"], handedness["pitcher"], platoon_delta
    )

    # Duel score (Section 4, Stage 1)
    D_raw = w1 * (CON - STF) + w2 * (EYE - CTL) + platoon_mod
    D = _clamp(D_raw / DUEL_SCALE, -DUEL_LOGISTIC_CLAMP, DUEL_LOGISTIC_CLAMP)
    p_batter_adv = _logistic(D)

    # K% — driven by STF-CON edge; AK suppresses strikeouts with interaction effect.
    # AK is normalized over its meaningful above-average range (50–100 → 0.0–1.0).
    # When the batter already has a CON advantage, high AK amplifies that suppression
    # further — a contact-skilled hitter punishes weak stuff even more.
    stf_con_edge  = (STF - CON) / 100.0
    ak_normalized = max(0.0, (AK - 50) / 50.0)          # 0.0–1.0; below 50 = no bonus
    batter_edge   = max(0.0, -stf_con_edge)              # how much CON exceeds STF
    ak_suppression = ak_normalized * (0.08 + batter_edge * 0.08)
    k_pct = _clamp(k_base + stf_con_edge * 0.15 - ak_suppression, PROB_FLOOR, K_CEILING)

    # BB% — driven purely by EYE-CTL; POW has zero influence (per PRD)
    eye_ctl_edge = (EYE - CTL) / 100.0
    bb_pct = _clamp(bb_base + eye_ctl_edge * 0.10, PROB_FLOOR, BB_CEILING)

    # HBP% — small baseline; rises when pitcher CTL is below average
    ctl_wildness = max(0.0, (50 - CTL) / 100.0)
    hbp_pct = _clamp(0.010 + ctl_wildness * 0.01, PROB_FLOOR, HBP_CEILING)

    # BIP% = remainder; renormalize so all four always sum to exactly 1.0
    bip_pct = max(PROB_FLOOR, 1.0 - k_pct - bb_pct - hbp_pct)
    total = k_pct + bb_pct + hbp_pct + bip_pct
    k_pct   /= total
    bb_pct  /= total
    hbp_pct /= total
    bip_pct /= total

    # Resolve outcome using seeded RNG roll
    roll = rng.random()
    cumulative = 0.0
    outcome = "BIP"
    for label, prob in [("K", k_pct), ("BB", bb_pct), ("HBP", hbp_pct), ("BIP", bip_pct)]:
        cumulative += prob
        if roll < cumulative:
            outcome = label
            break

    return {
        "outcome": outcome,
        "duel_score": round(D_raw, 4),
        "p_batter_advantage": round(p_batter_adv, 4),
        "probabilities": {
            "K":   round(k_pct, 4),
            "BB":  round(bb_pct, 4),
            "HBP": round(hbp_pct, 4),
            "BIP": round(bip_pct, 4),
        },
    }


# ---------------------------------------------------------------------------
# Stage 2 — Contact Quality
# ---------------------------------------------------------------------------

# Contact score thresholds (Q is on a 0–100 scale after sigmoid normalization)
_HARD_THRESHOLD   = 62.0
_WEAK_THRESHOLD   = 38.0

# How strongly CMD variance spreads outcomes.
# At CMD=0: sigma ≈ 5.0; at CMD=100: sigma ≈ 0.0 (perfectly tight CMD).
_CMD_VARIANCE_DIVISOR = 20.0

# Sigmoid normalization parameters for the raw contact score.
# The raw Q is centered around 50 (average matchup ≈ 46–50).
# Scale of 12 gives a smooth spread: Q_raw=80 → ~92, Q_raw=20 → ~8.
# This prevents pile-up at hard clamp boundaries and compresses extreme values gracefully.
_CONTACT_SIGMOID_CENTER = 50.0
_CONTACT_SIGMOID_SCALE  = 12.0


def _resolve_spray(pow_val: float, gap_val: float, quality: str, rng: random.Random) -> str:
    """
    Determines spray direction (Pull / Center / Oppo) based on contact quality
    and batter POW / GAP traits.

    - High POW shifts toward Pull, especially on Hard contact.
    - High GAP shifts toward Center/Oppo (gap power = all-fields hitter).
    - Weak contact flattens the distribution (beaten up balls go anywhere).
    """
    if quality == "Hard":
        pull, center, oppo = 0.50, 0.30, 0.20
    elif quality == "Medium":
        pull, center, oppo = 0.40, 0.35, 0.25
    else:  # Weak
        pull, center, oppo = 0.35, 0.35, 0.30

    pull  = _clamp(pull  + (pow_val - 50) / 100.0 * 0.12, 0.10, 0.75)
    oppo  = _clamp(oppo  + (gap_val - 50) / 100.0 * 0.10, 0.10, 0.50)
    center = max(0.05, 1.0 - pull - oppo)

    total = pull + center + oppo
    pull   /= total
    center /= total
    oppo   /= total

    roll = rng.random()
    if roll < pull:
        return "Pull"
    elif roll < pull + center:
        return "Center"
    return "Oppo"


def resolve_contact(
    batter: dict,
    pitcher: dict,
    constants: dict,
    rng: random.Random,
) -> dict:
    """
    Stage 2 — Contact Quality: resolves Weak / Medium / Hard contact + spray direction.
    Only called when Stage 1 outcome is BIP.

    PRD formula: Q = a*POW + b*CON + d*GAP - c*STF + CMD_variance_term
    PRD rules:
      - POW is gated by CON (high POW + low CON = inconsistent hard contact)
      - CMD tightens distribution; low CMD = higher variance
      - POW cannot affect K% or BB% — those are Duel-only

    Args:
        batter:    trait dict — must contain POW, CON, GAP
        pitcher:   trait dict — must contain STF, CMD
        constants: loaded sim_constants.json
        rng:       seeded random.Random (same instance, advanced from Stage 1)

    Returns:
        {
            "contact_score": float,
            "contact_quality": "Weak" | "Medium" | "Hard",
            "spray_vector": "Pull" | "Center" | "Oppo",
            "effective_pow": float,
            "cmd_noise": float,
        }
    """
    weights = constants["contact_quality_weights"]
    a = weights["a_pow"]
    b = weights["b_con"]
    d = weights["d_gap"]
    c = weights["c_stf"]

    POW = batter["POW"]
    CON = batter["CON"]
    GAP = batter["GAP"]
    STF = pitcher["STF"]
    CMD = pitcher["CMD"]

    # POW gated by CON: logistic gate centered at CON=50.
    # CON=95 → gate≈0.97 (nearly full POW); CON=50 → gate=0.50; CON=30 → gate≈0.18.
    con_gate = _logistic((CON - 50) / 15.0)
    effective_pow = POW * con_gate

    # Raw contact score (unbounded; batter side is a weighted avg since a+b+d=1.0)
    Q_raw = a * effective_pow + b * CON + d * GAP - c * STF

    # CMD variance: low CMD = wider spread of outcomes (more mistakes, more weak contact)
    cmd_sigma = (100.0 - CMD) / _CMD_VARIANCE_DIVISOR
    cmd_noise = rng.gauss(0.0, cmd_sigma) if cmd_sigma > 0 else 0.0
    Q_noisy = Q_raw + cmd_noise

    # Sigmoid normalization → bounded 0–100.
    # Smoothly compresses extreme values instead of hard-clamping at boundaries.
    Q_final = 100.0 * _logistic((Q_noisy - _CONTACT_SIGMOID_CENTER) / _CONTACT_SIGMOID_SCALE)

    # Map score to contact quality tier
    if Q_final >= _HARD_THRESHOLD:
        quality = "Hard"
    elif Q_final >= _WEAK_THRESHOLD:
        quality = "Medium"
    else:
        quality = "Weak"

    spray = _resolve_spray(POW, GAP, quality, rng)

    return {
        "contact_score":   round(Q_final, 4),
        "contact_quality": quality,
        "spray_vector":    spray,
        "effective_pow":   round(effective_pow, 4),
        "cmd_noise":       round(cmd_noise, 4),
    }


# ---------------------------------------------------------------------------
# Stage 2.5 — BIP Outcome Mapping (Contact → HR/Triple/Double/Single/Out)
# ---------------------------------------------------------------------------

# Probability floors — no BIP outcome can be fully impossible.
_BIP_FLOOR: dict[str, float] = {
    "HR": 0.003, "Triple": 0.001, "Double": 0.010, "Single": 0.030, "Out": 0.030,
}

# Base distributions by contact tier.
# These anchor the simulation at realistic BABIP/XBH rates before trait modifiers.
# Triples are intentionally rare; removed probability redistributed to Doubles.
_BIP_BASE: dict[str, dict[str, float]] = {
    "Hard":   {"HR": 0.22, "Triple": 0.008, "Double": 0.215, "Single": 0.260, "Out": 0.317},
    "Medium": {"HR": 0.04, "Triple": 0.005, "Double": 0.140, "Single": 0.340, "Out": 0.475},
    "Weak":   {"HR": 0.01, "Triple": 0.002, "Double": 0.028, "Single": 0.200, "Out": 0.760},
}


def map_bip_outcome(
    contact_quality: str,
    contact_score: float,
    spray_vector: str,
    effective_pow: float,
    batter: dict,
    constants: dict,
    rng: random.Random,
) -> dict:
    """
    Stage 2.5 — BIP Outcome Mapping.
    Converts a batted ball (contact quality + score + spray) into a final
    pre-defense outcome distribution: HR / Triple / Double / Single / Out.

    Sits between Stage 2 (Contact) and Stage 3 (Defense). Defense will later
    convert Out → potential Error and adjust hit types by fielder traits.

    HR formula (nonlinear / diminishing returns):
      hr_driver = effective_pow * 0.6 + contact_score * 0.4
      hr_norm   = hr_driver / 100
      hr_prob   = 0.04 + (hr_norm ** 2) * 0.18
    This caps HR at ~22% for perfect traits rather than allowing 30%+ runaway rates.

    Args:
        contact_quality: "Weak" | "Medium" | "Hard"
        contact_score:   0–100 (sigmoid-normalized from Stage 2)
        spray_vector:    "Pull" | "Center" | "Oppo"
        effective_pow:   CON-gated POW from Stage 2 (passed through to avoid recompute)
        batter:          trait dict — must contain GAP
        constants:       loaded sim_constants.json
        rng:             seeded random.Random (same instance, advanced from Stage 2)

    Returns:
        {
            "bip_outcome": "HR" | "Triple" | "Double" | "Single" | "Out",
            "bip_probabilities": {HR, Triple, Double, Single, Out},
            "hr_driver": float  (exposed for debugging)
        }
    """
    GAP = batter["GAP"]

    base = dict(_BIP_BASE[contact_quality])

    # HR: nonlinear diminishing returns — squares hr_norm so extreme traits
    # compress rather than produce runaway rates. Caps at ~19% for perfect inputs.
    hr_driver = effective_pow * 0.6 + contact_score * 0.4
    hr_norm   = hr_driver / 100.0
    hr_prob   = 0.035 + (hr_norm ** 2) * 0.155

    # XBH: GAP drives doubles and triples. Oppo/Center spray amplifies Triple.
    gap_edge           = (GAP - 50) / 50.0              # -1 to +1
    xbh_mod            = gap_edge * 0.07
    # Triple spray bonus tightened — keeps triples rare even on favorable spray
    triple_spray_bonus = 0.004 if spray_vector in ("Center", "Oppo") else -0.002

    # Apply all modifiers; HR is fully replaced by the nonlinear formula above.
    # Triple GAP multiplier reduced to 0.10 (was 0.35); excess redirected to Double.
    probs = {
        "HR":     hr_prob,
        "Triple": base["Triple"] + xbh_mod * 0.10 + triple_spray_bonus,
        "Double": base["Double"] + xbh_mod,
        "Single": base["Single"],
        "Out":    base["Out"],
    }

    # Enforce floors and renormalize so probabilities always sum to exactly 1.0
    for label, floor in _BIP_FLOOR.items():
        probs[label] = max(floor, probs[label])
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    # Sample outcome using the seeded RNG
    roll = rng.random()
    cumulative = 0.0
    bip_outcome = "Out"
    for label in ("HR", "Triple", "Double", "Single", "Out"):
        cumulative += probs[label]
        if roll < cumulative:
            bip_outcome = label
            break

    return {
        "bip_outcome":       bip_outcome,
        "bip_probabilities": {k: round(v, 4) for k, v in probs.items()},
        "hr_driver":         round(hr_driver, 4),
    }


# ---------------------------------------------------------------------------
# Stage 3 — Defense
# ---------------------------------------------------------------------------

# Contact quality adjusts how hard it is for a fielder to execute.
# Hard liners give less reaction time (lower p_catch for Outs);
# weak pop-ups and grounders are easier to handle.
_CONTACT_QUALITY_FACTOR: dict[str, float] = {
    "Hard": -0.06, "Medium": 0.0, "Weak": 0.08
}


def resolve_defense(
    bip_outcome: str,
    spray_vector: str,
    contact_quality: str,
    fielder: dict,
    constants: dict,
    rng: random.Random,
) -> dict:
    """
    Stage 3 — Defense: applies fielder traits to the pre-defense BIP outcome.

    Sequence (per PRD Section 4, Stage 3):
      1. HR  → skips defense entirely. Final outcome is HR.
      2. RNG → does the fielder reach the ball?
               - Pre-defense Out:    high RNG holds it as Out; miss → drops in as Single
               - Pre-defense Single: elite RNG can convert to Out (diving catch)
      3. HND → if fielder reaches, is the play clean?
               - Very low HND produces Error on any reached ball
      4. ARM → on extra-base hits, high ARM holds runners at fewer bases
               - Triple → Double possible
               - Double → Single possible

    Args:
        bip_outcome:     pre-defense outcome from Stage 2.5
        spray_vector:    "Pull" | "Center" | "Oppo"
        contact_quality: "Weak" | "Medium" | "Hard"
        fielder:         trait dict — must contain RNG, HND, ARM
        constants:       loaded sim_constants.json (reserved for future tuning)
        rng:             seeded random.Random (same instance, advanced from Stage 2.5)

    Returns:
        {
            "final_outcome": "HR"|"Triple"|"Double"|"Single"|"Out"|"Error",
            "defense_resolution": {
                "RNG_check": str, "HND_check": str, "ARM_check": str, "result": str
            }
        }
    """
    RNG_trait = fielder["RNG"]
    HND       = fielder["HND"]
    ARM       = fielder["ARM"]

    rng_norm = (RNG_trait - 50) / 50.0   # -1.0 to +1.0
    hnd_norm = (HND       - 50) / 50.0
    arm_norm = (ARM       - 50) / 50.0
    cq_factor = _CONTACT_QUALITY_FACTOR.get(contact_quality, 0.0)

    rng_check = "skipped"
    hnd_check = "skipped"
    arm_check = "skipped"

    # ── HR: defense irrelevant ──────────────────────────────────────────────
    if bip_outcome == "HR":
        return {
            "final_outcome": "HR",
            "defense_resolution": {
                "RNG_check": "skipped",
                "HND_check": "skipped",
                "ARM_check": "skipped",
                "result":    "HR",
            },
        }

    # ── Out ─────────────────────────────────────────────────────────────────
    if bip_outcome == "Out":
        # Ball heading toward the fielder — does he convert it?
        # Harder hit balls (liners) are trickier despite being "outs" by trajectory.
        p_catch = _clamp(0.55 + rng_norm * 0.35 + cq_factor, 0.50, 0.97)
        if rng.random() < p_catch:
            rng_check = "reached"
            # HND check: muffed ball → Error
            p_error = _clamp(0.025 - hnd_norm * 0.020, 0.003, 0.08)
            if rng.random() < p_error:
                hnd_check = "error"
                final_outcome = "Error"
            else:
                hnd_check = "clean"
                final_outcome = "Out"
        else:
            rng_check = "not_reached"
            hnd_check = "skipped"
            final_outcome = "Single"   # ball drops in

    # ── Single ──────────────────────────────────────────────────────────────
    elif bip_outcome == "Single":
        # Elite RNG can convert a Single to an Out (diving/rangy play).
        p_convert = _clamp(0.07 + rng_norm * 0.10 + cq_factor, 0.02, 0.22)
        if rng.random() < p_convert:
            rng_check = "reached"
            hnd_check = "clean"
            final_outcome = "Out"
        else:
            rng_check = "not_reached"
            # HND check: muffed grounder → Error
            p_error = _clamp(0.030 - hnd_norm * 0.025, 0.004, 0.10)
            if rng.random() < p_error:
                hnd_check = "error"
                final_outcome = "Error"
            else:
                hnd_check = "clean"
                final_outcome = "Single"

    # ── Double ──────────────────────────────────────────────────────────────
    elif bip_outcome == "Double":
        rng_check = "not_reached"   # fielder already conceded the gap
        # ARM check: strong arm holds runner, effectively turns Double → Single
        arm_check = "checked"
        p_hold = _clamp(0.06 + arm_norm * 0.12, 0.01, 0.25)
        if rng.random() < p_hold:
            arm_check = "held"
            final_outcome = "Single"
        else:
            arm_check = "not_held"
            # HND check: bobble in the outfield → Error
            p_error = _clamp(0.018 - hnd_norm * 0.015, 0.002, 0.055)
            if rng.random() < p_error:
                hnd_check = "error"
                final_outcome = "Error"
            else:
                hnd_check = "clean"
                final_outcome = "Double"

    # ── Triple ──────────────────────────────────────────────────────────────
    elif bip_outcome == "Triple":
        rng_check = "not_reached"
        arm_check = "checked"
        # Strong arm cuts down the runner rounding second, holds to Double
        p_hold = _clamp(0.15 + arm_norm * 0.15, 0.03, 0.40)
        if rng.random() < p_hold:
            arm_check = "held"
            final_outcome = "Double"
        else:
            arm_check = "not_held"
            hnd_check  = "clean"
            final_outcome = "Triple"

    else:
        final_outcome = bip_outcome   # safety fallback

    return {
        "final_outcome": final_outcome,
        "defense_resolution": {
            "RNG_check": rng_check,
            "HND_check": hnd_check,
            "ARM_check": arm_check,
            "result":    final_outcome,
        },
    }
