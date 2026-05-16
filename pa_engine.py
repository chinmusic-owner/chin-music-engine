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

# Global BABIP calibration multiplier (Stage 2.5 — non-HR BIP outcomes).
# Applied to Single/Double/Triple probabilities before renormalization.
# 1.0 = no adjustment. Raise to increase BABIP; lower to decrease it.
# Calibrated so that Stone (1906) vs Falkenberg (1906) produces BABIP ≈ .300.
GLOBAL_BABIP_ADJUST = 0.82


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
#
# Tier classification uses DIRECT probability sampling from bounded logistic
# functions, not fixed thresholds on a normalized score.  This naturally
# enforces soft floors/ceilings across the entire trait space:
#
#   P(Hard) ∈ [_HARD_FLOOR, _HARD_CEIL]  — never hits 0% or 100%
#   P(Weak) ∈ [_WEAK_FLOOR, _WEAK_CEIL]
#   P(Medium) = 1 − P(Hard) − P(Weak)   (renormalized, always > 0)
#
# Both curves are centered on Q_noisy (Q_raw + CMD noise).  The Hard logistic
# rises with Q_noisy; the Weak logistic falls with Q_noisy.
#
#   P(Hard) = FLOOR + (CEIL−FLOOR) × σ((Q_noisy − MID) / SCALE)
#   P(Weak) = FLOOR + (CEIL−FLOOR) × σ((MID_WEAK  − Q_noisy) / SCALE)
#
# Calibrated so that:
#   • Ruth vs avg pitcher   →  ~45% Hard
#   • Ruth vs elite pitcher →  ~20% Hard   (above floor)
#   • Avg hitter vs Hoyt    →  ~10% Hard   (at floor)
#   • Contact hitter vs Hoyt → ~30% Hard

_HARD_FLOOR   = 0.10
_HARD_CEIL    = 0.60
_HARD_Q_MID   = 44.0    # Q_noisy where P(Hard) = midpoint (35%)
_HARD_Q_SCALE = 3.0     # steepness; smaller = sharper cliff (DO NOT go below 2.5)

_WEAK_FLOOR   = 0.03
_WEAK_CEIL    = 0.40
_WEAK_Q_MID   = 35.0    # Q_noisy where P(Weak) = midpoint (21.5%)
_WEAK_Q_SCALE = 3.0

# CMD variance: low CMD = wider spread. Floor prevents elite pitchers from
# eliminating variance entirely, which would cause Hard% to collapse to the floor.
_CMD_VARIANCE_DIVISOR = 13.0
_CMD_SIGMA_FLOOR      = 2.5

# Sigmoid for Q_final (contact_score) — used by Stage 2.5 HR gate only.
# Not used for tier classification any more.
_CONTACT_SIGMOID_CENTER = 37.0
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
    # Shift center from 50 → 45 so CON 50–60 hitters lose ~35–40% of POW
    # instead of 40–50%.  Elite CON (80+) is barely affected; low-end CON still penalized.
    con_gate = _logistic((CON - 45) / 15.0)
    effective_pow = POW * con_gate

    # Raw contact score (unbounded; batter side is a weighted avg since a+b+d=1.0)
    Q_raw = a * effective_pow + b * CON + d * GAP - c * STF

    # CMD variance: low CMD = wider spread. Floor prevents sigma from collapsing
    # for elite pitchers, which would create an artificial Hard% cliff.
    cmd_sigma = max((100.0 - CMD) / _CMD_VARIANCE_DIVISOR, _CMD_SIGMA_FLOOR)
    cmd_noise = rng.gauss(0.0, cmd_sigma)
    Q_noisy   = Q_raw + cmd_noise

    # Sigmoid normalization for contact_score (used by Stage 2.5 HR gate).
    Q_final = 100.0 * _logistic((Q_noisy - _CONTACT_SIGMOID_CENTER) / _CONTACT_SIGMOID_SCALE)

    # ── Soft-bounded tier sampling ──────────────────────────────────────
    # Each tier probability is a bounded logistic of Q_noisy, ensuring
    # Hard% stays in [10%, 60%] and Weak% in [3%, 40%] for any matchup.
    p_hard   = _HARD_FLOOR + (_HARD_CEIL - _HARD_FLOOR) * _logistic(
                   (Q_noisy - _HARD_Q_MID) / _HARD_Q_SCALE)
    p_weak   = _WEAK_FLOOR + (_WEAK_CEIL - _WEAK_FLOOR) * _logistic(
                   (_WEAK_Q_MID - Q_noisy) / _WEAK_Q_SCALE)
    p_medium = max(0.0, 1.0 - p_hard - p_weak)

    # Renormalize in case floating-point pushes sum above 1.0
    _tier_sum = p_hard + p_medium + p_weak
    p_hard   /= _tier_sum
    p_medium /= _tier_sum
    p_weak   /= _tier_sum

    roll = rng.random()
    if roll < p_hard:
        quality = "Hard"
    elif roll < p_hard + p_medium:
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
# Stage 2.5 — BIP Outcome Mapping
# ---------------------------------------------------------------------------
#
# Two-step process:
#   Step 1 — HR gate: nonlinear function of effective_pow + contact_score.
#             Fires only on Medium or Hard contact (Weak → hr_prob = 0).
#   Step 2 — Non-HR multinomial: tier-based table over Out/1B/2B/3B/ROE,
#             modified by pitcher suppression and batter GAP/spray.
#
# Tier non-HR base probabilities (sum to 1.0):
#   Weak:   Out 0.830  1B 0.140  2B 0.020  3B 0.005  ROE 0.005  (+0.02 to Single, -0.02 Out)
#   Medium: Out 0.760  1B 0.160  2B 0.065  3B 0.008  ROE 0.007  (-0.04 Out, +0.03 Single, +0.01 Double)
#   Hard:   Out 0.650  1B 0.140  2B 0.180  3B 0.015  ROE 0.015  (unchanged)
_NON_HR_BASE: dict[str, dict[str, float]] = {
    "Weak":   {"Out": 0.830, "Single": 0.140, "Double": 0.020, "Triple": 0.005, "Error": 0.005},
    "Medium": {"Out": 0.780, "Single": 0.145, "Double": 0.060, "Triple": 0.008, "Error": 0.007},
    "Hard":   {"Out": 0.650, "Single": 0.140, "Double": 0.180, "Triple": 0.015, "Error": 0.015},
}

# Pitcher suppression scaling — bonus added to Out weight before renormalization.
# Formula: bonus = max(0, (trait - 50) / 50) * _PITCHER_SUP_SCALE
# At trait=70: bonus ≈ 0.06  |  At trait=90: bonus ≈ 0.12  |  capped at 0.15
_PITCHER_SUP_SCALE = 0.12   # CMD suppresses Medium/Weak Out; STF suppresses Hard Out

# GAP shifts probability from Out toward Double; capped so totals stay sane.
_GAP_DOUBLE_MOD  = 0.04   # max ±4% on Double at extreme GAP
# Spray bonus on Triple for Center/Oppo hits (gap power into the alleys).
_SPRAY_TRIPLE_BONUS = 0.003


def map_bip_outcome(
    contact_quality: str,
    contact_score: float,
    spray_vector: str,
    effective_pow: float,
    batter: dict,
    pitcher: dict,
    constants: dict,
    rng: random.Random,
) -> dict:
    """
    Stage 2.5 — BIP Outcome Mapping.

    Step 1 — HR gate (Medium/Hard only):
        pow_norm     = POW / 100                          (raw POW — not CON-gated)
        cs_norm      = contact_score / 100
        hr_base      = 0.08 + (pow_norm ** 2.5) * 0.18 * (0.6 + 0.4 * cs_norm)
        quality_mult = 1.0 (Hard) | 0.25 (Medium)
        hr_prob      = quality_mult * hr_base
        Weak contact → hr_prob = 0.

        Calibration targets:
            POW 50  → HR/Hard ~8–10 %
            POW 75  → HR/Hard ~12–15 %
            POW 95+ → HR/Hard ~18–25 %

    Step 2 — Non-HR multinomial sampled from _NON_HR_BASE[tier] with:
        - Pitcher suppression: CMD raises Out% on Medium/Weak; STF raises Out% on Hard.
          Formula: bonus = max(0, (trait - 50) / 50) * _PITCHER_SUP_SCALE
          Excess Out weight is taken proportionally from all hit columns.
        - Batter GAP: shifts weight from Out toward Double.
        - Spray: Center/Oppo nudges a sliver of Double into Triple.

    Returns:
        {
            "bip_outcome":       str,
            "bip_probabilities": dict,   # full pre-defense distribution including HR gate
            "hr_driver":         float,
        }
    """
    GAP = batter["GAP"]
    POW = batter["POW"]
    CMD = pitcher["CMD"]
    STF = pitcher["STF"]

    # ── Step 1: HR gate ──────────────────────────────────────────────────
    # Raw POW (not CON-gated) drives HR conversion — clearing the fence is a
    # function of raw power, not contact consistency.
    # POW^2.5 tail gives strong separation: POW 50 ≈ 9%, POW 75 ≈ 13%, POW 99 ≈ 23%.
    # Contact score boosts HR% on sharper contact (better-hit ball = more carry).
    # Medium contact can still produce HRs but at 25% the rate of Hard contact.
    pow_norm = POW / 100.0
    cs_norm  = contact_score / 100.0
    # HR floor scales with POW so low-power batters cannot clear the fence at
    # 8% per Hard BIP.  At POW=100 the total is identical to the original
    # formula (0.02 + 0.06 = 0.08); at POW=40 the floor drops to 0.044.
    hr_floor = 0.02 + 0.06 * pow_norm
    hr_base  = hr_floor + (pow_norm ** 2.5) * 0.18 * (0.6 + 0.4 * cs_norm)

    if contact_quality == "Weak":
        hr_prob   = 0.0
        hr_driver = 0.0
    elif contact_quality == "Hard":
        hr_prob   = hr_base
        hr_driver = round(pow_norm ** 2.5 * 100.0, 4)
    else:  # Medium
        hr_prob   = hr_base * 0.25
        hr_driver = round(pow_norm ** 2.5 * 100.0, 4)

    if rng.random() < hr_prob:
        probs = {"HR": 1.0, "Triple": 0.0, "Double": 0.0, "Single": 0.0, "Out": 0.0, "Error": 0.0}
        return {
            "bip_outcome":       "HR",
            "bip_probabilities": probs,
            "hr_driver":         hr_driver,
        }

    # ── Step 2: Non-HR multinomial ───────────────────────────────────────
    probs = dict(_NON_HR_BASE[contact_quality])

    # Pitcher suppression: bonus added to Out, removed proportionally from hit cols.
    # CMD suppresses Medium and Weak contact; STF suppresses Hard contact.
    if contact_quality in ("Medium", "Weak"):
        sup_bonus = max(0.0, (CMD - 50) / 50.0) * _PITCHER_SUP_SCALE
    else:  # Hard
        sup_bonus = max(0.0, (STF - 50) / 50.0) * _PITCHER_SUP_SCALE

    if sup_bonus > 0:
        hit_keys = ("Single", "Double", "Triple", "Error")
        hit_total = sum(probs[k] for k in hit_keys)
        actual_bonus = min(sup_bonus, hit_total * 0.90)  # never strip more than 90% of hits
        probs["Out"] += actual_bonus
        scale = (hit_total - actual_bonus) / hit_total if hit_total else 1.0
        for k in hit_keys:
            probs[k] *= scale

    # GAP shifts weight from Out toward Double (positive) or away (negative)
    gap_edge   = (GAP - 50) / 50.0          # –1 to +1
    double_adj = gap_edge * _GAP_DOUBLE_MOD
    probs["Double"] = max(0.0, probs["Double"] + double_adj)
    probs["Out"]    = max(0.0, probs["Out"]    - double_adj)

    # Center/Oppo spray nudges a tiny amount of Double into Triple
    if spray_vector in ("Center", "Oppo"):
        shift = min(_SPRAY_TRIPLE_BONUS, probs["Double"])
        probs["Triple"] += shift
        probs["Double"] -= shift

    # Global BABIP calibration: scale hit outcomes before final renormalization.
    # Renormalization then redistributes the remaining weight onto Out/Error,
    # keeping all probabilities valid and summing to exactly 1.0.
    if GLOBAL_BABIP_ADJUST != 1.0:
        for hit_key in ("Single", "Double", "Triple"):
            probs[hit_key] = probs.get(hit_key, 0.0) * GLOBAL_BABIP_ADJUST

    # Renormalize to exactly 1.0
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}

    # Sample
    roll       = rng.random()
    cumulative = 0.0
    bip_outcome = "Out"
    for label in ("Single", "Double", "Triple", "Error", "Out"):
        cumulative += probs[label]
        if roll < cumulative:
            bip_outcome = label
            break

    # Merge HR=0 into the returned probability dict for transparency
    probs["HR"] = round(hr_prob, 4)
    probs = {k: round(v, 4) for k, v in probs.items()}

    return {
        "bip_outcome":       bip_outcome,
        "bip_probabilities": probs,
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
        p_catch = _clamp(0.82 + rng_norm * 0.12 + cq_factor, 0.60, 0.97)
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
