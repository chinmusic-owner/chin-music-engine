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

# ── K% formula tuning ────────────────────────────────────────────────────────
#
# K% is computed as three additive terms on top of the league-average base:
#
#   k_pct = k_base
#         + K_STF_ABSOLUTE * stf_dominance    (pitcher's raw stuff — matchup-independent)
#         + K_STF_CON_COEFF * stf_con_edge    (relative STF−CON matchup, can be negative)
#         − ak_suppression                    (high AK: batter's bat-to-ball skill)
#         + ak_boost                          (low AK: batter's poor bat-to-ball skill)
#
# Term 1 — Absolute STF dominance:
#   stf_dominance = max(0, (STF − 50) / 100) ** 1.5
#   K_STF_ABSOLUTE = 0.25:
#     STF=50 → +0.00pp  |  STF=75 → +3.1pp  |  STF=90 → +6.4pp  |  STF=98 → +8.3pp
#
# Term 2 — Relative STF-CON edge:
#   K_STF_CON_COEFF = 0.20: a full 50-pt STF advantage adds 10pp; negative when CON > STF.
#
# Term 3 — AK (bat-to-ball skill), symmetric and nonlinear:
#   HIGH AK (above 50): suppresses K% — good bat-to-ball skill avoids strikeouts.
#     ak_suppression = max(0, (AK−50)/50) * (0.08 + batter_edge * 0.08)  [linear]
#   LOW AK (below 50): boosts K% — poor bat-to-ball skill gifts strikeouts to the pitcher.
#     ak_deficiency  = max(0, (50−AK)/50)
#     ak_boost       = ak_deficiency ** AK_LOW_EXP * AK_LOW_SCALE
#   The concave exponent (< 1) gives diminishing returns: the gap from AK=30 to AK=20
#   adds less boost than the gap from AK=50 to AK=30.  AK=50 → 0pp; AK=30 → +2.5pp;
#   AK=20 → +3.4pp.  This pulls Avg Pitcher vs low-AK hitters into the high-single-digit
#   K% range without letting extreme low AK produce implausible strikeout rates.
K_STF_ABSOLUTE  = 0.25
K_STF_CON_COEFF = 0.20
AK_LOW_EXP      = 0.70    # concave exponent — diminishing returns for each AK point below 50
AK_LOW_SCALE    = 0.047   # AK=30 → +2.5pp; AK=20 → +3.4pp; AK=0 → +4.7pp (capped by clamp)

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

    # K% — three-term formula on top of the league-average base.
    # See module-level constant block for full documentation.
    stf_con_edge  = (STF - CON) / 100.0
    stf_dominance = max(0.0, (STF - 50) / 100.0) ** 1.5   # 0 at STF≤50, ≈0.33 at STF=98

    # High AK: suppresses K% linearly (bat-to-ball skill avoids strikeouts).
    ak_hi          = max(0.0, (AK - 50) / 50.0)            # 0 at AK≤50, 1.0 at AK=100
    batter_edge    = max(0.0, -stf_con_edge)                # how much CON exceeds STF
    ak_suppression = ak_hi * (0.08 + batter_edge * 0.08)

    # Low AK: boosts K% with diminishing returns (poor bat-to-ball skill gives pitchers Ks).
    # Concave curve (exponent < 1) means AK=20 doesn't give double the boost of AK=30.
    ak_lo    = max(0.0, (50 - AK) / 50.0)                  # 0 at AK≥50, 1.0 at AK=0
    ak_boost = ak_lo ** AK_LOW_EXP * AK_LOW_SCALE

    k_pct = _clamp(
        k_base
        + K_STF_ABSOLUTE * stf_dominance
        + K_STF_CON_COEFF * stf_con_edge
        - ak_suppression
        + ak_boost,
        PROB_FLOOR,
        K_CEILING,
    )

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
# Calibrated so that (MID=28):
#   • Avg hitter vs avg pitcher   →  ~24% Hard  (intentional; was at floor with MID=44)
#   • Avg hitter vs elite pitcher →  ~10% Hard  (floor still reachable at low Q)
#   • Gehrig vs avg pitcher       →  ~59% Hard  (near ceiling for elite power vs avg)
#   • Gehrig vs Pedro 2000        →  ~50% Hard  (elite power making contact; fewer BIPs)

_HARD_FLOOR   = 0.10
_HARD_CEIL    = 0.60
_HARD_Q_MID   = 28.0    # Q_noisy where P(Hard) = midpoint (35%).
                         # Lowered from 44→28: avg-hitter Q_noisy≈25 was pinned at the
                         # floor; MID=28 places it near the inflection point (~24% Hard).
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

# ── Stage 2 contact-score saturation ─────────────────────────────────────────
#
# The raw batter contribution (a*eff_pow + b*CON + d*GAP) is linear and unbounded.
# A hitter like Gehrig (POW=99, CON=87, GAP=91) reaches batter_raw ≈ 90, which
# pushes Q_raw to 80+ and locks P(Hard) at the 0.60 ceiling vs ANY pitcher.
# The resulting BABIP against average pitching is .396+ — unrealistically high.
#
# Fix: apply a tanh soft cap so the batter contribution saturates at _BATTER_Q_CAP.
# The formula: batter_contribution = _BATTER_Q_CAP × tanh(batter_raw / _BATTER_Q_CAP)
#
#   batter_raw =  44 (avg hitter)   → contribution ≈ 37  (–7, modest effect)
#   batter_raw =  70 (good hitter)  → contribution ≈ 55  (–15, noticeable)
#   batter_raw =  91 (Gehrig)       → contribution ≈ 50  (–41, major cap)
#
# This prevents elite hitter × weak pitcher from always saturating Hard%,
# while keeping avg-vs-avg essentially unchanged.
_BATTER_Q_CAP = 55.0


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

    # Batter's raw contact contribution (linear, unbounded for high-trait players).
    batter_raw = a * effective_pow + b * CON + d * GAP

    # Soft cap via tanh saturation — see _BATTER_Q_CAP docstring above.
    # Prevents elite hitters from always pinning P(Hard) to the ceiling vs avg pitching.
    batter_contribution = _BATTER_Q_CAP * math.tanh(batter_raw / _BATTER_Q_CAP)

    # Q_raw: pitcher's STF subtracts from the capped batter contribution.
    Q_raw = batter_contribution - c * STF

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
#   Medium: Out 0.740  1B 0.185  2B 0.060  3B 0.008  ROE 0.007  (−0.04 Out, +0.04 Single vs prior 0.780/0.145)
#   Hard:   Out 0.680  1B 0.140  2B 0.180  3B 0.015  ROE 0.015  (unchanged)
_NON_HR_BASE: dict[str, dict[str, float]] = {
    "Weak":   {"Out": 0.830, "Single": 0.140, "Double": 0.020, "Triple": 0.005, "Error": 0.005},
    "Medium": {"Out": 0.740, "Single": 0.185, "Double": 0.060, "Triple": 0.008, "Error": 0.007},
    "Hard":   {"Out": 0.680, "Single": 0.140, "Double": 0.180, "Triple": 0.015, "Error": 0.015},
}

# Pitcher suppression scaling — bonus added to Out weight before renormalization.
# Formula: bonus = max(0, (trait - 50) / 50) * _PITCHER_SUP_SCALE
# At trait=70: bonus ≈ 0.06  |  At trait=90: bonus ≈ 0.12  |  capped at 0.15
_PITCHER_SUP_SCALE = 0.12   # CMD suppresses Medium/Weak Out; STF suppresses Hard Out

# GAP shifts probability from Out toward Double; capped so totals stay sane.
_GAP_DOUBLE_MOD  = 0.04   # max ±4% on Double at extreme GAP
# Spray bonus on Triple for Center/Oppo hits (gap power into the alleys).
_SPRAY_TRIPLE_BONUS = 0.003

# ── CMD-based hit suppressor (Stage 2.5) ─────────────────────────────────────
#
# GLOBAL_BABIP_ADJUST (0.82) is the CMD=50 baseline — it calibrates avg-vs-avg
# BABIP to ~.300.  CMD above 50 applies an additional multiplier that further
# reduces hit-in-play probability.  CMD below 50 uses the base (no inflation here;
# the CMD_VARIANCE mechanism already widens the distribution for wild pitchers).
#
# Formula:  effective_hit_adjust = GLOBAL_BABIP_ADJUST × cmd_hit_factor
#           cmd_hit_factor = 1.0 − max(0, CMD−50) / 100 × _CMD_HIT_SCALE
#
#   CMD = 50  → factor = 1.000  (neutral; effective = 0.82 × 1.00 = 0.820)
#   CMD = 70  → factor = 0.960  (effective = 0.82 × 0.96 = 0.787)
#   CMD = 93  → factor = 0.914  (effective = 0.82 × 0.91 = 0.749)  ← Pedro
#   CMD = 99  → factor = 0.902  (effective = 0.82 × 0.90 = 0.740)
#
# Pedro 2000 (CMD=93) gets ~8.6% more hit suppression than an average pitcher,
# on top of the Out-shift suppression already applied via _PITCHER_SUP_SCALE.
_CMD_HIT_SCALE = 0.10

# ── HR Gate tuning ───────────────────────────────────────────────────────────
#
# _HR_POW_SCALE: the main coefficient that converts (POW^2.5) into HR probability
# on Hard contact.  Reduced from 0.18 → 0.13 to bring raw HR/BIP into the
# realistic 6-12% range for elite power hitters vs average pitching.
_HR_POW_SCALE = 0.13
#
# Hard-contact HR floor bonus — additive constant applied ONLY to Hard-contact
# hr_prob, AFTER hr_base is computed.  Medium contact is NOT affected.
#
# Rationale: hr_base at avg-hitter traits (POW=50) ≈ 0.061.  The Hard contact
# path was producing only 10.1% Hard BIPs (pre-MID fix) and only 6% HR/Hard BIP,
# resulting in HR/PA ≈ .011.  After the MID=28 fix (step 1), Hard% rose to ~26%
# but HR/Hard BIP remained ~0.060 → HR/PA ≈ .016.  Adding 0.055 here lifts
# HR/Hard BIP to ~0.116 and targets avg HR/PA ≈ .028.
#
# For a POW=50 hitter: bonus raises Hard hr_prob by 90% (0.061→0.116).
# For a POW=99 hitter: bonus raises Hard hr_prob by 31% (0.175→0.230).
# The additive shape is intentional — it narrows the gap at the low-POW end
# without blowing up elite power hitters' conversion rates.
_HARD_HR_FLOOR_BONUS = 0.055
#
# CMD Mistake Factor — nonlinear suppression applied to hr_prob at high CMD:
#   cmd_mistake_factor = (1 - max(0, CMD - 50) / 100) ** CMD_HR_DAMPENER_EXP
#
# The quadratic exponent (2.0) makes the effect strongly nonlinear:
#   CMD = 50  → factor = 1.00  (no suppression — league-average command)
#   CMD = 70  → factor = 0.64  (above-average command)
#   CMD = 90  → factor = 0.36  (elite command — rare meatballs)
#   CMD = 93  → factor = 0.32  (Pedro 2000 — almost never grooves one)
#   CMD = 99  → factor = 0.26  (historically unprecedented command)
#
# Crucially: CMD 90 is NOT "a bit better than average." A pitcher with CMD 93
# should hold an elite power hitter's HR% per PA in the low-single-digit range,
# not double digits.
#
# Only applies above CMD=50.  Below CMD=50 the existing CMD_VARIANCE mechanism
# already handles wild pitchers (wider spread → more hard contact opportunities).
CMD_HR_DAMPENER_EXP = 0.6


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
    # POW^2.5 tail gives strong separation across the power spectrum.
    # Contact score boosts HR% on sharper contact (better-hit ball = more carry).
    # Medium contact can still produce HRs but at 25% the rate of Hard contact.
    pow_norm = POW / 100.0
    cs_norm  = contact_score / 100.0
    # HR floor scales with POW so low-power batters have a much lower floor.
    # _HR_POW_SCALE (0.13) reduced from original 0.18 to bring raw HR/BIP
    # into the realistic 6-12% range for elite power vs average pitching.
    hr_floor = 0.02 + 0.05 * pow_norm
    hr_base  = hr_floor + (pow_norm ** 2.5) * _HR_POW_SCALE * (0.6 + 0.4 * cs_norm)

    if contact_quality == "Weak":
        hr_prob   = 0.0
        hr_driver = 0.0
    elif contact_quality == "Hard":
        # Hard contact gets an additional floor bonus (_HARD_HR_FLOOR_BONUS)
        # on top of hr_base.  Medium is NOT modified — tuned independently.
        hr_prob   = hr_base + _HARD_HR_FLOOR_BONUS
        hr_driver = round(pow_norm ** 2.5 * 100.0, 4)
    else:  # Medium
        hr_prob   = hr_base * 0.25
        hr_driver = round(pow_norm ** 2.5 * 100.0, 4)

    # CMD Mistake Factor — nonlinear suppression of HR probability.
    # High CMD means the pitcher almost never "leaves one up" for the power hitter.
    # Applied only above CMD=50; below that the variance mechanism handles it.
    cmd_edge          = max(0.0, CMD - 50) / 100.0
    cmd_mistake_factor = (1.0 - cmd_edge) ** CMD_HR_DAMPENER_EXP
    hr_prob           *= cmd_mistake_factor

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

    # Combined hit suppressor: global BABIP calibration × CMD-based modifier.
    # GLOBAL_BABIP_ADJUST (0.82) is the CMD=50 baseline.
    # CMD above 50 compounds an additional reduction; CMD below 50 is neutral here
    # (wild pitchers are handled by CMD_VARIANCE, not by inflating hit rates).
    cmd_hit_factor    = 1.0 - max(0.0, CMD - 50) / 100.0 * _CMD_HIT_SCALE
    effective_hit_adj = GLOBAL_BABIP_ADJUST * cmd_hit_factor
    for hit_key in ("Single", "Double", "Triple"):
        probs[hit_key] = probs.get(hit_key, 0.0) * effective_hit_adj

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
