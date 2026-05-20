"""
game_engine.py — Inning Engine (PRD 04)

State-machine game simulation wrapped around the frozen PA Engine.

Owns:
    - Inning / half-inning progression
    - Base state and runner advancement
    - Outs counter and half-inning transitions
    - Batting order cycling (1–9, wraps across innings)
    - Run scoring and walk-off detection
    - Box score construction

All plate appearances are resolved via pa_wrapper.resolve_pa_seeded.
pa_engine.py is never imported or modified here.

Usage:
    from game_engine import simulate_game

    result = simulate_game(away_team, home_team)
    print(result.final_score)         # {"away": 3, "home": 5}
    print(result.linescore["away"])   # [0, 0, 1, 0, 2, 0, 0, 0, 0]
    for event in result.pa_events:
        print(event.inning, event.outcome, event.runs_scored)

Team format (preferred — explicit dict):
    {
        "team_id": "NYA",                  # optional display label
        "lineup":  [<9 batter cards>],     # player dicts with "traits", "bats", etc.
        "pitcher": <pitcher card>,         # player dict with "traits", "throws", etc.
    }

Team format (raw list of player cards):
    The first card with primary_role == "Pitcher" becomes the starter;
    the first 9 cards with primary_role == "Hitter" form the lineup.

Player card shape follows SIM_TECHNICAL_DOC.md (same as pa_wrapper expectations):
    batter  — {"player_id": ..., "name": ..., "bats": ..., "traits": {"CON", "EYE", "AK", "POW", "GAP"}}
    pitcher — {"player_id": ..., "name": ..., "throws": ..., "traits": {"STF", "CTL", "CMD"}}
"""

from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass, field

from pa_wrapper import resolve_pa_seeded
from narrative_dictionary import narrate, narrate_grounder


# ── Seed derivation ─────────────────────────────────────────────────────────────

def _derive_seed(game_id: str, pa_index: int) -> int:
    """Deterministic PA seed: sha256(game_id:pa_index) mod 2^32."""
    raw = hashlib.sha256(f"{game_id}:{pa_index}".encode()).hexdigest()
    return int(raw, 16) % (2 ** 32)


# ── Groundball resolution tables ────────────────────────────────────────────────
#
# These drive the two-stage "Out" outcome:
#   Stage 1 — Infield hit check (BEFORE counting the out)
#   Stage 2 — Groundout + optional double play (if Stage 1 miss)
#
# All rates are conditional on outcome == "Out" (post-PA-engine resolution).
# SPD modifier is applied on top of the base rate (centered at SPD 50).

# Infield hit base probability by contact quality
# Applied to all BIP outs (including fly balls), so rates are calibrated to
# the full out pool rather than groundball-only.  Weak rate reduced from
# 0.15 → 0.09: the 0.15 baseline was adding ~11–12% phantom singles on
# top of already-inflated post-defense Weak-contact outs, pushing blended
# BABIP toward .330+.  Medium reduced from 0.045 → 0.035 for same reason.
_INFIELD_HIT_BASE: dict[str, float] = {
    "Hard":   0.010,  # rocket right at a fielder — almost never an infield hit
    "Medium": 0.035,  # routine grounder — occasional leg hit
    "Weak":   0.090,  # slow roller / weak chopper — beaten out, but not every time
}
# Double play base probability by contact quality (runner on 1B, <2 outs)
_DP_BASE: dict[str, float] = {
    "Hard":   0.70,   # two-hopper straight to SS/2B — very easy to turn two
    "Medium": 0.475,  # routine grounder — coin-flip DP
    "Weak":   0.125,  # slow roller — fielder must charge; only one play possible
}
# SPD mod per point from 50: ±0.00225 → gives ±0.045 at SPD 30 / SPD 70
_SPD_IH_MOD:  float = 0.00225   # fast batters more likely to beat it out
_SPD_DP_MOD:  float = 0.003     # slow runners more likely to be doubled up


# ── Out-zone tables for FIX 4 (fielder naming) ──────────────────────────────────
#
# Each entry is (zone_code, cumulative_probability).  Zones are assigned
# probabilistically from contact_quality using a deterministic RNG seeded
# off the PA seed.  The zone drives both the narrative template and the
# defensive position lookup in print_game_log.
#
# Zone codes:
#   gb_p  — comebacker to pitcher
#   gb_1b, gb_2b, gb_ss, gb_3b — grounder to that infield position
#   fb_lf, fb_cf, fb_rf — fly ball to that outfield position
#   pu_if — infield popup (position rotated round-robin)
#   ld_if, ld_lf, ld_cf, ld_rf — line drive caught at that zone
_OUT_ZONES_BY_CQ: dict[str, list[tuple[str, float]]] = {
    "Weak": [
        ("gb_p",  0.18),
        ("gb_1b", 0.33),
        ("gb_3b", 0.50),
        ("gb_ss", 0.63),
        ("gb_2b", 0.75),
        ("pu_if", 1.00),
    ],
    "Medium": [
        ("gb_ss", 0.13),
        ("gb_2b", 0.21),
        ("gb_3b", 0.28),
        ("gb_1b", 0.35),
        ("fb_lf", 0.49),
        ("fb_cf", 0.66),
        ("fb_rf", 0.77),
        ("pu_if", 0.84),
        ("ld_if", 1.00),
    ],
    "Hard": [
        ("gb_ss", 0.05),
        ("gb_3b", 0.10),
        ("ld_if", 0.18),
        ("fb_lf", 0.33),
        ("fb_cf", 0.56),
        ("fb_rf", 0.74),
        ("ld_lf", 0.82),
        ("ld_cf", 0.91),
        ("ld_rf", 1.00),
    ],
}

# Zone → (defensive position code, narrative template key, display position label)
_ZONE_META: dict[str, tuple[str, str, str]] = {
    "gb_p":  ("P",  "OUT_COMEBACKER", "the mound"),
    "gb_1b": ("1B", "OUT_GROUNDER",   "first base"),
    "gb_2b": ("2B", "OUT_GROUNDER",   "second base"),
    "gb_ss": ("SS", "OUT_GROUNDER",   "short"),
    "gb_3b": ("3B", "OUT_GROUNDER",   "third base"),
    "fb_lf": ("LF", "OUT_FLY",        "left field"),
    "fb_cf": ("CF", "OUT_FLY",        "center field"),
    "fb_rf": ("RF", "OUT_FLY",        "right field"),
    "pu_if": ("IF", "OUT_POPUP",      "the infield"),
    "ld_if": ("IF", "OUT_LINER",      "the infield"),
    "ld_lf": ("LF", "OUT_LINER",      "left field"),
    "ld_cf": ("CF", "OUT_LINER",      "center field"),
    "ld_rf": ("RF", "OUT_LINER",      "right field"),
}

# (Slot-based defensive position proxy removed — real positions from field_pos DB column.)


def _assign_out_zone(contact_quality: str, pa_seed: int) -> str:
    """Return a zone code for a batted-ball out, driven by contact quality."""
    rng   = random.Random(pa_seed ^ 0xF00D)
    roll  = rng.random()
    zones = _OUT_ZONES_BY_CQ.get(contact_quality, _OUT_ZONES_BY_CQ["Medium"])
    for zone, cum_prob in zones:
        if roll <= cum_prob:
            return zone
    return zones[-1][0]


# ── Fatigue model constants (PRD 06) ────────────────────────────────────────────
#
# fatigue_level = batters_faced / (STA * 0.35 + 15),  clamped [0.0, 1.0]
#
# Stages:  < 0.40  → "fresh"
#          0.40–0.70 → "working"
#          0.70–0.90 → "tired"
#          > 0.90  → "gassed"
#
# Each trait begins decaying at its own threshold and cannot fall below 20% of
# its original value (_MAX_TRAIT_DECAY = 0.80 → at most 80% loss).
# Decay is non-linear (exponent 1.5) so the curve accelerates late.
#
#   STF  starts at fatigue_level > 0.30   (stuff breaks first)
#   CTL  starts at fatigue_level > 0.50   (command erodes second)
#   CMD  starts at fatigue_level > 0.70   (location fails last)
#
_STF_DECAY_START: float = 0.30
_CTL_DECAY_START: float = 0.50
_CMD_DECAY_START: float = 0.70
_MAX_TRAIT_DECAY: float = 0.80   # → trait floor = 20% of original


def _trait_decay_factor(fatigue_level: float, start_threshold: float) -> float:
    """
    Return the decay multiplier [0.0, 0.80] for a trait that begins degrading
    once fatigue_level exceeds *start_threshold*.

    The curve uses a 1.5-power function so decay accelerates as fatigue
    approaches 1.0 (natural collapse under heavy workload).

      excess = (fatigue_level - start_threshold) / (1.0 - start_threshold)
      decay  = excess^1.5 * 0.80,  clamped to [0.0, 0.80]

    At fatigue_level == start_threshold  → decay = 0.00  (trait at 100%)
    At fatigue_level == 1.0             → decay = 0.80  (trait at 20%)
    """
    if fatigue_level <= start_threshold:
        return 0.0
    span   = 1.0 - start_threshold
    excess = (fatigue_level - start_threshold) / span
    return min(_MAX_TRAIT_DECAY, (excess ** 1.5) * _MAX_TRAIT_DECAY)


# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class PAEvent:
    """Immutable receipt for one plate appearance — the 'receipt' log."""
    inning: int
    half: str               # "top" (away bats) or "bottom" (home bats)
    pa_number: int          # 1-indexed global PA counter across the whole game
    batter_id: str
    batter_name: str
    pitcher_id: str
    pitcher_name: str
    outcome: str            # The resolved outcome string from pa_wrapper
    outs_before: int        # Outs on the board when this PA began (0, 1, or 2)
    bases_before: list      # [1B_occ, 2B_occ, 3B_occ] bool snapshot before this PA
    runs_scored: int        # Runs that crossed the plate on this PA
    seed: int               # Seed used for this PA (for full determinism replay)
    raw_result: dict        # Full dict returned by resolve_pa_seeded (stage metadata)
    base_runners_before: list = field(default_factory=list)  # runner cards before this PA
    outs_recorded:    int  = 0   # Outs added this PA (0-2; DPs correctly contribute 2)
    fielder_zone:     str  = ""  # Zone code for batted-ball outs (drives fielder narrative)
    baserunning_note: str = ""   # Narrative for any extra-base attempt on this PA
    fatigue_note: str     = ""   # Fatigue stage transition or pitching change narrative


@dataclass
class BoxScore:
    """Complete game record returned by simulate_game."""
    away_team_id: str
    home_team_id: str
    final_score: dict       # {"away": int, "home": int}
    linescore: dict         # {"away": [int, ...], "home": [int, ...]} — one entry per inning
    pa_events: list         # list[PAEvent]
    innings_played: int
    walk_off: bool    = False
    pitcher_log: list = field(default_factory=list)  # per-half-inning PitcherState snapshots
    away_lineup:    list = field(default_factory=list)  # away batting lineup (player cards)
    home_lineup:    list = field(default_factory=list)  # home batting lineup (player cards)
    away_def_align: dict = field(default_factory=dict)  # away defensive alignment {pos: name}
    home_def_align: dict = field(default_factory=dict)  # home defensive alignment {pos: name}


@dataclass
class GameState:
    """
    Mutable state machine — the single source of truth during simulation.

    bases[0] = 1B occupied (bool)
    bases[1] = 2B occupied (bool)
    bases[2] = 3B occupied (bool)

    batter_idx tracks where each team is in their 1–9 order (0-indexed, wraps).
    """
    inning: int = 1
    half: str = "top"       # "top" (away bats) or "bottom" (home bats)
    outs: int = 0
    score: dict = field(default_factory=lambda: {"away": 0, "home": 0})
    bases: list = field(default_factory=lambda: [False, False, False])
    base_runners: list = field(default_factory=lambda: [None, None, None])
    batter_idx: dict = field(default_factory=lambda: {"away": 0, "home": 0})


@dataclass
class PitcherState:
    """
    Tracks the active pitcher's fatigue state for one team's staff.  (PRD 06)

    Fatigue model
    ─────────────
    fatigue_level = batters_faced / (STA × 0.35 + 15),  clamped [0.0, 1.0]

    Stages:
        < 0.40  → "fresh"     — no meaningful decay
        0.40–0.70 → "working" — STF begins to decay
        0.70–0.90 → "tired"   — STF + CTL decaying; CMD just starting
        > 0.90  → "gassed"    — all three traits under heavy decay

    Trait decay order (sequential, non-linear):
        STF  starts at fatigue_level > 0.30   (stuff breaks first)
        CTL  starts at fatigue_level > 0.50   (command erodes second)
        CMD  starts at fatigue_level > 0.70   (location fails last)

    Floor: no trait can fall below 20% of its original starting value.

    Pull triggers (any one is sufficient):
        1. fatigue_stage == "gassed"
        2. batters_faced > STA × 0.35 + 10   (workload hard cap)
        3. fatigue_stage == "tired" AND runs_allowed_this_inning >= 3

    Bullpen cycling
    ───────────────
    pull_next() pops bullpen[0], resets all per-pitcher counters, and
    snapshots the new pitcher's original traits.
    If the bullpen is empty the fatigued pitcher stays in.
    """
    current:                 dict
    original_traits:         dict = field(default_factory=dict)
    batters_faced:           int  = 0
    runs_allowed:            int  = 0
    runs_allowed_this_inning: int = 0
    bullpen:                 list = field(default_factory=list)

    def __post_init__(self) -> None:
        # Snapshot the pitcher's traits at the moment they take the mound.
        # This reference value never changes for this pitcher's outing —
        # effective_card() computes decay against it every PA.
        if not self.original_traits:
            self.original_traits = dict(self.current.get("traits", {}))

    # ── Fatigue level & stage ────────────────────────────────────────────────

    @property
    def fatigue_level(self) -> float:
        """Workload ratio [0.0, 1.0].  1.0 = pitcher is fully spent."""
        sta   = self.original_traits.get("STA", 60)
        limit = sta * 0.35 + 15
        return min(1.0, self.batters_faced / limit)

    @property
    def fatigue_stage(self) -> str:
        """Human-readable fatigue stage.  Drives narrative and pull logic."""
        fl = self.fatigue_level
        if fl < 0.40:
            return "fresh"
        if fl < 0.70:
            return "working"
        if fl < 0.90:
            return "tired"
        return "gassed"

    # ── Trait decay ──────────────────────────────────────────────────────────

    def effective_card(self) -> dict:
        """
        Return the pitcher card with fatigue-decayed STF, CTL, CMD.

        Original traits are never mutated; decay is computed fresh each call.
        Floors are enforced at 20% of each trait's original value (≥ 1).
        """
        fl   = self.fatigue_level
        orig = self.original_traits

        def _decayed(original: float, threshold: float) -> int:
            factor = _trait_decay_factor(fl, threshold)
            floor  = max(1, round(original * (1.0 - _MAX_TRAIT_DECAY)))
            return max(floor, round(original * (1.0 - factor)))

        return {
            **self.current,
            "traits": {
                **self.current.get("traits", {}),
                "STF": _decayed(orig.get("STF", 50), _STF_DECAY_START),
                "CTL": _decayed(orig.get("CTL", 50), _CTL_DECAY_START),
                "CMD": _decayed(orig.get("CMD", 50), _CMD_DECAY_START),
            },
        }

    # ── Pull logic ───────────────────────────────────────────────────────────

    def should_pull(self) -> bool:
        """
        Return True if the pitcher should be replaced before the next batter.

        Three independent triggers (any one is sufficient):
          1. Stage is "gassed"  (fatigue_level > 0.90)
          2. Workload exceeds hard cap  (batters_faced > STA × 0.35 + 10)
          3. Stage is "tired" AND 3+ runs allowed this inning
        """
        stage = self.fatigue_stage
        if stage == "gassed":
            return True
        sta = self.original_traits.get("STA", 60)
        if self.batters_faced > (sta * 0.35 + 10):
            return True
        if stage == "tired" and self.runs_allowed_this_inning >= 3:
            return True
        return False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def pull_next(self) -> bool:
        """
        Swap in the next bullpen arm (top-down, deterministic).
        Resets all per-pitcher counters and snapshots the new pitcher's
        original traits.  Returns True if a change was made.
        """
        if not self.bullpen:
            return False
        self.current                  = self.bullpen.pop(0)
        self.original_traits          = dict(self.current.get("traits", {}))
        self.batters_faced            = 0
        self.runs_allowed             = 0
        self.runs_allowed_this_inning = 0
        return True

    def reset_inning(self) -> None:
        """Call at the start of every half-inning to reset inning-level run counter."""
        self.runs_allowed_this_inning = 0

    def record_pa(self, runs: int) -> None:
        """Increment workload counters after one PA has resolved."""
        self.batters_faced            += 1
        self.runs_allowed             += runs
        self.runs_allowed_this_inning += runs

    # ── Audit snapshot ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Return the per-inning pitcher_state dict required by PRD 06:
          { id, name, curr_stf, curr_ctl, curr_cmd, fatigue_level,
            fatigue_stage, is_fatigued, batters_faced }
        """
        card = self.effective_card()
        t    = card.get("traits", {})
        return {
            "id":            self.current.get("player_id", "unknown"),
            "name":          self.current.get("name", "unknown"),
            "curr_stf":      t.get("STF", 0),
            "curr_ctl":      t.get("CTL", 0),
            "curr_cmd":      t.get("CMD", 0),
            "fatigue_level": round(self.fatigue_level, 3),
            "fatigue_stage": self.fatigue_stage,
            "is_fatigued":   self.fatigue_stage in ("tired", "gassed"),
            "batters_faced": self.batters_faced,
        }


# ── Team parsing ────────────────────────────────────────────────────────────────

def _parse_team(team, default_id: str) -> tuple[str, list, dict, list]:
    """
    Normalize a team argument into (team_id, lineup, pitcher, bullpen).

    Accepts:
        dict  — must have "lineup" (list) and "pitcher" (dict); "team_id" optional.
                "bullpen" (list of pitcher dicts) is optional.
        list  — raw player cards; auto-split on primary_role.
    """
    if isinstance(team, dict) and "lineup" in team:
        team_id = team.get("team_id", default_id)
        lineup  = team["lineup"]
        pitcher = team["pitcher"]
        bullpen = list(team.get("bullpen") or [])
    else:
        cards    = list(team)
        team_id  = cards[0].get("team_id", default_id) if cards else default_id
        pitchers = [c for c in cards if c.get("primary_role") == "Pitcher"]
        hitters  = [c for c in cards if c.get("primary_role") == "Hitter"]
        if not pitchers:
            raise ValueError(
                f"Team '{team_id}' has no pitcher card (primary_role='Pitcher')."
            )
        if len(hitters) < 9:
            raise ValueError(
                f"Team '{team_id}' has only {len(hitters)} hitters; need at least 9."
            )
        pitcher = pitchers[0]
        lineup  = hitters[:9]
        bullpen = pitchers[1:]

    return team_id, lineup, pitcher, bullpen


# ── Baserunning resolution ──────────────────────────────────────────────────────

def resolve_baserunning(
    runner:      dict,
    fielder_arm: int,
    outcome:     str,
    seed:        int,
) -> tuple[str, str]:
    """
    Decide whether a baserunner attempts and completes an extra-base advance.

    Called for:
        Single  — 1B runner optionally tries for 3B  (vs stopping at 2B)
        Double  — 1B runner optionally tries to score (vs stopping at 3B)

    Probability model
    ─────────────────
    speed_edge  = (SPD − fielder_arm) / 100        # ±40 typical range
    p_safe      = clamp(0.50 + speed_edge, 0.10, 0.90)   # % to make it if attempted

    BIQ determines the attempt threshold:
        high BIQ (smart) → only goes when p_safe > ~0.57   (won't risk bad odds)
        mid  BIQ         → goes when p_safe > ~0.45
        low  BIQ (reckless) → goes even when p_safe is only ~0.33

    If p_safe < attempt_thresh → "conservative" (runner holds; note only on close call)
    Otherwise: flip at p_safe probability → "advance_safely" or "thrown_out"

    Returns:
        (result_type, narrative_note)
        result_type ∈ {"conservative", "advance_safely", "thrown_out"}
    """
    rng    = random.Random(seed)
    traits = runner.get("traits", {})
    spd    = traits.get("SPD", 50)
    biq    = traits.get("BIQ", 50)
    name   = runner.get("name", "The runner")

    speed_edge     = (spd - fielder_arm) / 100
    p_safe         = max(0.10, min(0.90, 0.50 + speed_edge))
    attempt_thresh = 0.25 + biq * 0.004   # BIQ 20→0.33, 50→0.45, 80→0.57

    if p_safe < attempt_thresh:
        # Runner (or 3B coach) correctly reads the play and holds.
        # Only annotate when it was genuinely close — avoids wall-to-wall notes.
        note = narrate("SMART_HOLD", name) if p_safe >= attempt_thresh - 0.05 else ""
        return "conservative", note

    # Runner is going for it.
    if rng.random() < p_safe:
        return "advance_safely", narrate("ADVANCE_EXTRA", name)
    else:
        # Destination-specific thrown-out narrative:
        #   Double → runner was trying to score   → "out at home"
        #   Single → runner was trying for third  → "cut down at third"
        # Low-BIQ runners get RECKLESS_ADVANCE regardless of destination.
        if biq < 45:
            key = "RECKLESS_ADVANCE"
        elif outcome == "Double":
            key = "THROWN_OUT_AT_HOME"
        else:
            key = "THROWN_OUT_AT_THIRD"
        return "thrown_out", narrate(key, name)


# ── Base advancement ────────────────────────────────────────────────────────────

def _apply_outcome(
    state:           GameState,
    outcome:         str,
    batter:          dict,
    fielder_arm:     int,
    br_seed:         int,
    contact_quality: str = "Medium",
) -> tuple[int, str, str]:
    """
    Apply one PA outcome to the game state.

    Mutates state.bases, state.base_runners, state.outs, and state.score in-place.
    Returns (runs_scored, baserunning_note, resolved_outcome).

    resolved_outcome is either "" (use the PA-engine outcome unchanged) or
    "InfieldHit" when a groundball "Out" is converted to a safe play.

    Groundball two-stage logic (when outcome == "Out")
    ──────────────────────────────────────────────────
    Stage 1 — Infield hit check (before counting the out):
        Probability depends on contact_quality + batter SPD.
        If fired: batter reaches 1B, runners advance one base, NO out recorded.
    Stage 2 — Groundout / double play (if Stage 1 missed):
        Out counted at 1B.  If runner on 1B and <3 outs, DP check fires using
        contact_quality + runner SPD.  Weak contact triggers WEAK_ROLLER_NO_DP
        note when DP doesn't occur.

    Baserunning decisions on hits
    ──────────────────────────────
    On a Single or Double, the runner who was on 1B faces an optional extra-base
    attempt.  resolve_baserunning() uses their SPD/BIQ vs the outfield ARM:
        Single  — runner on 1B tries for 3B  vs safe 2B
        Double  — runner on 1B tries to score vs safe 3B

    Advancement rules (deterministic):
        Single / Error : 3B scores, 2B scores, 1B→optional-extra, batter→1B
        Double         : 3B scores, 2B scores, 1B→optional-extra, batter→2B
        Triple         : All runners score, batter→3B
        HR             : Batter + all runners score, bases clear
        BB / HBP       : Force advancement only
        K              : Out, no base changes
    """
    b    = state.bases           # mutable [1B, 2B, 3B] occupancy flags
    br   = state.base_runners    # mutable [1B, 2B, 3B] player card | None
    side = "away" if state.half == "top" else "home"
    runs    = 0
    br_note = ""

    # ── K : simple out, no base changes ─────────────────────────────────────
    if outcome == "K":
        state.outs += 1
        return 0, "", ""

    # ── Batted out : two-stage groundball resolution ─────────────────────────
    if outcome == "Out":
        batter_spd = batter.get("traits", {}).get("SPD", 50)
        batter_name = batter.get("name", "")
        rng_ih = random.Random(br_seed ^ 0x1F3)   # infield-hit roll
        rng_dp = random.Random(br_seed ^ 0x4321)  # double-play roll (independent)

        # ── Stage 1: Infield hit check ────────────────────────────────────────
        ih_base   = _INFIELD_HIT_BASE.get(contact_quality, 0.045)
        ih_chance = max(0.0, min(0.30, ih_base + (batter_spd - 50) * _SPD_IH_MOD))

        if rng_ih.random() < ih_chance:
            # Batter safe — conservative single-like advancement (one base each)
            if b[2]:  runs += 1;  b[2] = False; br[2] = None          # 3B scores
            if b[1]:  b[2] = True; br[2] = br[1]; b[1] = False; br[1] = None  # 2B→3B
            if b[0]:  b[1] = True; br[1] = br[0]; b[0] = False; br[0] = None  # 1B→2B
            b[0] = True; br[0] = batter                                # batter→1B
            state.score[side] += runs
            return runs, "", "InfieldHit"   # br_note="" — primary template handles it

        # ── Stage 2: Groundout / Fielder's Choice / Double Play ───────────────
        # outs increment happens inside each branch so the right player is credited.
        if b[0] and state.outs < 3:
            # Runner on 1B is in play — evaluate DP, FC, or weak-roller groundout.
            runner     = br[0]
            runner_spd = runner.get("traits", {}).get("SPD", 50) if runner else 50
            dp_base    = _DP_BASE.get(contact_quality, 0.475)
            dp_chance  = max(0.05, min(0.80, dp_base + (50 - runner_spd) * _SPD_DP_MOD))

            if state.outs < 2 and rng_dp.random() < dp_chance:
                # 6-4-3 DP: batter out at 1B + r1 out at 2B = 2 outs
                state.outs += 2

                # Snapshot 2B/3B before mutating.
                r2 = br[1] if b[1] else None   # runner on 2B → advances to 3B
                r3 = br[2] if b[2] else None   # runner on 3B → stays, or scores

                # Clear all bases; rebuild from surviving runners only.
                b[0] = False;  br[0] = None
                b[1] = False;  br[1] = None
                b[2] = False;  br[2] = None

                if r3 and r2:
                    runs += 1;  b[2] = True;  br[2] = r2   # r3 scores; r2 takes 3B
                elif r3:
                    b[2] = True;  br[2] = r3               # r3 stays
                elif r2:
                    b[2] = True;  br[2] = r2               # r2 advances to empty 3B

                state.score[side] += runs
                br_note = narrate("DOUBLE_PLAY", batter_name)

            elif contact_quality != "Weak":
                # True Fielder's Choice: fielder fires to 2B for the force; batter
                # safely reaches 1B.  One out (the runner) is recorded, not the batter.
                state.outs += 1          # runner on 1B retired at 2B
                br[0] = batter           # batter safely takes the open spot at 1B
                # b[0] stays True
                return runs, "", "FC"   # br_note="" — primary FC template covers it

            else:
                # Weak roller: batter out at 1B; runner on 1B holds (no time for 2B)
                state.outs += 1
                br_note = narrate("WEAK_ROLLER_NO_DP", batter_name)

        else:
            # No runner on 1B: simple groundout — batter out at 1B
            state.outs += 1
        return runs, br_note, ""

    # ── Singles and errors (batter reaches 1B) ──────────────────────────────
    if outcome in ("Single", "Error"):
        # Deterministic: 3B scores, 2B scores
        if b[2]:  runs += 1;  b[2] = False;  br[2] = None
        if b[1]:  runs += 1;  b[1] = False;  br[1] = None

        # Optional: 1B runner tries for 3rd vs stops at 2nd
        if b[0]:
            runner     = br[0]
            b[0]       = False;  br[0] = None
            result, br_note = (
                resolve_baserunning(runner, fielder_arm, outcome, br_seed)
                if runner else ("conservative", "")
            )
            if result == "thrown_out":
                state.outs += 1
                if state.outs >= 3:
                    # Inning over mid-play: credit confirmed runs, skip batter placement
                    state.score[side] += runs
                    return runs, br_note, ""
            elif result == "advance_safely":
                b[2] = True;  br[2] = runner
            else:                               # conservative
                b[1] = True;  br[1] = runner

        b[0] = True;  br[0] = batter            # batter reaches 1B

    # ── Double ──────────────────────────────────────────────────────────────
    elif outcome == "Double":
        # Deterministic: 3B scores, 2B scores
        if b[2]:  runs += 1;  b[2] = False;  br[2] = None
        if b[1]:  runs += 1;  b[1] = False;  br[1] = None

        # Optional: 1B runner tries to score vs stops at 3rd
        if b[0]:
            runner     = br[0]
            b[0]       = False;  br[0] = None
            result, br_note = (
                resolve_baserunning(runner, fielder_arm, outcome, br_seed)
                if runner else ("conservative", "")
            )
            if result == "thrown_out":
                state.outs += 1
                if state.outs >= 3:
                    # Runner thrown out at plate is the third out — NO run scored
                    state.score[side] += runs
                    return runs, br_note, ""
            elif result == "advance_safely":
                runs += 1                       # runner crosses the plate
            else:                               # conservative
                b[2] = True;  br[2] = runner

        b[1] = True;  br[1] = batter            # batter reaches 2B

    # ── Triple ──────────────────────────────────────────────────────────────
    elif outcome == "Triple":
        runs = int(b[0]) + int(b[1]) + int(b[2])
        b[0] = b[1] = b[2] = False
        br[0] = br[1] = br[2] = None
        b[2] = True;  br[2] = batter            # batter to 3B

    # ── Home run ────────────────────────────────────────────────────────────
    elif outcome == "HR":
        runs = int(b[0]) + int(b[1]) + int(b[2]) + 1
        b[0] = b[1] = b[2] = False
        br[0] = br[1] = br[2] = None

    # ── Walk / HBP (force advancement only) ─────────────────────────────────
    elif outcome in ("BB", "HBP"):
        b1, b2, b3 = b[0], b[1], b[2]
        r1, r2, r3 = br[0], br[1], br[2]
        runs   = 1 if (b1 and b2 and b3) else 0
        # 3B runner stays unless forced home (bases loaded);
        # 2B runner advances to 3B only when 1B is also occupied (force chain).
        br[2]  = r3 if (b3 and not (b1 and b2)) else (r2 if (b1 and b2) else r3)
        b[2]   = b3 or (b1 and b2)
        br[1]  = r1 if b1 else r2
        b[1]   = b1 or b2
        br[0]  = batter
        b[0]   = True

    state.score[side] += runs
    return runs, br_note, ""


# ── Half-inning transition ──────────────────────────────────────────────────────

def _end_half_inning(state: GameState) -> None:
    """Clear bases, reset outs, and advance to the next half (or inning)."""
    state.bases        = [False, False, False]
    state.base_runners = [None, None, None]
    state.outs         = 0
    if state.half == "top":
        state.half = "bottom"
    else:
        state.half = "top"
        state.inning += 1


# ── Half-inning simulation ──────────────────────────────────────────────────────

def _play_half_inning(
    state:        GameState,
    batting_side: str,
    lineup:       list,
    ps:           PitcherState,   # mutable — persists across innings for one team
    pa_events:    list,
    pa_counter:   list,           # single-element list [int] — mutable global PA count
    game_id:      str,
    fielder_arm:  int = 55,       # outfield ARM for the defensive team
    check_walkoff: bool = False,
) -> tuple[int, bool]:
    """
    Simulate one half-inning until 3 outs or a walk-off.

    ps (PitcherState) is shared across all half-innings for this pitching side so
    fatigue accumulates correctly across the whole game.

    fielder_arm: outfield ARM strength for the team currently in the field.
    Used by resolve_baserunning() to decide extra-base attempt outcomes.

    Returns:
        (runs_scored_this_half, walk_off_occurred)
    """
    half_runs    = 0
    walk_off     = False
    pending_note = ""   # carries PITCHING_CHANGE narrative to the very next PAEvent

    # Reset inning-level run counter so should_pull()'s tired+3-run trigger
    # only fires on runs allowed in the current half-inning.
    ps.reset_inning()

    while state.outs < 3:

        # ── Manager hook: pull tired pitcher before this batter ──────────────
        if ps.should_pull():
            old_name = ps.current.get("name", "unknown")
            changed  = ps.pull_next()          # swap in next bullpen arm; resets counters
            if changed:
                new_name     = ps.current.get("name", "unknown")
                pending_note = narrate("PITCHING_CHANGE", new_name)
                # Reset inning counter for the new pitcher's accountability window
                ps.reset_inning()

        # Snapshot fatigue stage BEFORE this PA so we can detect transitions
        stage_before = ps.fatigue_stage

        # Snapshot pre-PA state for the receipt
        bases_snap   = list(state.bases)
        runners_snap = list(state.base_runners)   # runner cards (or None) at each base
        outs_before  = state.outs

        # Pull the next batter in the 1–9 order (wraps across inning boundaries)
        idx    = state.batter_idx[batting_side]
        batter = lineup[idx]
        state.batter_idx[batting_side] = (idx + 1) % len(lineup)

        # Generate a unique, deterministic seed for this PA
        pa_counter[0] += 1
        seed    = _derive_seed(game_id, pa_counter[0])
        br_seed = seed ^ 0xBA5E0000   # distinct but deterministic baserunning seed

        # ── THE PA ENGINE CALL (with fatigue-adjusted pitcher card) ──────────
        raw_result      = resolve_pa_seeded(batter, ps.effective_card(), seed=seed)
        outcome         = raw_result["outcome"]
        contact_quality = (
            (raw_result or {}).get("contact", {}) or {}
        ).get("contact_quality", "Medium")

        # Apply outcome; baserunning decisions included for Single/Double.
        # resolved_outcome is "" unless a groundball "Out" was converted to
        # an infield hit, in which case it returns "InfieldHit".
        runs, br_note, resolved_outcome = _apply_outcome(
            state, outcome, batter, fielder_arm, br_seed, contact_quality
        )
        half_runs += runs

        # Outs added by this PA (0 for hits/walks, 1 for routine outs,
        # 2 for double plays, possibly 2 for baserunner throw-outs on same play).
        outs_recorded = state.outs - outs_before

        # Assign fielder zone for true batted-ball outs (not DP, FC, or IH).
        # DP uses DOUBLE_PLAY_PRIMARY; FC uses FIELDERS_CHOICE; IH uses INFIELD_HIT.
        # K has no zone. The zone drives fielder-named narratives in print_game_log.
        if (outcome == "Out"
                and resolved_outcome not in ("InfieldHit", "FC")
                and outs_recorded == 1):
            fielder_zone = _assign_out_zone(contact_quality, br_seed)
        else:
            fielder_zone = ""

        # Update pitcher's fatigue counters AFTER the PA resolves
        ps.record_pa(runs)

        # ── Fatigue narrative ─────────────────────────────────────────────────
        pitcher_name = ps.current.get("name", "unknown")
        stage_after  = ps.fatigue_stage
        fatigue_note = pending_note          # consume any queued pitching-change line
        pending_note = ""

        if stage_after != stage_before:
            # Stage transition — tag it once (prioritise pitching-change over transition)
            if stage_after == "tired" and not fatigue_note:
                fatigue_note = narrate("PITCHER_TIRED", pitcher_name)
            elif stage_after == "gassed" and not fatigue_note:
                fatigue_note = narrate("PITCHER_GASSED", pitcher_name)
        elif runs > 0 and stage_after in ("tired", "gassed") and not fatigue_note:
            # Collapse: fatigued pitcher allows runs (no transition this PA).
            # Choose language that matches how the run actually scored —
            # "hard contact" is wrong when the run came via a walk or HBP.
            _collapse_outcome = resolved_outcome or outcome
            if _collapse_outcome in ("BB", "HBP"):
                fatigue_note = narrate("PITCHER_COLLAPSE_WALK", pitcher_name)
            else:
                fatigue_note = narrate("PITCHER_COLLAPSE", pitcher_name)

        # Record the receipt — pitcher_id/name reflect who actually faced the batter
        pa_events.append(PAEvent(
            inning               = state.inning,
            half                 = state.half,
            pa_number            = pa_counter[0],
            batter_id            = batter.get("player_id", "unknown"),
            batter_name          = batter.get("name", "unknown"),
            pitcher_id           = ps.current.get("player_id", "unknown"),
            pitcher_name         = ps.current.get("name", "unknown"),
            outcome              = resolved_outcome or outcome,
            outs_before          = outs_before,
            outs_recorded        = outs_recorded,
            bases_before         = bases_snap,
            base_runners_before  = runners_snap,
            runs_scored          = runs,
            seed                 = seed,
            raw_result           = raw_result,
            fielder_zone         = fielder_zone,
            baserunning_note     = br_note,
            fatigue_note         = fatigue_note,
        ))

        # Walk-off: home takes the lead in the bottom of inning >= 9
        if check_walkoff and state.score["home"] > state.score["away"]:
            walk_off = True
            break

    return half_runs, walk_off


# ── Main simulation entry point ─────────────────────────────────────────────────

def simulate_game(
    away_team,
    home_team,
    game_id: str | None = None,
    verbose: bool = False,
) -> BoxScore:
    """
    Simulate a complete baseball game and return a BoxScore.

    Args:
        away_team : Team dict {"team_id", "lineup", "pitcher"} or list of player cards.
        home_team : Same format as away_team.
        game_id   : Optional string identifier for deterministic seed derivation.
                    Defaults to a random UUID4. Pass the same ID to replay identically.

    Returns:
        BoxScore with final_score, linescore, pa_events, innings_played, walk_off.

    Loop invariant:
        Plays while inning <= 9 OR the score is tied.
        Extra innings continue until a winner exists after a complete inning.
        Walk-offs end the game immediately within a bottom-half PA.
        Home team skips their bottom half if they already lead after top of inning >= 9.
    """
    if game_id is None:
        game_id = str(uuid.uuid4())

    away_id, away_lineup, away_pitcher, away_bullpen = _parse_team(away_team, "AWAY")
    home_id, home_lineup, home_pitcher, home_bullpen = _parse_team(home_team, "HOME")

    # Outfield ARM for each team's defense (used in baserunning decisions)
    away_arm = int(away_team.get("arm", 55)) if isinstance(away_team, dict) else 55
    home_arm = int(home_team.get("arm", 55)) if isinstance(home_team, dict) else 55

    # Real defensive alignment: {position: player_name} built from field_pos data.
    # away team defends when home bats (bottom half) and vice versa.
    away_def_align = (away_team.get("defensive_alignment", {})
                      if isinstance(away_team, dict) else {})
    home_def_align = (home_team.get("defensive_alignment", {})
                      if isinstance(home_team, dict) else {})

    # PitcherState persists across all innings; fatigue accumulates all game long
    away_ps = PitcherState(current=away_pitcher, bullpen=list(away_bullpen))
    home_ps = PitcherState(current=home_pitcher, bullpen=list(home_bullpen))

    state       = GameState()
    pa_events: list[PAEvent] = []
    pa_counter  = [0]                        # mutable so _play_half_inning can increment it
    linescore   = {"away": [], "home": []}
    pitcher_log = []                         # per-half-inning PitcherState snapshots
    walk_off    = False
    last_inning = 1

    while True:
        current_inning = state.inning
        last_inning    = current_inning

        # ── TOP HALF — away team bats vs home pitching + home defense ────────
        state.half = "top"
        away_runs, _ = _play_half_inning(
            state, "away", away_lineup, home_ps,
            pa_events, pa_counter, game_id,
            fielder_arm=home_arm,
        )
        linescore["away"].append(away_runs)
        pitcher_log.append({
            "inning": current_inning, "half": "top",
            "pitching_team": home_id,
            **home_ps.snapshot(),
        })
        _end_half_inning(state)             # state.half → "bottom"

        # Home already leads after top of inning >= 9 → they don't need to bat
        if current_inning >= 9 and state.score["home"] > state.score["away"]:
            linescore["home"].append(0)     # empty bottom half
            break

        # ── BOTTOM HALF — home team bats vs away pitching + away defense ──────
        state.half = "bottom"
        home_runs, walk_off = _play_half_inning(
            state, "home", home_lineup, away_ps,
            pa_events, pa_counter, game_id,
            fielder_arm=away_arm,
            check_walkoff=(current_inning >= 9),
        )
        linescore["home"].append(home_runs)
        pitcher_log.append({
            "inning": current_inning, "half": "bottom",
            "pitching_team": away_id,
            **away_ps.snapshot(),
        })
        _end_half_inning(state)             # state.half → "top", state.inning += 1

        # Walk-off: home won mid-inning
        if walk_off:
            break

        # After a complete inning >= 9: if the score is no longer tied, game over
        if current_inning >= 9 and state.score["away"] != state.score["home"]:
            break

    box = BoxScore(
        away_team_id   = away_id,
        home_team_id   = home_id,
        final_score    = dict(state.score),
        linescore      = linescore,
        pa_events      = pa_events,
        innings_played = last_inning,
        walk_off       = walk_off,
        pitcher_log    = pitcher_log,
        away_lineup    = away_lineup,
        home_lineup    = home_lineup,
        away_def_align = away_def_align,
        home_def_align = home_def_align,
    )

    if verbose:
        print_game_log(box)

    return box


# ── Display utilities ───────────────────────────────────────────────────────────

def print_game_log(box: BoxScore) -> None:
    """
    Print a complete game log derived entirely from box.pa_events.

    Sections:
        1. Inning-by-inning run line (with "x" for unplayed home last half)
        2. Play-by-play — one line per PA showing batter, bases, outcome, runs
        3. Batting stats  — AB / H / HR / RBI / BB per player
        4. Pitching stats — IP / H / R / BB / K per pitcher

    No re-simulation: reads only data already in BoxScore.
    """
    HITS = {"Single", "Double", "Triple", "HR", "InfieldHit"}
    OUTS = {"K", "Out"}

    away = box.away_team_id
    home = box.home_team_id
    n    = box.innings_played

    # Detect unplayed bottom of final inning (home won without batting last half).
    # simulate_game appends a 0 placeholder; we show "x" instead.
    home_batted_last = any(
        ev.inning == n and ev.half == "bottom" for ev in box.pa_events
    )
    unplayed_bottom = (
        not home_batted_last
        and box.final_score["home"] > box.final_score["away"]
    )

    # ── 1. Run line ──────────────────────────────────────────────────────────────
    col_w    = 3
    header   = f"{'':12}" + "".join(f"{i + 1:>{col_w}}" for i in range(n)) + "  │  R"
    line_sep = "═" * len(header)
    row_sep  = "─" * len(header)

    def _cells(runs_list, is_home: bool) -> str:
        out = []
        for i in range(n):
            if i < len(runs_list):
                if is_home and unplayed_bottom and i == n - 1:
                    out.append(f"{'x':>{col_w}}")
                else:
                    out.append(f"{runs_list[i]:>{col_w}}")
            else:
                out.append(" " * col_w)
        return "".join(out)

    walkoff_tag = "  ← walk-off" if box.walk_off else ""

    print(f"\n{line_sep}")
    print(f"{away:^{len(header) // 2}}  vs  {home}")
    print(header)
    print(row_sep)
    print(f"{away:<12}" + _cells(box.linescore["away"], False) + f"  │ {box.final_score['away']:>2}")
    print(f"{home:<12}" + _cells(box.linescore["home"], True)  + f"  │ {box.final_score['home']:>2}{walkoff_tag}")
    print(line_sep)

    if not box.pa_events:
        return

    # ── Accumulate per-batter and per-pitcher stats in a single pass ─────────────
    hitter_order:  dict[str, list] = {"away": [], "home": []}
    hitter_stats:  dict[str, dict] = {}
    pitcher_order: dict[str, list] = {"away": [], "home": []}
    pitcher_stats: dict[str, dict] = {}

    for ev in box.pa_events:
        bat_side = "away" if ev.half == "top" else "home"
        pit_side = "home" if ev.half == "top" else "away"

        # Batter
        if ev.batter_id not in hitter_stats:
            hitter_stats[ev.batter_id] = {
                "name": ev.batter_name, "ab": 0, "h": 0, "hr": 0, "rbi": 0, "bb": 0,
            }
            hitter_order[bat_side].append(ev.batter_id)
        hs = hitter_stats[ev.batter_id]
        if ev.outcome in ("BB", "HBP"):
            if ev.outcome == "BB":
                hs["bb"] += 1
        else:
            hs["ab"] += 1
        if ev.outcome in HITS:
            hs["h"] += 1
        if ev.outcome == "HR":
            hs["hr"] += 1
        hs["rbi"] += ev.runs_scored

        # Pitcher
        if ev.pitcher_id not in pitcher_stats:
            pitcher_stats[ev.pitcher_id] = {
                "name": ev.pitcher_name, "outs": 0, "h": 0, "r": 0, "bb": 0, "k": 0,
            }
            pitcher_order[pit_side].append(ev.pitcher_id)
        ps = pitcher_stats[ev.pitcher_id]
        # outs_recorded correctly counts all outs (1 for routine, 2 for DP)
        ps["outs"] += ev.outs_recorded
        if ev.outcome == "K":
            ps["k"] += 1
        if ev.outcome in HITS:
            ps["h"] += 1
        if ev.outcome == "BB":
            ps["bb"] += 1
        ps["r"] += ev.runs_scored

    W = 62   # width for PBP and stat section separators

    # Outcome string → narrative key
    _OUTCOME_KEY: dict[str, str] = {
        "K":           "STRIKEOUT",
        "BB":          "WALK",
        "HBP":         "HIT_BY_PITCH",
        "Single":      "SINGLE",
        "Double":      "DOUBLE",
        "Triple":      "TRIPLE",
        "HR":          "HOME_RUN",
        "Out":         "OUT",
        "Error":       "REACHED_ON_ERROR",
        "InfieldHit":  "INFIELD_HIT",
        "FC":          "FIELDERS_CHOICE",
    }

    # ── 2. Play-by-play ──────────────────────────────────────────────────────────
    def _base_str(bases: list) -> str:
        return (
            ("1" if bases[0] else "_") +
            ("2" if bases[1] else "_") +
            ("3" if bases[2] else "_")
        )

    print(f"\n{'─' * W}")
    print("  PLAY BY PLAY")
    print(f"{'─' * W}")

    def _build_hr_narrative(ev_: "PAEvent", score_: dict[str, int]) -> str:
        """Return a situation-aware HR narrative string for this PAEvent."""
        bat_side_  = "away" if ev_.half == "top" else "home"
        pit_side_  = "home" if ev_.half == "top" else "away"
        bat_team_  = away   if ev_.half == "top" else home
        runs_      = ev_.runs_scored
        name_      = ev_.batter_name

        # ── Multi-run HRs: name the runners who score ─────────────────────────
        if runs_ >= 4:
            return narrate("HOME_RUN_GRAND_SLAM", name_)
        if runs_ == 3:
            scorers_  = [r["name"] for r in (ev_.base_runners_before or []) if r]
            runner1_  = scorers_[0] if len(scorers_) > 0 else "the first runner"
            runner2_  = scorers_[1] if len(scorers_) > 1 else "the second runner"
            return narrate("HOME_RUN_THREE_RUN", name_, runner1=runner1_, runner2=runner2_)
        if runs_ == 2:
            scorers_  = [r["name"] for r in (ev_.base_runners_before or []) if r]
            runner1_  = scorers_[0] if scorers_ else "the runner"
            return narrate("HOME_RUN_TWO_RUN", name_, runner1=runner1_)

        # ── Solo HR: choose template by game situation ─────────────────────────
        score_bat_before_ = score_[bat_side_]
        score_pit_before_ = score_[pit_side_]
        score_bat_after_  = score_bat_before_ + runs_

        go_ahead_ = score_bat_after_ > score_pit_before_
        was_close_ = abs(score_bat_before_ - score_pit_before_) <= 1
        late_      = ev_.inning >= 7

        if late_ and go_ahead_ and was_close_:
            return narrate("HOME_RUN_TIEBREAKER", name_, team=bat_team_)
        if go_ahead_:
            return narrate("HOME_RUN_GOAHEAD", name_, team=bat_team_)
        if late_:
            return narrate("HOME_RUN_SOLO_LATE", name_)
        return narrate("HOME_RUN", name_)

    def _build_scorer_tag(ev_: "PAEvent") -> str:
        """
        Return a scoring suffix like ' — Garciaparra scores.' for any non-HR
        play that scores at least one run.  HR narratives already name their
        scorers, so skip them here.  Returns "" when no runs scored.
        """
        if ev_.runs_scored <= 0 or ev_.outcome == "HR":
            return ""

        runners = ev_.base_runners_before   # [r1_card|None, r2_card|None, r3_card|None]
        bases   = ev_.bases_before          # [bool, bool, bool]  — 1B/2B/3B occupancy

        def _rname(idx: int) -> str:
            if idx >= len(runners) or idx >= len(bases):
                return ""
            if not bases[idx]:
                return ""
            r = runners[idx]
            return (r or {}).get("name", "") if r else ""

        r1, r2, r3 = _rname(0), _rname(1), _rname(2)
        scorers: list[str] = []

        if ev_.outcome == "Triple":
            # All runners on base score
            scorers = [n for n in [r3, r2, r1] if n]
        elif ev_.outcome == "InfieldHit":
            # Only the 3B runner scores on an infield hit
            if r3:
                scorers = [r3]
        elif ev_.outcome in ("BB", "HBP"):
            # Force-advancement: only r3 scores (loaded bases forces r3 home)
            if r3:
                scorers = [r3]
        elif ev_.outcome == "Out":
            # The only way a run scores on an Out is a DP when r2+r3 both existed
            # (r3 is forced home as r2 takes 3B).  Only r3 scores.
            if r3:
                scorers = [r3]
        else:
            # Single, Double, Error: r3 and r2 always score (deterministic);
            # r1 also scores if runs_scored exceeds the det count (baserunning decision).
            det = 0
            if r3:
                scorers.append(r3); det += 1
            if r2:
                scorers.append(r2); det += 1
            if ev_.runs_scored > det and r1:
                scorers.append(r1)

        if not scorers:
            return ""
        if len(scorers) == 1:
            return f" — {scorers[0]} scores."
        if len(scorers) == 2:
            return f" — {scorers[0]} and {scorers[1]} score."
        return f" — {', '.join(scorers[:-1])}, and {scorers[-1]} all score."

    # Running score — updated after each event so HR situation detection
    # always reads the score as it stood BEFORE the current plate appearance.
    running_score: dict[str, int] = {"away": 0, "home": 0}
    grounder_history: list[str] = []    # last-2 grounder templates; prevents repeat phrasing

    current_key = None
    _mid_inning_change = False   # flag: current PA is a mid-inning pitcher change
    # Track the last known pitcher separately for each half so alternating
    # Pedro / Hoyt / Pedro across innings is not mis-read as a substitution.
    # Key = "top" or "bottom" (which side is pitching).
    # Value = pitcher_id of the last arm seen pitching in that half-context,
    #         or None before the very first appearance for that side.
    last_pit_by_half: dict[str, str | None] = {"top": None, "bottom": None}

    for ev in box.pa_events:
        key  = (ev.inning, ev.half)
        half = ev.half

        if key != current_key:
            current_key = key
            bat_team   = away if half == "top" else home
            half_label = "TOP" if half == "top" else "BOT"
            label      = f"  INN {ev.inning} {half_label}  ({bat_team} batting) "
            print(f"\n{label}{'─' * max(0, W - len(label))}")

            # Pitcher announcement — compare against the last arm for THIS half.
            prev_pit = last_pit_by_half[half]
            if ev.pitcher_id != prev_pit:
                if prev_pit is None:
                    # First time this pitching side has appeared — game opener line.
                    print(f"  ⚾  {narrate('PITCHER_STARTS', ev.pitcher_name)}")
                else:
                    # Between-inning substitution (different arm than last time this
                    # side pitched).  fatigue_note carries mid-inning changes, but a
                    # between-inning swap has no fatigue_note, so announce it here.
                    print(f"  ⚾  {narrate('PITCHING_CHANGE', ev.pitcher_name)}")
                last_pit_by_half[half] = ev.pitcher_id

        elif ev.pitcher_id != last_pit_by_half[half]:
            # Mid-inning substitution — fatigue_note on this PAEvent carries the
            # PITCHING_CHANGE narrative.  Set flag so the note prints BEFORE this
            # PA (it announces the new pitcher, not reacts to the previous play).
            _mid_inning_change = True
            last_pit_by_half[half] = ev.pitcher_id

        # _mid_inning_change was set in the elif branch just above; default False otherwise.
        is_mid_inning_change = _mid_inning_change
        _mid_inning_change = False   # reset for the next iteration

        # Build play narrative.
        # ── Double play detection ────────────────────────────────────────────
        # outs_recorded == 2 with outcome "Out" is the unique DP fingerprint:
        # _apply_outcome only posts two outs via this path when a DP fires.
        # Use DOUBLE_PLAY_PRIMARY as the sole main-line narrative — it already
        # tells the whole story, so the ↳ sub-note is suppressed.
        is_double_play = (ev.outcome == "Out" and ev.outs_recorded == 2)

        if is_double_play:
            outcome_key = "DOUBLE_PLAY_PRIMARY"
        elif ev.outcome == "Out":
            outcome_key = "OUT"   # fallback; overridden below if zone is available
        else:
            outcome_key = _OUTCOME_KEY.get(ev.outcome, "OUT")

        # ── Build narrative — single source of truth: ev fields drive all choices ──
        if ev.outcome == "HR":
            narrative = _build_hr_narrative(ev, running_score)
        elif ev.outcome == "FC":
            # STEP 5: name the specific runner thrown out at second (always 1B runner)
            r1_card     = ev.base_runners_before[0] if ev.base_runners_before else None
            runner_out  = (r1_card or {}).get("name", "the runner") if r1_card else "the runner"
            narrative   = narrate("FIELDERS_CHOICE", ev.batter_name, runner_out=runner_out)
        elif ev.outcome == "Out" and ev.fielder_zone and not is_double_play:
            # Zone-aware fielder-named out narrative using real defensive alignment.
            # Defensive team: home defends when away bats (top), away defends when home bats (bottom).
            zone_meta = _ZONE_META.get(ev.fielder_zone)
            if zone_meta:
                def_pos, tmpl_key, pos_name = zone_meta
                def_align = box.home_def_align if ev.half == "top" else box.away_def_align

                if def_pos == "P":
                    fielder_name = ev.pitcher_name
                elif def_pos == "IF":
                    if_positions = [p for p in ["SS", "2B", "3B", "1B"] if p in def_align]
                    if if_positions:
                        pos_key = if_positions[ev.pa_number % len(if_positions)]
                        fielder_name = def_align[pos_key]
                    else:
                        fielder_name = "the infielder"
                else:
                    fielder_name = def_align.get(def_pos, f"the {def_pos}")

                if tmpl_key == "OUT_GROUNDER":
                    narrative = narrate_grounder(
                        ev.batter_name, fielder_name, def_pos, pos_name,
                        grounder_history,
                    )
                else:
                    narrative = narrate(tmpl_key, ev.batter_name,
                                        fielder=fielder_name, pos=pos_name)
            else:
                narrative = narrate(outcome_key, ev.batter_name)
        else:
            narrative = narrate(outcome_key, ev.batter_name)

        # STEP 6: embed scorer names for all non-HR run-scoring plays
        narrative += _build_scorer_tag(ev)

        # ── Inning closure: 3rd-out plays get closing language ───────────────
        _INNING_CLOSERS = [
            "Side retired.", "Three out.", "Inning over.",
            "That's three.", "And three.",
        ]
        is_third_out = (ev.outs_before + ev.outs_recorded >= 3)
        if is_third_out:
            # Strip count-specific phrases that mislead on final out
            for _stale in ("One away.", "one away.", "Two away.", "two away.",
                           "One down.", "one down.", "One out.", "one out.",
                           "Routine.", "routine."):
                narrative = narrative.replace(_stale, "").rstrip()
            # FC on 3rd out: batter hit the grounder, runner is the 3rd out.
            if ev.outcome == "FC":
                r1_card    = ev.base_runners_before[0] if ev.base_runners_before else None
                runner_out = (r1_card or {}).get("name", "the runner") if r1_card else "the runner"
                narrative  = (
                    f"{ev.batter_name} hits a grounder — "
                    f"{runner_out} thrown out at second. Side retired."
                )
            else:
                _closer = random.choice(_INNING_CLOSERS)
                if narrative and not narrative.endswith("."):
                    narrative += "."
                narrative += f" {_closer}"

        outs_str   = f"[{ev.outs_before} out{'s' if ev.outs_before != 1 else ' '}]"
        base_str   = _base_str(ev.bases_before)
        base_label = f"on: {base_str}" if any(ev.bases_before) else "     "

        # PITCHING_CHANGE announcements (is_mid_inning_change) print BEFORE the PA
        # so the new pitcher is identified before his first batter.
        # TIRED / GASSED / COLLAPSE notes print AFTER the play (reactive).
        if ev.fatigue_note and is_mid_inning_change:
            print(f"  ⚡  {ev.fatigue_note}")

        print(f"  {outs_str:<9} {base_label}  {narrative}")

        if ev.fatigue_note and not is_mid_inning_change:
            print(f"  ⚡  {ev.fatigue_note}")

        # Advance the running-score tracker AFTER this PA is displayed so that
        # HR situation detection always sees the score as it was before the hit.
        bat_side_ev = "away" if ev.half == "top" else "home"
        running_score[bat_side_ev] += ev.runs_scored

    # ── 3. Batting stats ─────────────────────────────────────────────────────────
    BAT_HDR = f"  {'Name':<24}  AB   H  HR RBI  BB"
    BAT_SEP = "  " + "─" * (len(BAT_HDR) - 2)

    print(f"\n{'═' * W}")
    print("  BATTING")
    print(f"{'═' * W}")
    for side, team_id in [("away", away), ("home", home)]:
        print(f"\n  {team_id}")
        print(BAT_HDR)
        print(BAT_SEP)
        for bid in hitter_order[side]:
            s = hitter_stats[bid]
            print(f"  {s['name']:<24} {s['ab']:>3}  {s['h']:>2}  {s['hr']:>2}  {s['rbi']:>2}  {s['bb']:>2}")

    # ── 4. Pitching stats ────────────────────────────────────────────────────────
    PIT_HDR = f"  {'Name':<24}   IP   H   R  BB   K"
    PIT_SEP = "  " + "─" * (len(PIT_HDR) - 2)

    print(f"\n{'═' * W}")
    print("  PITCHING")
    print(f"{'═' * W}")
    for side, team_id in [("away", away), ("home", home)]:
        print(f"\n  {team_id}")
        print(PIT_HDR)
        print(PIT_SEP)
        for pid in pitcher_order[side]:
            p = pitcher_stats[pid]
            ip_str = f"{p['outs'] // 3}.{p['outs'] % 3}"
            print(f"  {p['name']:<24} {ip_str:>5}  {p['h']:>2}  {p['r']:>2}  {p['bb']:>2}  {p['k']:>2}")

    # ── Footer: repeat the final score so it stays visible at the bottom ─────────
    away_r = box.final_score["away"]
    home_r = box.final_score["home"]
    winner = away if away_r > home_r else home
    loser  = home if away_r > home_r else away
    w_r    = max(away_r, home_r)
    l_r    = min(away_r, home_r)
    wo_tag = "  (walk-off)" if box.walk_off else ""

    print(f"\n{'═' * W}")
    print(f"  FINAL:  {winner} {w_r},  {loser} {l_r}{wo_tag}")
    print(f"  {box.innings_played} innings  ·  {len(box.pa_events)} plate appearances")
    print(f"{'═' * W}\n")


def print_box_score(box: BoxScore) -> None:
    """Thin wrapper kept for backwards compatibility — delegates to print_game_log."""
    print_game_log(box)
