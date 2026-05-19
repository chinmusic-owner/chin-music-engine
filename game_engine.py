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
from narrative_dictionary import narrate


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
    baserunning_note: str = ""   # Narrative for any extra-base attempt on this PA


@dataclass
class BoxScore:
    """Complete game record returned by simulate_game."""
    away_team_id: str
    home_team_id: str
    final_score: dict       # {"away": int, "home": int}
    linescore: dict         # {"away": [int, ...], "home": [int, ...]} — one entry per inning
    pa_events: list         # list[PAEvent]
    innings_played: int
    walk_off: bool = False


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
    Tracks the active pitcher and fatigue state for one team's staff.

    Stamina model
    ─────────────
    Each pitcher has a STA trait.  max_pa is derived from STA and role:
        SP  →  round(STA × 0.40)   e.g. STA=75  →  30 PA  (~7 innings)
        RP  →  round(STA × 0.20)   e.g. STA=50  →  10 PA  (~2 innings)

    Once pa_faced >= max_pa the pitcher is "over-limit":
        • effective_card() applies a 2-point STF/CTL penalty per extra batter
          (floor of 40 on any trait).
        • should_pull() returns True if:
            - 6 or more batters faced over max_pa  (hard limit: auto-hook), OR
            - 2 or more runs allowed while over-limit  (soft limit: manager hook).

    Bullpen cycling
    ───────────────
    pull_next() replaces current with bullpen[0] and resets per-pitcher counters.
    If the bullpen is empty the tired starter stays in.
    """
    current:          dict
    pa_faced:         int  = 0
    runs_while_tired: int  = 0
    bullpen:          list = field(default_factory=list)

    # Tuning constants
    _HARD_OVER_LIMIT: int   = 6     # auto-hook after this many batters over limit
    _SOFT_RUNS_LIMIT: int   = 2     # hook if pitcher allows this many runs while tired
    _FATIGUE_RATE:    int   = 2     # STF/CTL points lost per extra batter
    _FATIGUE_FLOOR:   int   = 40    # minimum value after fatigue degrades a trait

    @property
    def max_pa(self) -> int:
        sta  = self.current["traits"].get("STA", 60)
        role = self.current.get("pitcher_role", "SP")
        mult = 0.40 if role == "SP" else 0.20
        return max(3, round(sta * mult))

    @property
    def is_over_limit(self) -> bool:
        return self.pa_faced >= self.max_pa

    @property
    def tired_pa(self) -> int:
        """Batters faced above the stamina limit (0 when not yet over-limit)."""
        return max(0, self.pa_faced - self.max_pa)

    def effective_card(self) -> dict:
        """Return the pitcher card with fatigue-adjusted STF and CTL."""
        tp = self.tired_pa
        if tp <= 0:
            return self.current
        penalty = tp * self._FATIGUE_RATE
        t = self.current["traits"]
        return {
            **self.current,
            "traits": {
                **t,
                "STF": max(self._FATIGUE_FLOOR, t["STF"] - penalty),
                "CTL": max(self._FATIGUE_FLOOR, t["CTL"] - penalty),
            },
        }

    def should_pull(self) -> bool:
        """True if the pitcher should be replaced before the next batter."""
        if not self.is_over_limit:
            return False
        if self.tired_pa >= self._HARD_OVER_LIMIT:
            return True
        if self.runs_while_tired >= self._SOFT_RUNS_LIMIT:
            return True
        return False

    def pull_next(self) -> bool:
        """
        Swap in the next bullpen pitcher.
        Resets per-pitcher counters.  Returns True if a change was made.
        """
        if not self.bullpen:
            return False
        self.current          = self.bullpen.pop(0)
        self.pa_faced         = 0
        self.runs_while_tired = 0
        return True

    def record_pa(self, runs: int) -> None:
        """Update counters after one PA has been resolved."""
        self.pa_faced += 1
        if self.is_over_limit:
            self.runs_while_tired += runs


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
            br_note = narrate("INFIELD_HIT", batter_name)
            state.score[side] += runs
            return runs, br_note, "InfieldHit"

        # ── Stage 2: Groundout — batter out at 1B, possible DP ───────────────
        state.outs += 1
        if state.outs < 3 and b[0]:
            runner    = br[0]
            runner_spd = runner.get("traits", {}).get("SPD", 50) if runner else 50
            dp_base   = _DP_BASE.get(contact_quality, 0.475)
            dp_chance = max(0.05, min(0.80, dp_base + (50 - runner_spd) * _SPD_DP_MOD))
            if rng_dp.random() < dp_chance:
                state.outs += 1
                b[0] = False;  br[0] = None
                br_note = narrate("DOUBLE_PLAY", batter_name)
            elif contact_quality == "Weak":
                br_note = narrate("WEAK_ROLLER_NO_DP", batter_name)
        return 0, br_note, ""

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
    half_runs = 0
    walk_off  = False

    while state.outs < 3:
        # ── Manager hook: pull tired pitcher before this batter ──────────────
        if ps.should_pull():
            ps.pull_next()          # swap in next bullpen arm; resets per-pitcher counters

        # Snapshot pre-PA state for the receipt (bool flags)
        bases_snap  = list(state.bases)
        outs_before = state.outs

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

        # Update pitcher's fatigue counters for this PA
        ps.record_pa(runs)

        # Record the receipt — pitcher_id/name reflect who actually faced the batter
        pa_events.append(PAEvent(
            inning           = state.inning,
            half             = state.half,
            pa_number        = pa_counter[0],
            batter_id        = batter.get("player_id", "unknown"),
            batter_name      = batter.get("name", "unknown"),
            pitcher_id       = ps.current.get("player_id", "unknown"),
            pitcher_name     = ps.current.get("name", "unknown"),
            outcome          = resolved_outcome or outcome,
            outs_before      = outs_before,
            bases_before     = bases_snap,
            runs_scored      = runs,
            seed             = seed,
            raw_result       = raw_result,
            baserunning_note = br_note,
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

    # PitcherState persists across all innings; fatigue accumulates all game long
    away_ps = PitcherState(current=away_pitcher, bullpen=list(away_bullpen))
    home_ps = PitcherState(current=home_pitcher, bullpen=list(home_bullpen))

    state      = GameState()
    pa_events: list[PAEvent] = []
    pa_counter = [0]                        # mutable so _play_half_inning can increment it
    linescore  = {"away": [], "home": []}
    walk_off   = False
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
        if ev.outcome in OUTS:
            ps["outs"] += 1
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

    current_key      = None
    current_pit      = None      # pitcher_id of whichever arm is on the mound
    current_pit_name = None      # name of that pitcher (for PITCHER_TIRED line)

    for ev in box.pa_events:
        key = (ev.inning, ev.half)

        if key != current_key:
            current_key = key
            bat_team    = away if ev.half == "top" else home
            half_label  = "TOP" if ev.half == "top" else "BOT"
            label       = f"  INN {ev.inning} {half_label}  ({bat_team} batting) "
            print(f"\n{label}{'─' * max(0, W - len(label))}")
            # Pitcher announcement at top of each half-inning
            if ev.pitcher_id != current_pit:
                if current_pit is None:
                    # Game opener — use PITCHER_STARTS
                    print(f"  ⚾  {narrate('PITCHER_STARTS', ev.pitcher_name)}")
                else:
                    # Between-inning change
                    print(f"  ⚾  {narrate('PITCHER_CHANGE', ev.pitcher_name)}")
                current_pit      = ev.pitcher_id
                current_pit_name = ev.pitcher_name

        elif ev.pitcher_id != current_pit:
            # Mid-inning change — outgoing pitcher was tired; new arm enters
            print(f"\n  {narrate('PITCHER_TIRED', current_pit_name)}")
            print(f"  ⚾  {narrate('PITCHER_CHANGE', ev.pitcher_name)}")
            current_pit      = ev.pitcher_id
            current_pit_name = ev.pitcher_name

        # Build play narrative
        outcome_key = _OUTCOME_KEY.get(ev.outcome, "OUT")
        narrative   = narrate(outcome_key, ev.batter_name)

        # Build run note
        if ev.runs_scored == 1:
            run_note = "  " + narrate("RUN_SCORES")
        elif ev.runs_scored > 1:
            run_note = "  " + narrate("RUNS_SCORE", count=ev.runs_scored)
        else:
            run_note = ""

        outs_str = f"[{ev.outs_before} out{'s' if ev.outs_before != 1 else ' '}]"
        base_str = _base_str(ev.bases_before)
        print(f"  {outs_str:<9} {base_str}  {narrative}{run_note}")
        if ev.baserunning_note:
            print(f"             {'':5}    ↳ {ev.baserunning_note}")

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
