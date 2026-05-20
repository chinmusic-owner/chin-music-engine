"""
roster_manager.py — PRD 05: Lineup & Staff Loader

Builds game-ready TeamRoster objects from two sources:
    1. Supabase `players` table (load_team)
    2. Local pilot JSON files produced by ratings.py (load_team_from_pilot)

Public API:
    load_team(season_year, team_name, use_dh=True, rotation_size=5) -> TeamRoster
    load_team_from_pilot(json_path, team_id=None, use_dh=True, rotation_size=5) -> TeamRoster

    TeamRoster.lineup_card   -> Lineup          (batting order + defensive map + bench)
    TeamRoster.staff         -> PitchingStaff   (rotation depth chart + bullpen)
    TeamRoster.for_game_engine(rotation_index, rotation_size) -> dict   # plug into simulate_game()

Data flow:
    Supabase row
        ↓ _batter_card_from_row / _pitcher_card_from_row
    Raw player card (field_pos may be None here — preserved for inspection)
        ↓ _assemble_roster
    Lineup + PitchingStaff (field_pos fallback to "UTIL" enforced here, slot assigned)
        ↓ TeamRoster.for_game_engine()
    simulate_game() dict (defensive_alignment fully populated, no None values)

Schema notes (Supabase `players` table as of PRD 05):
    Hitter traits   : contact, power, eye, speed
    Pitcher traits  : stuff, control, movement
    Role fields     : pitcher_role ('SP'|'RP'|null), field_pos ('SS'|'OF'|'CF'|…|null)
    Missing columns : bats, throws, pa, ip, ak, gap, sta — all derived or defaulted here.

PA Engine trait mapping (Supabase column → engine key):
    contact → CON
    power   → POW
    eye     → EYE
    AK      ≈ contact  (bat-to-ball skill correlates with contact rating)
    GAP     ≈ (contact + power) // 2  (gap power between raw power and contact)
    stuff   → STF
    control → CTL
    movement → CMD
    STA     → derived from pitcher_role (SP=75, RP=50)
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field

from database import supabase


# ── Team → Lahman teamID map ────────────────────────────────────────────────────
# Maps common user abbreviations to the canonical Lahman teamID(s) stored in the
# `team` column by ingest_historical.py.  A list allows historical franchise
# renames (e.g., Devil Rays → Rays both map to the same user alias "TBR").
_TEAM_LAHMAN_IDS: dict[str, list[str]] = {
    # American League
    "NYY": ["NYA"],                     # Yankees used NYA in Lahman throughout
    "NYA": ["NYA"],
    "BOS": ["BOS"],
    "BAL": ["BAL"],
    "TBR": ["TBA"],                     # Rays (was TBD / Devil Rays pre-2008)
    "TBD": ["TBA", "TBD"],
    "TOR": ["TOR"],
    "CLE": ["CLE"],                     # Cleveland — Guardians since 2022
    "CLG": ["CLE"],
    "MIN": ["MIN"],
    "CWS": ["CHA"],
    "CHA": ["CHA"],
    "DET": ["DET"],
    "KCR": ["KCA"],
    "KCA": ["KCA"],
    "HOU": ["HOU"],
    "OAK": ["OAK"],
    "SEA": ["SEA"],
    "LAA": ["LAA", "ANA", "CAL"],       # Angels franchise IDs across eras
    "ANA": ["ANA"],
    "CAL": ["CAL"],
    "TEX": ["TEX"],
    # National League
    "NYM": ["NYN"],
    "PHI": ["PHI"],
    "ATL": ["ATL"],
    "MIA": ["MIA", "FLO"],              # Marlins (was Florida pre-2012)
    "FLA": ["FLO"],
    "WSN": ["WAS"],                     # Nationals (was Montreal Expos pre-2005)
    "MON": ["MON"],
    "CHC": ["CHN"],
    "STL": ["SLN"],
    "MIL": ["MIL"],
    "CIN": ["CIN"],
    "PIT": ["PIT"],
    "SFG": ["SFN"],
    "SFN": ["SFN"],
    "LAD": ["LAN"],
    "LAN": ["LAN"],
    "SDP": ["SDN"],
    "SDN": ["SDN"],
    "COL": ["COL"],
    "ARI": ["ARI"],
}

# STA defaults by pitcher role (not stored in DB)
_STA_DEFAULTS: dict[str, int] = {"SP": 75, "RP": 50}

# All nine defensive slots the narrative engine can route to.
# validate() ensures every slot is filled before the lineup reaches the engine.
_ALL_DEF_POSITIONS: frozenset[str] = frozenset(
    {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "P"}
)


# ── Role detection ──────────────────────────────────────────────────────────────

def _is_pitcher_row(row: dict) -> bool:
    """
    True if this DB row represents a pitcher.

    Primary signal: pitcher_role column ('SP' or 'RP').
    Fallback heuristic: pitcher traits present + hitter traits absent.
    Handles old-ingestion rows where pitcher_role was not written (e.g., Pedro 2000).
    """
    if row.get("pitcher_role") in ("SP", "RP"):
        return True
    pitcher_score = (
        int(row.get("stuff")    or 0) +
        int(row.get("control")  or 0) +
        int(row.get("movement") or 0)
    )
    hitter_score = (
        int(row.get("contact") or 0) +
        int(row.get("power")   or 0) +
        int(row.get("eye")     or 0)
    )
    return pitcher_score > 0 and hitter_score == 0


def _is_hitter_row(row: dict) -> bool:
    """True if this DB row represents a position player (has hitter traits, not a pitcher)."""
    if _is_pitcher_row(row):
        return False
    return (
        int(row.get("contact") or 0) +
        int(row.get("power")   or 0) +
        int(row.get("eye")     or 0)
    ) > 0


# ── DB row → engine card ────────────────────────────────────────────────────────

def _batter_card_from_row(row: dict) -> dict:
    """
    Map a Supabase players row to a pa_wrapper-compatible batter card.

    All six traits and pa are read directly from DB columns written by
    ingest_historical.py.  load_team() filters for data_source='historical',
    so ak, gap, and pa are always present — no proxy derivation here.

    If ak or gap is None despite the filter (schema bug, partial upsert, etc.)
    a ValueError is raised immediately so the problem is visible rather than
    silently producing wrong engine inputs.

    field_pos is preserved as-is here (may be None for some historical rows).
    The fallback to "UTIL" is enforced in _assemble_roster at the boundary
    where field_pos must be non-null for the narrative engine.
    """
    con  = int(row.get("contact") or 50)
    pow_ = int(row.get("power")   or 50)
    eye  = int(row.get("eye")     or 50)
    pa   = int(row.get("pa")  or 0)

    ak_raw  = row.get("ak")
    gap_raw = row.get("gap")
    if ak_raw is None or gap_raw is None:
        name = row.get("player_name", row.get("player_id", "unknown"))
        raise ValueError(
            f"_batter_card_from_row: ak={ak_raw!r} gap={gap_raw!r} are NULL for "
            f"'{name}' (season={row.get('season_year')}, data_source="
            f"{row.get('data_source')!r}). Re-run ingest_historical.py --push "
            f"for this team-season to populate these columns."
        )
    ak  = int(ak_raw)
    gap = int(gap_raw)
    # SPD: read from DB if populated (future: SB-based), otherwise default 50.
    # BIQ: read from DB if populated, otherwise proxy from EYE (plate discipline
    #      correlates with situational awareness on the bases).
    spd = int(row.get("spd") or 0) or 50
    biq_raw = row.get("biq")
    biq = int(biq_raw) if biq_raw is not None else max(20, min(90, int(eye * 0.85 + 7)))

    return {
        "player_id":    row.get("player_id") or str(row.get("id", "unknown")),
        "name":         row.get("player_name", "Unknown"),
        "bats":         row.get("bats")   or "R",
        "throws":       row.get("throws") or "R",
        "primary_role": "Hitter",
        "field_pos":    row.get("field_pos"),   # granular Lahman position; None preserved here
        "season":       row.get("season_year"),
        "team_id":      row.get("team", ""),
        "traits": {
            "CON": con,
            "EYE": eye,
            "AK":  ak,
            "POW": pow_,
            "GAP": gap,
            "SPD": spd,
            "BIQ": biq,
        },
        "_pa_sort":  pa if pa > 0 else (con + eye + pow_),  # real PA; proxy if missing
        "_obp_sort": eye * 2 + con,                          # OBP proxy for lineup order
    }


def _pitcher_card_from_row(row: dict) -> dict:
    """
    Map a Supabase players row to a pa_wrapper-compatible pitcher card.

    pitcher_role is now always populated by ingest_historical.py.
    ip is a real DB column; falls back to trait-based proxy for legacy rows.
    STA is derived from pitcher_role (not stored) — reserved for the Inning Engine.
    """
    pitcher_role = row.get("pitcher_role") or "SP"   # always set post-PRD05
    stf = int(row.get("stuff")    or 50)
    ctl = int(row.get("control")  or 50)
    cmd = int(row.get("movement") or 50)
    ip  = float(row.get("ip") or 0.0)
    return {
        "player_id":    row.get("player_id") or str(row.get("id", "unknown")),
        "name":         row.get("player_name", "Unknown"),
        "bats":         row.get("bats")   or "R",
        "throws":       row.get("throws") or "R",
        "primary_role": "Pitcher",
        "pitcher_role": pitcher_role,
        "season":       row.get("season_year"),
        "team_id":      row.get("team", ""),
        "traits": {
            "STF": stf,
            "CTL": ctl,
            "CMD": cmd,
            "STA": _STA_DEFAULTS.get(pitcher_role, 60),
        },
        "_ip_sort": ip,    # real IP from DB; 0.0 only if column is null (ingestion bug)
    }


# ── Pilot JSON card wrappers ────────────────────────────────────────────────────

def _batter_card_from_pilot(card: dict) -> dict:
    """
    Wrap a pilot JSON hitter card with sort metadata.
    Pilot cards already have engine-ready traits (CON, EYE, AK, POW, GAP).
    SPD and BIQ are injected if absent: SPD defaults to 50, BIQ proxied from EYE.
    Sort proxies are derived from normalized_rates when available.
    """
    rates    = card.get("normalized_rates") or {}
    babip_p  = float(rates.get("babip_plus") or 1.0)
    bb_p     = float(rates.get("bb_plus")    or 1.0)

    # Inject SPD/BIQ into traits if not already present
    t   = dict(card.get("traits") or {})
    eye = t.get("EYE", 50)
    if "SPD" not in t:
        t["SPD"] = 50
    if "BIQ" not in t:
        t["BIQ"] = max(20, min(90, int(eye * 0.85 + 7)))

    return {
        **card,
        "traits":    t,
        "_pa_sort":  int(babip_p * 50 + bb_p * 50),
        "_obp_sort": int(bb_p * 100 + babip_p * 50),
    }


def _pitcher_card_from_pilot(card: dict) -> dict:
    """
    Wrap a pilot JSON pitcher card with sort metadata.
    Pilot cards already have engine-ready traits (STF, CTL, CMD, STA).
    """
    t = card.get("traits") or {}
    return {
        **card,
        "_ip_sort": (
            t.get("STF", 50) + t.get("CTL", 50) + t.get("CMD", 50) +
            (20 if card.get("pitcher_role") == "SP" else 0)
        ),
    }


# ── Roster assembly helpers ─────────────────────────────────────────────────────

def _clean(card: dict) -> dict:
    """Strip internal sort keys before hand-off to game_engine."""
    return {k: v for k, v in card.items() if not k.startswith("_")}


def _build_defensive_alignment(starters: list, bench: list | None = None) -> dict[str, str]:
    """
    Build a real {position: player_name} map using granular Lahman field_pos.

    Infield / C: first player encountered at each slot wins (starters take
    priority over bench since starters are already ranked by PA descending).

    Outfield assignment:
      1. If field_pos is a specific slot ('CF', 'LF', 'RF'), assign directly.
      2. For generic 'OF' (Lahman's standard across most eras): assign by
         POW ascending — lowest-power OF → CF, corner OFs get LF/RF.
         CF players are typically contact/speed hitters (lower POW) while
         corner OFs are power bats (e.g. Ruth POW=99 → RF, Combs POW=55 → CF).
         Crucially, starters are sorted before bench players so that backup OFs
         (who may have lower POW than starters) don't steal starting slots.
    """
    alignment: dict[str, str] = {}
    bench = bench or []

    _OF_SLOTS = ("CF", "LF", "RF")
    _IF_SLOTS = ("C", "1B", "2B", "3B", "SS")

    # Separate starters and bench OFs to ensure starters fill slots first
    starter_of_generic: list[dict] = []
    bench_of_generic:   list[dict] = []

    for player, is_starter in [(p, True) for p in starters] + [(p, False) for p in bench]:
        fp   = player.get("field_pos")
        name = player.get("name", "")
        if not fp or not name or fp == "UTIL":
            continue
        if fp in _OF_SLOTS:
            if fp not in alignment:
                alignment[fp] = name
        elif fp == "OF":
            (starter_of_generic if is_starter else bench_of_generic).append(player)
        elif fp in _IF_SLOTS:
            if fp not in alignment:
                alignment[fp] = name

    # Assign generic OF: starters first (sorted by POW asc), bench fills gaps
    for of_pool in (starter_of_generic, bench_of_generic):
        of_sorted = sorted(of_pool, key=lambda p: p.get("traits", {}).get("POW", 50))
        for pos in _OF_SLOTS:
            if pos not in alignment and of_sorted:
                alignment[pos] = of_sorted.pop(0).get("name", "")

    return alignment


# ── Lineup ──────────────────────────────────────────────────────────────────────

@dataclass
class Lineup:
    """
    Formal lineup card: 9-man batting order with defensive assignments.

    batting_order       9 player cards, each with a 'slot' key (1–9) and a
                        non-null 'field_pos' key (fallback: "UTIL").
    defensive_positions {position: player_name} covering all 9 defensive slots.
                        Fed directly to game_engine's defensive_alignment and
                        the narrative engine's zone router.
    bench               Remaining position players not in the starting lineup.
    """
    batting_order:        list   # 9 player cards with 'slot' assigned
    defensive_positions:  dict   # {pos: player_name}, all 9 slots filled
    bench:                list   # remaining hitters (field_pos fallback applied)

    def validate(self) -> list[str]:
        """
        Returns a list of validation error strings. Empty list = valid card.

        Checks:
          1. Exactly 9 players in batting_order.
          2. Slot numbers 1–9 all present, each used exactly once.
          3. Every player has a non-null field_pos (fallback enforced upstream).
          4. All 9 defensive positions are filled with non-empty names.
        """
        errors: list[str] = []

        if len(self.batting_order) != 9:
            errors.append(
                f"batting_order has {len(self.batting_order)} players; expected 9."
            )

        slots = [p.get("slot") for p in self.batting_order]
        missing_slots = set(range(1, 10)) - set(slots)
        dupe_slots    = {s for s in slots if slots.count(s) > 1}
        if missing_slots:
            errors.append(f"Missing batting slots: {sorted(missing_slots)}.")
        if dupe_slots:
            errors.append(f"Duplicate batting slot(s): {sorted(dupe_slots)}.")

        for i, player in enumerate(self.batting_order, 1):
            if not player.get("field_pos"):
                errors.append(
                    f"Slot {i} ({player.get('name', '?')}): field_pos is null — "
                    f"position fallback was not applied."
                )

        for pos in _ALL_DEF_POSITIONS:
            val = self.defensive_positions.get(pos)
            if not val:
                errors.append(f"Defensive position '{pos}' is unassigned.")

        return errors


# ── PitchingStaff ────────────────────────────────────────────────────────────────

@dataclass
class PitchingStaff:
    """
    Formal pitching staff: active rotation + bullpen depth chart.

    rotation    SP-eligible cards in priority order (index 0 = ace / #1 starter).
                Length is capped at rotation_size (default 5) during assembly.
    bullpen     RP-role cards in priority order (index 0 = first reliever out).
                Overflow SPs (beyond rotation_size) are prepended here.
    """
    rotation: list   # SP cards, IP-ranked, capped at rotation_size
    bullpen:  list   # RP cards + overflow SPs, IP-ranked

    def get_starter(self, rotation_index: int = 0) -> dict:
        """
        Return the starting pitcher for game N.
        rotation_index is taken modulo len(rotation) so it wraps cleanly
        regardless of how many games have been played.
        """
        if not self.rotation:
            raise ValueError("PitchingStaff has no rotation members.")
        return self.rotation[rotation_index % len(self.rotation)]

    def get_bullpen(self, rotation_index: int = 0) -> list:
        """
        Return the full in-game bullpen queue for game N:
          - remaining rotation SPs (other than today's starter), in order
          - all RP-role relievers, in depth order

        Pass this directly to simulate_game() as 'bullpen'.
        The game engine pops from the front when it pulls the starter.
        """
        starter_id     = self.get_starter(rotation_index).get("player_id")
        remaining_sps  = [
            p for p in self.rotation if p.get("player_id") != starter_id
        ]
        return remaining_sps + self.bullpen


# ── Roster assembly ─────────────────────────────────────────────────────────────

def _assemble_roster(
    team_id:       str,
    season_year:   int,
    hitter_cards:  list,
    pitcher_cards: list,
    use_dh:        bool,
    rotation_size: int = 5,
) -> "TeamRoster":
    """
    Sort raw player cards into a TeamRoster (Lineup + PitchingStaff).

    Lineup logic (Manager AI):
        1. Sort all hitters by _pa_sort desc → most-played starters rise first.
        2. Top 9 become the starting lineup pool; remainder → bench.
        3. Re-sort those 9 by _obp_sort desc → semi-logical 1–9 batting order.
        4. Assign slot numbers 1–9.
        5. Enforce field_pos fallback: any card with field_pos=None gets "UTIL".

    Staff logic:
        1. Sort all pitchers by _ip_sort desc.
        2. SP-role pitchers: top rotation_size → rotation; remainder → overflow bullpen.
        3. RP-role pitchers → bullpen (after overflow SPs).
        4. If no SP-role pitchers exist, the top-ranked pitcher becomes the lone starter.

    DH toggle:
        use_dh=False → the 9th lineup slot is replaced by the #1 starter's card.
        The bumped hitter moves to the bench.
    """
    if not pitcher_cards:
        raise ValueError(
            f"Team '{team_id}' {season_year}: no pitcher cards — cannot build roster."
        )

    # ── PitchingStaff ───────────────────────────────────────────────────────────
    pitchers_ranked = sorted(pitcher_cards, key=lambda c: -c["_ip_sort"])
    sp_pool = [p for p in pitchers_ranked if p.get("pitcher_role") == "SP"]
    rp_pool = [p for p in pitchers_ranked if p.get("pitcher_role") != "SP"]

    if sp_pool:
        rotation     = [_clean(p) for p in sp_pool[:rotation_size]]
        overflow_sps = [_clean(p) for p in sp_pool[rotation_size:]]
    else:
        # No SP-role pitchers — treat the highest-ranked pitcher as the sole starter
        rotation     = [_clean(pitchers_ranked[0])]
        overflow_sps = []
        rp_pool      = pitchers_ranked[1:]

    staff = PitchingStaff(
        rotation = rotation,
        bullpen  = overflow_sps + [_clean(p) for p in rp_pool],
    )

    # ── Lineup ──────────────────────────────────────────────────────────────────
    hitters_by_pa  = sorted(hitter_cards, key=lambda c: -c["_pa_sort"])
    starters_pool  = hitters_by_pa[:9]
    bench_raw      = list(hitters_by_pa[9:])
    lineup_ordered = sorted(starters_pool, key=lambda c: -c["_obp_sort"])

    # DH toggle: swap pitcher into #9 slot
    if not use_dh:
        starter_card = staff.get_starter(0)
        if len(lineup_ordered) >= 9:
            bench_raw.insert(0, lineup_ordered[8])   # bump 9th hitter to bench
            lineup_ordered = lineup_ordered[:8] + [starter_card]
        else:
            lineup_ordered.append(starter_card)

    # Assign slot numbers + enforce field_pos fallback for every batting-order card
    batting_order: list[dict] = []
    for slot_num, raw_card in enumerate(lineup_ordered, 1):
        card = _clean(raw_card)
        if not card.get("field_pos"):
            card = {**card, "field_pos": "UTIL"}
        card = {**card, "slot": slot_num}
        batting_order.append(card)

    # Apply field_pos fallback to bench cards too
    bench: list[dict] = []
    for raw_card in bench_raw:
        card = _clean(raw_card)
        if not card.get("field_pos"):
            card = {**card, "field_pos": "UTIL"}
        bench.append(card)

    # ── Defensive alignment ──────────────────────────────────────────────────────
    # Build from batting order + bench.  Add the P slot from the #1 starter so
    # the ground_pitcher zone always resolves to a name.
    def_alignment = _build_defensive_alignment(batting_order, bench)
    if "P" not in def_alignment:
        def_alignment["P"] = staff.get_starter(0).get("name", "")

    lineup_card = Lineup(
        batting_order       = batting_order,
        defensive_positions = def_alignment,
        bench               = bench,
    )

    return TeamRoster(
        team_id       = team_id,
        season_year   = season_year,
        lineup_card   = lineup_card,
        staff         = staff,
        use_dh        = use_dh,
        _hitter_pool  = list(hitter_cards),  # raw cards with sort keys preserved
    )


# ── TeamRoster ──────────────────────────────────────────────────────────────────

@dataclass
class TeamRoster:
    """
    Complete game-ready team object.

    Container for one Lineup and one PitchingStaff.  Backward-compatible
    properties (.lineup, .pitcher, .bullpen, .bench) delegate to the inner
    objects so existing call sites outside this module continue to work.
    """
    team_id:     str
    season_year: int
    lineup_card: Lineup
    staff:       PitchingStaff
    use_dh:      bool
    # Raw hitter cards (with sort keys) preserved for generate_default_lineup().
    # Not shown in repr — internal bookkeeping only.
    _hitter_pool:     list = field(default_factory=list, repr=False)
    _manual_override: bool = field(default=False,        repr=False)

    # ── Backward-compatible shims ─────────────────────────────────────────────
    # Code outside this module that references .lineup / .pitcher / .bullpen /
    # .bench continues to work without modification.

    @property
    def lineup(self) -> list:
        """Batting order (9 cards, each with 'slot' and non-null 'field_pos')."""
        return self.lineup_card.batting_order

    @property
    def pitcher(self) -> dict:
        """#1 starter card (game 1 default)."""
        return self.staff.get_starter(0)

    @property
    def bullpen(self) -> list:
        """Full in-game bullpen queue for game 1 (game-1 starter excluded)."""
        return self.staff.get_bullpen(0)

    @property
    def bench(self) -> list:
        """Hitters not in the starting lineup."""
        return self.lineup_card.bench

    # ── Game Engine interface ─────────────────────────────────────────────────

    def for_game_engine(self, rotation_index: int = 0) -> dict:
        """
        Return a dict ready to pass directly to simulate_game().

        rotation_index cycles through the starting rotation:
            0 → #1 starter (ace / highest IP)
            1 → #2 starter
            … wraps via modulo so any integer is safe.

        Uses the current lineup_card regardless of whether a manual override
        is active — call generate_default_lineup() first to revert to defaults.

        The defensive_alignment P slot is patched to reflect today's actual
        starter so the narrative engine always resolves ground_pitcher correctly.

        Outfield ARM defaults to 55; set TeamRoster.arm to override once
        fielding data is ingested.
        """
        starter   = self.staff.get_starter(rotation_index)
        bullpen   = self.staff.get_bullpen(rotation_index)
        arm       = getattr(self, "arm", None) or 55

        # Patch P slot for this game's starter (Lineup stores game-1 default)
        def_align = {**self.lineup_card.defensive_positions}
        def_align["P"] = starter.get("name", def_align.get("P", ""))

        return {
            "team_id":             self.team_id,
            "lineup":              self.lineup_card.batting_order,
            "pitcher":             starter,
            "bullpen":             bullpen,
            "arm":                 arm,
            "defensive_alignment": def_align,
        }

    # ── Manual overrides ──────────────────────────────────────────────────────

    def set_batting_order(self, player_names: list[str]) -> None:
        """
        Reorder the batting order to match player_names (exactly 9 names).

        All available players (current lineup + bench) are pooled.  The 9
        named players become the new batting order (slots 1–9 in list order).
        Everyone not listed moves to the bench.  The defensive alignment is
        rebuilt automatically from the new order.

        Sets _manual_override = True so generate_default_lineup() can detect
        and revert it later.

        Raises:
            ValueError if len(player_names) != 9, or if any name is not found
            in the current player pool.
        """
        if len(player_names) != 9:
            raise ValueError(
                f"set_batting_order requires exactly 9 names; got {len(player_names)}."
            )

        # Pool = everyone currently available (lineup + bench), keyed by name.
        pool: dict[str, dict] = {
            p["name"]: p
            for p in self.lineup_card.batting_order + self.lineup_card.bench
        }
        unknown = [n for n in player_names if n not in pool]
        if unknown:
            raise ValueError(
                f"Unknown player(s): {unknown}.\n"
                f"Available: {sorted(pool.keys())}"
            )

        # Build new batting order with updated slot numbers.
        new_order: list[dict] = []
        for slot, name in enumerate(player_names, 1):
            card = {**pool[name], "slot": slot}
            new_order.append(card)

        # Everyone not in the new order moves to the bench (no slot key).
        new_bench: list[dict] = [
            {k: v for k, v in p.items() if k != "slot"}
            for name, p in pool.items()
            if name not in player_names
        ]

        # Rebuild defensive alignment from the new order + remaining bench.
        new_align = _build_defensive_alignment(new_order, new_bench)
        if "P" not in new_align:
            new_align["P"] = self.staff.get_starter(0).get("name", "")

        self.lineup_card.batting_order       = new_order
        self.lineup_card.bench               = new_bench
        self.lineup_card.defensive_positions = new_align
        self._manual_override                = True

    def set_defensive_alignment(self, alignment: dict[str, str]) -> None:
        """
        Explicitly override the {Position: Name} defensive map.

        Validates that every assigned name is actually on the active roster:
          - Non-'P' positions must map to a player in the batting order.
          - 'P' must map to a pitcher on the staff (rotation or bullpen).

        The override is a full replacement — pass the complete 9-slot dict.
        To update a single position, read the current dict first:

            align = dict(roster.lineup_card.defensive_positions)
            align["RF"] = "Babe Ruth"
            roster.set_defensive_alignment(align)

        Raises:
            ValueError if any name fails validation.
        """
        lineup_names  = {p["name"] for p in self.lineup_card.batting_order}
        pitcher_names = {
            p["name"] for p in self.staff.rotation + self.staff.bullpen
        }

        errors: list[str] = []
        for pos, name in alignment.items():
            if pos == "P":
                if name not in pitcher_names:
                    errors.append(f"'P' → '{name}' is not in the pitching staff.")
            else:
                if name not in lineup_names:
                    errors.append(
                        f"'{pos}' → '{name}' is not in the batting lineup."
                    )
        if errors:
            raise ValueError(
                "set_defensive_alignment validation failed:\n  "
                + "\n  ".join(errors)
            )

        self.lineup_card.defensive_positions = dict(alignment)

    def generate_default_lineup(self) -> None:
        """
        Factory reset: rebuild the lineup from the original PA/OBP heuristics.

        Reverts any manual batting order or defensive alignment override.
        Respects the current value of self.use_dh.

        Raises:
            RuntimeError if _hitter_pool is empty (loaded without the internal
            pool — this should never happen for rosters built by load_team or
            load_team_from_pilot).
        """
        if not self._hitter_pool:
            raise RuntimeError(
                "generate_default_lineup() requires _hitter_pool to be populated. "
                "Load the team via load_team() or load_team_from_pilot()."
            )

        hitters_by_pa  = sorted(self._hitter_pool, key=lambda c: -c["_pa_sort"])
        starters_pool  = hitters_by_pa[:9]
        bench_raw      = list(hitters_by_pa[9:])
        lineup_ordered = sorted(starters_pool, key=lambda c: -c["_obp_sort"])

        if not self.use_dh:
            starter_card = self.staff.get_starter(0)
            if len(lineup_ordered) >= 9:
                bench_raw.insert(0, lineup_ordered[8])
                lineup_ordered = lineup_ordered[:8] + [starter_card]
            else:
                lineup_ordered.append(starter_card)

        batting_order: list[dict] = []
        for slot_num, raw_card in enumerate(lineup_ordered, 1):
            card = _clean(raw_card)
            if not card.get("field_pos"):
                card = {**card, "field_pos": "UTIL"}
            card = {**card, "slot": slot_num}
            batting_order.append(card)

        bench: list[dict] = []
        for raw_card in bench_raw:
            card = _clean(raw_card)
            if not card.get("field_pos"):
                card = {**card, "field_pos": "UTIL"}
            bench.append(card)

        def_alignment = _build_defensive_alignment(batting_order, bench)
        if "P" not in def_alignment:
            def_alignment["P"] = self.staff.get_starter(0).get("name", "")

        self.lineup_card      = Lineup(
            batting_order       = batting_order,
            defensive_positions = def_alignment,
            bench               = bench,
        )
        self._manual_override = False

    def toggle_dh(self, use_dh: bool) -> None:
        """
        Toggle the DH rule on or off.

        No manual override active → calls generate_default_lineup() to rebuild
        cleanly with the new DH setting.

        Manual override active → preserves the current 1–8 slot order and
        handles only slot 9:
          use_dh=False  pitcher inserted into slot 9; old #9 hitter → bench.
          use_dh=True   pitcher removed from slot 9 (if present); first bench
                        hitter promoted to slot 9.
        """
        if self.use_dh == use_dh:
            return  # nothing to do

        self.use_dh = use_dh

        if not self._manual_override:
            self.generate_default_lineup()
            return

        current = list(self.lineup_card.batting_order)
        bench   = list(self.lineup_card.bench)

        if not use_dh:
            # Remove pitcher if somehow already in batting order, then add at #9.
            current = [p for p in current if p.get("primary_role") != "Pitcher"]
            if len(current) >= 9:
                bumped = {k: v for k, v in current[8].items() if k != "slot"}
                bench.insert(0, bumped)
                current = current[:8]
            current.append({**self.staff.get_starter(0), "slot": 9})
        else:
            # Remove pitcher from the batting order (if present in no-DH slot).
            current = [p for p in current if p.get("primary_role") != "Pitcher"]
            # Promote the first available bench hitter to slot 9.
            next_hitter = next(
                (p for p in bench if p.get("primary_role") != "Pitcher"), None
            )
            if next_hitter and len(current) < 9:
                bench   = [p for p in bench if p["name"] != next_hitter["name"]]
                current.append({**next_hitter, "slot": 9})

        # Re-index slots to guarantee no gaps.
        for i, card in enumerate(current, 1):
            card["slot"] = i

        self.lineup_card.batting_order = current
        self.lineup_card.bench         = bench

    # ── Display ───────────────────────────────────────────────────────────────

    def print_roster(self) -> None:
        """Print a formatted roster card to stdout."""
        dh_tag = "(DH)" if self.use_dh else "(No DH — pitcher bats 9th)"
        rot_tag = (
            f"  |  {len(self.staff.rotation)}-man rotation"
            if self.staff.rotation else ""
        )
        print(f"\n{'═' * 62}")
        print(f"  {self.season_year} {self.team_id}   {dh_tag}{rot_tag}")
        print(f"{'─' * 62}")
        print(f"  {'#':<3} {'Name':<22} {'Pos':<5} CON  EYE   AK  POW  GAP")
        print(f"{'─' * 62}")
        for b in self.lineup_card.batting_order:
            slot = b.get("slot", "?")
            t    = b.get("traits", {})
            pos  = b.get("field_pos") or "UTIL"
            if b.get("primary_role") == "Pitcher":
                print(
                    f"  {slot:<3} {b['name']:<22} {pos:<5}"
                    f"STF:{t.get('STF', '--'):>3}  CTL:{t.get('CTL', '--'):>3}"
                    f"  CMD:{t.get('CMD', '--'):>3}  ← P"
                )
            else:
                print(
                    f"  {slot:<3} {b['name']:<22} {pos:<5}"
                    f"{t.get('CON', '--'):>3}  {t.get('EYE', '--'):>3}  "
                    f"{t.get('AK', '--'):>3}  {t.get('POW', '--'):>3}"
                    f"  {t.get('GAP', '--'):>3}"
                )
        if len(self.lineup_card.batting_order) < 9:
            for i in range(len(self.lineup_card.batting_order) + 1, 10):
                print(f"  {i:<3} {'— (incomplete)':22}")

        print(f"{'─' * 62}")
        print(f"  Rotation ({len(self.staff.rotation)}-man):")
        for i, sp in enumerate(self.staff.rotation, 1):
            t = sp.get("traits", {})
            print(
                f"    {i}. {sp['name']:<22}"
                f"STF:{t.get('STF', '--'):>3}  CTL:{t.get('CTL', '--'):>3}"
                f"  CMD:{t.get('CMD', '--'):>3}  STA:{t.get('STA', '--'):>3}"
            )
        if self.staff.bullpen:
            bp_names = ", ".join(p["name"] for p in self.staff.bullpen)
            print(f"  BP  {bp_names}")
        if self.lineup_card.bench:
            bn_names = ", ".join(b["name"] for b in self.lineup_card.bench)
            print(f"  BN  {bn_names}")

        # Run validate and surface any warnings
        errors = self.lineup_card.validate()
        if errors:
            print(f"{'─' * 62}")
            print(f"  ⚠  Lineup validation warnings:")
            for err in errors:
                print(f"     • {err}")

        print(f"{'═' * 62}\n")


# ── Public loaders ──────────────────────────────────────────────────────────────

def load_team(
    season_year:   int,
    team_name:     str,
    use_dh:        bool = True,
    rotation_size: int  = 5,
) -> TeamRoster:
    """
    Load a team roster from the Supabase `players` table.

    Args:
        season_year   : Season to load (e.g., 1927, 2000).
        team_name     : Team abbreviation ("NYY", "BOS") or Lahman ID ("NYA").
                        Common abbreviations are resolved via _TEAM_LAHMAN_IDS.
        use_dh        : If False, the 9th lineup slot is filled by the starter.
        rotation_size : Number of SP-role pitchers to place in the active rotation
                        (4-man or 5-man). Defaults to 5. Overflow SPs move to
                        the bullpen queue ahead of RP-role pitchers.

    Returns:
        TeamRoster — call .for_game_engine(rotation_index) to get the dict
        simulate_game() expects.

    Raises:
        ValueError if no rows are found, or if no pitcher can be identified.

    Warns:
        UserWarning if fewer than 9 hitter cards are found (incomplete DB data).
    """
    lahman_ids = _TEAM_LAHMAN_IDS.get(team_name.upper(), [team_name.upper()])
    response   = (
        supabase.table("players")
        .select("*")
        .eq("season_year", season_year)
        .in_("team", lahman_ids)
        .eq("data_source", "historical")
        .execute()
    )
    rows = response.data or []

    if not rows:
        raise ValueError(
            f"No historical rows found for season={season_year}, team='{team_name}' "
            f"(Lahman IDs searched: {lahman_ids}). "
            f"Run: python ingest_historical.py --season {season_year} "
            f"--team {lahman_ids[0]} --push"
        )

    hitter_cards  = [_batter_card_from_row(r)  for r in rows if _is_hitter_row(r)]
    pitcher_cards = [_pitcher_card_from_row(r) for r in rows if _is_pitcher_row(r)]

    if len(hitter_cards) < 9:
        warnings.warn(
            f"load_team({season_year}, '{team_name}'): only {len(hitter_cards)} hitter card(s) "
            f"found in DB — need 9 for a full lineup. "
            f"DB may be incomplete for this team-season. "
            f"Consider using load_team_from_pilot() if a pilot JSON exists.",
            UserWarning,
            stacklevel=2,
        )

    return _assemble_roster(
        team_id       = team_name.upper(),
        season_year   = season_year,
        hitter_cards  = hitter_cards,
        pitcher_cards = pitcher_cards,
        use_dh        = use_dh,
        rotation_size = rotation_size,
    )


def load_team_from_pilot(
    json_path:     str,
    team_id:       str | None = None,
    use_dh:        bool = True,
    rotation_size: int  = 5,
) -> TeamRoster:
    """
    Load a team roster from a local pilot JSON file (e.g., pilot_1927_nya.json).

    Pilot JSON files are produced by ratings.py / ingestion.py and contain full
    engine-ready player cards with "primary_role", "traits" (CON/EYE/AK/POW/GAP),
    "normalized_rates", etc. No trait mapping is needed.

    Args:
        json_path     : Path to the pilot JSON file.
        team_id       : Display label for the team. Defaults to the filename stem.
        use_dh        : If False, the 9th lineup slot is filled by the starter.
        rotation_size : Number of SPs to place in the active rotation (4 or 5).

    Returns:
        TeamRoster — call .for_game_engine(rotation_index) to get the dict
        simulate_game() expects.

    Raises:
        ValueError if no pitcher cards are found.
        FileNotFoundError if the JSON path does not exist.
    """
    with open(json_path) as f:
        cards = json.load(f)

    if team_id is None:
        team_id = os.path.splitext(os.path.basename(json_path))[0]

    season_year = int(cards[0].get("season", 0)) if cards else 0

    hitter_cards  = [_batter_card_from_pilot(c)  for c in cards if c.get("primary_role") == "Hitter"]
    pitcher_cards = [_pitcher_card_from_pilot(c) for c in cards if c.get("primary_role") == "Pitcher"]

    if len(hitter_cards) < 9:
        warnings.warn(
            f"load_team_from_pilot('{json_path}'): only {len(hitter_cards)} hitter card(s) found.",
            UserWarning,
            stacklevel=2,
        )

    return _assemble_roster(
        team_id       = team_id,
        season_year   = season_year,
        hitter_cards  = hitter_cards,
        pitcher_cards = pitcher_cards,
        use_dh        = use_dh,
        rotation_size = rotation_size,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _show_order(roster: "TeamRoster", label: str) -> None:
    """Print a compact one-line batting order for quick inspection."""
    slots = [(p["slot"], p["name"]) for p in roster.lineup_card.batting_order]
    slots.sort()
    order = "  ".join(f"{s}. {n}" for s, n in slots)
    flag  = " [MANUAL]" if roster._manual_override else " [DEFAULT]"
    print(f"  {label}{flag}")
    print(f"    {order}")
    print(f"    bench: {[p['name'] for p in roster.lineup_card.bench]}")
    print()


# ── Verification test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Test 1: Default lineup — 1927 NYA from Supabase ───────────────────────
    print("=" * 62)
    print("  TEST 1 — Default lineup (1927 NYA, Supabase)")
    print("=" * 62)
    try:
        nya27 = load_team(1927, "NYA")
        _show_order(nya27, "Default (PA / OBP sort)")
    except ValueError as e:
        print(f"  ERROR: {e}\n")

    # ── Test 2: set_batting_order — move Ruth to slot 1 ───────────────────────
    # Demonstrates: explicit order override, bench update, _manual_override flag
    print("=" * 62)
    print("  TEST 2 — set_batting_order() (1927 NYA)")
    print("=" * 62)
    try:
        nya27 = load_team(1927, "NYA")
        _show_order(nya27, "Before override")

        # Move Ruth to leadoff, bump whoever was #1 down to #3
        default_names = [p["name"] for p in nya27.lineup_card.batting_order]
        ruth_idx = default_names.index("Babe Ruth")
        # Build new order: Ruth first, everyone else keeps relative position
        new_order = ["Babe Ruth"] + [n for n in default_names if n != "Babe Ruth"]
        nya27.set_batting_order(new_order)
        _show_order(nya27, "After: Ruth → slot 1")

        # Swap two more players
        current = [p["name"] for p in nya27.lineup_card.batting_order]
        current[2], current[5] = current[5], current[2]   # flip slots 3 and 6
        nya27.set_batting_order(current)
        _show_order(nya27, "After: slots 3 & 6 swapped")
    except (ValueError, RuntimeError) as e:
        print(f"  ERROR: {e}\n")

    # ── Test 3: generate_default_lineup() — factory reset ─────────────────────
    print("=" * 62)
    print("  TEST 3 — generate_default_lineup() factory reset (1927 NYA)")
    print("=" * 62)
    try:
        nya27 = load_team(1927, "NYA")
        nya27.set_batting_order(
            [p["name"] for p in reversed(nya27.lineup_card.batting_order)]
        )
        _show_order(nya27, "After: reversed order")
        nya27.generate_default_lineup()
        _show_order(nya27, "After: factory reset")
    except (ValueError, RuntimeError) as e:
        print(f"  ERROR: {e}\n")

    # ── Test 4: set_defensive_alignment() ─────────────────────────────────────
    print("=" * 62)
    print("  TEST 4 — set_defensive_alignment() (1927 NYA)")
    print("=" * 62)
    try:
        nya27 = load_team(1927, "NYA")
        current_align = dict(nya27.lineup_card.defensive_positions)
        print(f"  Default: {current_align}")

        # Move Ruth to CF (instead of RF) and Combs to RF
        current_align["CF"] = "Babe Ruth"
        current_align["RF"] = "Earle Combs"
        nya27.set_defensive_alignment(current_align)
        print(f"  Override: {nya27.lineup_card.defensive_positions}")

        # Confirm validate() still passes
        errs = nya27.lineup_card.validate()
        print(f"  validate(): {errs if errs else 'clean ✓'}")

        # Attempt an invalid override (name not in lineup)
        bad = dict(nya27.lineup_card.defensive_positions)
        bad["SS"] = "Pedro Martinez"
        try:
            nya27.set_defensive_alignment(bad)
        except ValueError as ve:
            print(f"  Correctly rejected invalid override: {ve}")
    except (ValueError, RuntimeError) as e:
        print(f"  ERROR: {e}\n")

    # ── Test 5: toggle_dh() — preserves manual order ──────────────────────────
    print("=" * 62)
    print("  TEST 5 — toggle_dh() with manual override active (1927 NYA)")
    print("=" * 62)
    try:
        nya27 = load_team(1927, "NYA")
        # Set a manual order first
        names = [p["name"] for p in nya27.lineup_card.batting_order]
        names[0], names[3] = names[3], names[0]   # swap slots 1 & 4
        nya27.set_batting_order(names)
        _show_order(nya27, "Manual order (DH on)")

        # Toggle DH off — pitcher should appear in slot 9, old #9 → bench
        nya27.toggle_dh(use_dh=False)
        _show_order(nya27, "After toggle_dh(False) — pitcher in slot 9")

        # Toggle DH back on — pitcher removed, bench hitter returns to slot 9
        nya27.toggle_dh(use_dh=True)
        _show_order(nya27, "After toggle_dh(True) — pitcher removed")

        print(f"  Manual override still active: {nya27._manual_override}")
    except (ValueError, RuntimeError) as e:
        print(f"  ERROR: {e}\n")

    # ── Test 6: for_game_engine() rotation cycling ────────────────────────────
    print("=" * 62)
    print("  TEST 6 — for_game_engine() rotation + P slot contract (2000 BOS)")
    print("=" * 62)
    try:
        bos00 = load_team(2000, "BOS")
        for idx in range(3):
            gd = bos00.for_game_engine(rotation_index=idx)
            p_slot = gd["defensive_alignment"].get("P")
            assert p_slot == gd["pitcher"]["name"], "P slot mismatch"
            print(
                f"    game {idx+1}: starter={gd['pitcher']['name']:<22} "
                f"P slot={p_slot} ✓"
            )
    except (ValueError, AssertionError) as e:
        print(f"  ERROR: {e}\n")
