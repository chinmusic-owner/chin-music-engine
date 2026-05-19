"""
roster_manager.py — PRD 05: The Roster Loader

Builds game-ready TeamRoster objects from two sources:
    1. Supabase `players` table (load_team)
    2. Local pilot JSON files produced by ratings.py (load_team_from_pilot)

Public API:
    load_team(season_year, team_name, use_dh=True) -> TeamRoster
    load_team_from_pilot(json_path, team_id=None, use_dh=True) -> TeamRoster

    TeamRoster.for_game_engine() -> dict   # plug directly into simulate_game()

Schema notes (Supabase `players` table as of PRD 05):
    Hitter traits   : contact, power, eye, speed
    Pitcher traits  : stuff, control, movement
    Role fields     : pitcher_role ('SP'|'RP'|null), primary_position ('C'|'1B'|'IF'|'OF'|'UTIL'|null)
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


# ── Roster assembly (shared) ────────────────────────────────────────────────────

def _clean(card: dict) -> dict:
    """Strip internal sort keys before hand-off to game_engine."""
    return {k: v for k, v in card.items() if not k.startswith("_")}


def _assemble_roster(
    team_id: str,
    season_year: int,
    hitter_cards: list,
    pitcher_cards: list,
    use_dh: bool,
) -> "TeamRoster":
    """
    Sort and assemble a TeamRoster from pre-built batter and pitcher cards.

    Lineup logic (Manager AI):
        1. Sort all hitters by _pa_sort desc → "most frequent starters" rise to the top.
        2. Take top 9 as the starting lineup pool; remainder → bench.
        3. Re-sort those 9 by _obp_sort desc → semi-logical 1–9 batting order
           (high-OBP hitters bat earlier).

    Staff logic:
        1. Sort pitchers by _ip_sort desc (trait quality + SP bonus).
        2. Starter = pitchers[0] (highest overall workload proxy).
        3. Bullpen = pitchers[1:].

    DH toggle:
        use_dh=False → the 9th lineup slot is replaced by the starting pitcher's card.
        The bumped hitter moves to the bench.
    """
    if not pitcher_cards:
        raise ValueError(
            f"Team '{team_id}' {season_year}: no pitcher cards — cannot build roster."
        )

    # ── Staff ──────────────────────────────────────────────────────────────────
    pitchers_ranked = sorted(pitcher_cards, key=lambda c: -c["_ip_sort"])
    starter         = pitchers_ranked[0]
    bullpen         = pitchers_ranked[1:]

    # ── Lineup ─────────────────────────────────────────────────────────────────
    hitters_by_pa   = sorted(hitter_cards, key=lambda c: -c["_pa_sort"])
    starters_pool   = hitters_by_pa[:9]
    bench           = list(hitters_by_pa[9:])
    lineup_ordered  = sorted(starters_pool, key=lambda c: -c["_obp_sort"])

    # ── DH toggle ──────────────────────────────────────────────────────────────
    if not use_dh:
        if len(lineup_ordered) >= 9:
            bench.insert(0, lineup_ordered[8])  # bump 9th hitter to bench
            lineup_ordered = lineup_ordered[:8] + [starter]
        else:
            lineup_ordered.append(starter)      # pitcher fills empty 9th slot

    return TeamRoster(
        team_id      = team_id,
        season_year  = season_year,
        lineup       = [_clean(c) for c in lineup_ordered],
        pitcher      = _clean(starter),
        bullpen      = [_clean(c) for c in bullpen],
        bench        = [_clean(c) for c in bench],
        use_dh       = use_dh,
    )


# ── TeamRoster ──────────────────────────────────────────────────────────────────

@dataclass
class TeamRoster:
    """Complete game-ready team object."""
    team_id:     str
    season_year: int
    lineup:      list   # [≤9 batter cards] ordered 1–9 by OBP proxy
    pitcher:     dict   # starting pitcher card
    bullpen:     list   # remaining pitchers, sorted by ip_proxy desc
    bench:       list   # hitters not in starting lineup
    use_dh:      bool

    def for_game_engine(self, rotation_index: int = 0) -> dict:
        """
        Return a dict ready to pass to simulate_game().

        rotation_index cycles through the starting rotation:
            0  →  #1 starter (highest IP / Pedro)
            1  →  #2 starter
            …  wraps around via modulo

        The full pitching staff is passed as "bullpen" so the game engine can
        apply stamina limits and pull the starter mid-game.

        Rotation candidates are SP-role pitchers from the full staff
        (TeamRoster.pitcher + SP members of TeamRoster.bullpen), sorted by
        the original IP-ranked order.  RP-role pitchers always stay in the
        bullpen regardless of rotation_index.
        """
        # Build rotation: starter + any SP-role pitchers from the bullpen list
        sp_rotation = [self.pitcher] + [
            p for p in self.bullpen if p.get("pitcher_role") == "SP"
        ]
        rp_staff = [p for p in self.bullpen if p.get("pitcher_role") != "SP"]

        idx     = rotation_index % len(sp_rotation) if sp_rotation else 0
        starter = sp_rotation[idx]
        starter_id = starter.get("player_id")

        # Remaining SPs (other than starter) go first in the bullpen queue,
        # followed by RP staff (in their original ranked order)
        remaining_sps = [p for p in sp_rotation if p.get("player_id") != starter_id]
        bullpen = remaining_sps + rp_staff

        # Outfield ARM: read from roster metadata if present, otherwise default 55.
        # A higher value makes outfielders more likely to throw out advancing runners.
        # To set a real value: pass arm= when calling load_team() or set it in the
        # TeamRoster metadata once fielding data is ingested.
        arm = getattr(self, "arm", None) or 55

        return {
            "team_id": self.team_id,
            "lineup":  self.lineup,
            "pitcher": starter,
            "bullpen": bullpen,
            "arm":     arm,
        }

    def print_roster(self) -> None:
        """Print a formatted roster card to stdout."""
        dh_tag = "(DH)" if self.use_dh else "(No DH — pitcher bats 9th)"
        print(f"\n{'═' * 62}")
        print(f"  {self.season_year} {self.team_id}   {dh_tag}")
        print(f"{'─' * 62}")
        print(f"  {'#':<3} {'Name':<24}  CON  EYE   AK  POW  GAP")
        print(f"{'─' * 62}")
        for i, b in enumerate(self.lineup, 1):
            t      = b["traits"]
            if b.get("primary_role") == "Pitcher":
                # No-DH slot: pitcher has STF/CTL/CMD, not hitter traits
                print(
                    f"  {i:<3} {b['name']:<24}  "
                    f"STF:{t['STF']:>3}  CTL:{t['CTL']:>3}  CMD:{t['CMD']:>3}  ← P"
                )
            else:
                print(
                    f"  {i:<3} {b['name']:<24}  {t['CON']:>3}  {t['EYE']:>3}  "
                    f"{t.get('AK', '--'):>3}  {t['POW']:>3}  {t.get('GAP', '--'):>3}"
                )
        if len(self.lineup) < 9:
            for i in range(len(self.lineup) + 1, 10):
                print(f"  {i:<3} {'— (incomplete)':24}")
        sp = self.pitcher
        t  = sp["traits"]
        print(f"{'─' * 62}")
        print(
            f"  SP  {sp['name']:<24}  "
            f"STF:{t['STF']:>3}  CTL:{t['CTL']:>3}  CMD:{t['CMD']:>3}  STA:{t['STA']:>3}"
        )
        if self.bullpen:
            pen_names = ", ".join(p["name"] for p in self.bullpen)
            print(f"  BP  {pen_names}")
        if self.bench:
            bench_names = ", ".join(b["name"] for b in self.bench)
            print(f"  BN  {bench_names}")
        print(f"{'═' * 62}\n")


# ── Public loaders ──────────────────────────────────────────────────────────────

def load_team(
    season_year: int,
    team_name: str,
    use_dh: bool = True,
) -> TeamRoster:
    """
    Load a team roster from the Supabase `players` table.

    Args:
        season_year : Season to load (e.g., 1927, 2000).
        team_name   : Team abbreviation ("NYY", "BOS") or a partial full name.
                      Common abbreviations are resolved via _TEAM_ALIASES to
                      ILIKE search fragments (e.g. "NYY" → team ILIKE '%yankees%').
        use_dh      : If False, the 9th lineup spot is filled by the starting pitcher.

    Returns:
        TeamRoster — call .for_game_engine() to get the dict simulate_game() expects.

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
        team_id      = team_name.upper(),
        season_year  = season_year,
        hitter_cards = hitter_cards,
        pitcher_cards = pitcher_cards,
        use_dh       = use_dh,
    )


def load_team_from_pilot(
    json_path: str,
    team_id: str | None = None,
    use_dh: bool = True,
) -> TeamRoster:
    """
    Load a team roster from a local pilot JSON file (e.g., pilot_1927_nya.json).

    Pilot JSON files are produced by ratings.py / ingestion.py and contain full
    engine-ready player cards with "primary_role", "traits" (CON/EYE/AK/POW/GAP),
    "normalized_rates", etc. No trait mapping is needed.

    Args:
        json_path   : Path to the pilot JSON file.
        team_id     : Display label for the team. Defaults to the filename stem.
        use_dh      : If False, the 9th lineup spot is filled by the starting pitcher.

    Returns:
        TeamRoster — call .for_game_engine() to get the dict simulate_game() expects.

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
    )


# ── Verification test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Test 1: 1927 NYY — from pilot JSON (full 25-man roster) ────────────────
    print("=" * 62)
    print("  TEST 1 — 1927 NYY (source: pilot_1927_nya.json)")
    print("=" * 62)
    try:
        nya27 = load_team_from_pilot("pilot_1927_nya.json", team_id="1927-NYY")
        nya27.print_roster()
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERROR: {e}\n")

    # ── Test 1b: 1927 NYY no-DH (pitcher bats 9th) ────────────────────────────
    print("=" * 62)
    print("  TEST 1b — 1927 NYY, use_dh=False (source: pilot JSON)")
    print("=" * 62)
    try:
        nya27_nodh = load_team_from_pilot("pilot_1927_nya.json", team_id="1927-NYY", use_dh=False)
        nya27_nodh.print_roster()
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERROR: {e}\n")

    # ── Test 2: 2000 BOS — from Supabase (full roster post-ingestion) ─────────
    print("=" * 62)
    print("  TEST 2 — 2000 BOS (source: Supabase, data_source='historical')")
    print("=" * 62)
    try:
        bos00 = load_team(2000, "BOS")
        bos00.print_roster()
        # Spot-check Pedro's card
        sp = bos00.pitcher
        print(f"  Pedro check — pitcher_role={sp.get('pitcher_role')!r}  "
              f"throws={sp.get('throws')!r}  "
              f"STF={sp['traits']['STF']}  CTL={sp['traits']['CTL']}  CMD={sp['traits']['CMD']}")
    except ValueError as e:
        print(f"  ERROR: {e}\n")
