"""
run_mid_tier_validation.py — Simulation Equilibrium Validation Test

Runs a 1,000-game round robin using 6 genuinely average teams (historical
win% .450–.550), one from each simulation era.  Purpose: verify the engine
produces calibrated league-average outcomes without the noise introduced by
historically elite or historically terrible rosters.

Teams selected
──────────────
  1927 CIN  75-78  .490  Cincinnati Reds          (dead-ball / early power era)
  1954 PHI  75-79  .487  Philadelphia Phillies    (post-war era)
  1975 SLN  82-80  .506  St. Louis Cardinals      (NL speed era)
  1985 BOS  81-81  .500  Boston Red Sox           (AL transition era)
  2000 DET  79-83  .488  Detroit Tigers           (modern era)
  2001 TOR  80-82  .494  Toronto Blue Jays        (high-K era)

Sanity checks
─────────────
  • No team W% below .380 or above .620
  • League-average BABIP: .285–.320
  • League-average runs/game: 4.0–5.8
  • K% range: 7–22%
  • BB% range: 6–13%

Usage
─────
  python3 run_mid_tier_validation.py              # dry-ingest + 1 000 games
  python3 run_mid_tier_validation.py --n 2000     # more games
  python3 run_mid_tier_validation.py --skip-ingest  # skip DB check (teams already loaded)
  python3 run_mid_tier_validation.py --no-dh      # pitcher bats 9th
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from dataclasses import dataclass, field

# ── Engine imports ─────────────────────────────────────────────────────────────
from roster_manager import load_team
from game_engine    import simulate_game


# ══════════════════════════════════════════════════════════════════════════════
# Team definitions
# ══════════════════════════════════════════════════════════════════════════════

TARGETS: list[tuple[int, str, str, float]] = [
    # (season, lahman_team_id, display_label, historical_wp)
    (1927, "CIN", "1927 CIN", 0.490),   # Cincinnati Reds       75-78
    (1954, "PHI", "1954 PHI", 0.487),   # Philadelphia Phillies 75-79
    (1975, "SLN", "1975 SLN", 0.506),   # St. Louis Cardinals   82-80
    (1985, "BOS", "1985 BOS", 0.500),   # Boston Red Sox        81-81
    (2000, "DET", "2000 DET", 0.488),   # Detroit Tigers        79-83
    (2001, "TOR", "2001 TOR", 0.494),   # Toronto Blue Jays     80-82
]

# Hit + total-base mappings (must match game_engine outcomes)
_HITS    = {"Single", "Double", "Triple", "HR", "InfieldHit"}
_TB_VALS = {"InfieldHit": 1, "Single": 1, "Double": 2, "Triple": 3, "HR": 4}


# ══════════════════════════════════════════════════════════════════════════════
# Stats accumulator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MidTierStats:
    """Per-team batting + fielding accumulator for the validation test."""
    label:    str
    hist_wp:  float   # real-world historical win%
    games:    int = 0
    wins:     int = 0
    rs:       int = 0
    ra:       int = 0
    pa:       int = 0
    ab:       int = 0
    h:        int = 0
    hr:       int = 0
    tb:       int = 0
    bb:       int = 0
    hbp:      int = 0
    k:        int = 0
    # BABIP components (standard: H − HR / AB − K − HR)
    bip_h:    int = 0   # non-HR BIP hits (1B + 2B + 3B + InfieldHit)
    bip_ab:   int = 0   # at-bats that put ball in play (AB − K)
    bip_hr:   int = 0   # HR (excluded from BABIP denominator)

    def bat(self, ev) -> None:
        self.pa += 1
        o = ev.outcome
        is_walk = o in ("BB", "HBP")
        if not is_walk:
            self.ab += 1
        if o == "BB":
            self.bb  += 1
        elif o == "HBP":
            self.hbp += 1
        elif o == "K":
            self.k   += 1
        if o in _HITS:
            self.h  += 1
            self.tb += _TB_VALS.get(o, 1)
        if o == "HR":
            self.hr       += 1
            self.bip_hr   += 1
        # BABIP tracking: ball-in-play at-bats = AB − K (includes outs + non-HR hits)
        if not is_walk and o != "K":
            self.bip_ab += 1
            if o in _HITS and o != "HR":
                self.bip_h += 1

    def end_game(self, rs: int, ra: int, won: bool) -> None:
        self.games += 1
        self.wins  += int(won)
        self.rs    += rs
        self.ra    += ra

    # ── Derived rates ──────────────────────────────────────────────────────

    @property
    def l(self)      -> int:   return self.games - self.wins
    @property
    def wp(self)     -> float: return self.wins / self.games if self.games else 0.0
    @property
    def rd(self)     -> int:   return self.rs - self.ra
    @property
    def rpg(self)    -> float: return self.rs    / self.games if self.games else 0.0
    @property
    def rapg(self)   -> float: return self.ra    / self.games if self.games else 0.0
    @property
    def hrpg(self)   -> float: return self.hr    / self.games if self.games else 0.0
    @property
    def avg(self)    -> float: return self.h     / self.ab    if self.ab    else 0.0
    @property
    def obp(self)    -> float:
        return (self.h + self.bb + self.hbp) / self.pa if self.pa else 0.0
    @property
    def slg(self)    -> float: return self.tb    / self.ab    if self.ab    else 0.0
    @property
    def ops(self)    -> float: return self.obp + self.slg
    @property
    def kpct(self)   -> float: return self.k     / self.pa    if self.pa    else 0.0
    @property
    def bbpct(self)  -> float: return self.bb    / self.pa    if self.pa    else 0.0
    @property
    def babip(self)  -> float:
        denom = self.bip_ab - self.bip_hr
        return self.bip_h / denom if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Ingestion helpers
# ══════════════════════════════════════════════════════════════════════════════

def _teams_in_db() -> set[tuple[int, str]]:
    """Return the set of (season_year, team) pairs already in Supabase."""
    try:
        from ingest_historical import get_supabase_client
        sb  = get_supabase_client()
        res = sb.table("players").select("season_year,team").eq("data_source", "historical").execute()
        return {(r["season_year"], r["team"]) for r in res.data if r["team"]}
    except Exception as exc:
        print(f"  [warn] DB check failed: {exc}", file=sys.stderr)
        return set()


def _ingest_team(season: int, team: str, min_pa: int = 1, min_bf: int = 1) -> None:
    """Ingest a single team-season into Supabase using the same pipeline as
    reingest_roundrobin.py.  Loads Lahman tables once via a module-level cache
    in ingest_historical; subsequent calls are cheap."""
    from ingest_historical import (
        pick_lahman_dir, load_lahman, build_people_lookup,
        agg_batting_player_season, agg_pitching_player_season,
        get_primary_teams_batting, get_primary_teams_pitching,
        compute_pitcher_roles, compute_primary_positions,
        league_context_from_batting, league_context_from_pitching,
        build_hitter_traits, build_pitcher_traits,
        upsert_players, get_supabase_client,
    )
    import pandas as pd

    # ── Load Lahman tables (expensive once; cheap on repeat calls via module cache)
    lahman_dir = pick_lahman_dir(None)
    tables     = load_lahman(lahman_dir)
    ppl        = build_people_lookup(tables.people)
    bat_ps     = agg_batting_player_season(tables.batting)
    pit_ps     = agg_pitching_player_season(tables.pitching)
    bat_lu     = get_primary_teams_batting(tables.batting)
    pit_lu     = get_primary_teams_pitching(tables.pitching)
    pos_lu     = compute_primary_positions(tables.fielding)
    ctx        = league_context_from_batting(bat_ps)
    pit_ctx    = league_context_from_pitching(pit_ps)

    bat_s  = bat_ps[bat_ps["yearID"] == season].copy()
    pit_s  = pit_ps[pit_ps["yearID"] == season].copy()
    ctx_s  = ctx[ctx["yearID"] == season]
    if ctx_s.empty:
        raise ValueError(f"No league context for {season}")
    lg_row     = ctx_s.iloc[0].to_dict()
    pit_ctx_s  = pit_ctx[pit_ctx["yearID"] == season]
    pit_lg_row = pit_ctx_s.iloc[0].to_dict() if not pit_ctx_s.empty else lg_row

    bat_s = bat_s.merge(ppl, on="playerID", how="left")
    pit_s = pit_s.merge(ppl, on="playerID", how="left")

    C = {}
    hitter_rows: list[dict]  = []
    pitcher_rows: list[dict] = []

    for _, r in bat_s.iterrows():
        if r["PA"] < min_pa:
            continue
        key = (r["playerID"], int(r["yearID"]))
        if bat_lu.get(key) != team:
            continue
        traits = build_hitter_traits(r, lg_row, {}, C)
        hitter_rows.append({
            "player_id":        r["playerID"],
            "player_name":      r.get("player_name", r["playerID"]),
            "season_year":      int(r["yearID"]),
            "team":             team,
            "bats":             str(r["bats"])   if r.get("bats")   not in (None, float("nan")) else None,
            "throws":           str(r["throws"]) if r.get("throws") not in (None, float("nan")) else None,
            "pa":               int(r["PA"]),
            **traits,
            "stuff": None, "control": None, "movement": None,
            "ip": None, "pitcher_role": None,
            "primary_position": pos_lu.get(key),
            "data_source": "historical",
        })

    for _, r in pit_s.iterrows():
        if r["BF"] < min_bf:
            continue
        key = (r["playerID"], int(r["yearID"]))
        if pit_lu.get(key) != team:
            continue
        traits      = build_pitcher_traits(r, pit_lg_row, C)
        ip_val      = float(r.get("IPouts", 0)) / 3.0
        g           = int(r.get("G", 1))
        gs          = int(r.get("GS", 0))
        role        = "SP" if g > 0 and gs / g >= 0.5 else "RP"
        pitcher_rows.append({
            "player_id":        r["playerID"],
            "player_name":      r.get("player_name", r["playerID"]),
            "season_year":      int(r["yearID"]),
            "team":             team,
            "bats":             str(r["bats"])   if r.get("bats")   not in (None, float("nan")) else None,
            "throws":           str(r["throws"]) if r.get("throws") not in (None, float("nan")) else None,
            "ip":               round(ip_val, 1),
            **traits,
            "contact": None, "power": None, "eye": None,
            "ak": None, "gap": None, "speed": None, "pa": None,
            "pitcher_role": role, "primary_position": None,
            "data_source": "historical",
        })

    # Merge two-way players
    from typing import Dict, Tuple
    merged: Dict[Tuple[str, int], dict] = {}
    for row in hitter_rows + pitcher_rows:
        k = (row["player_id"], row["season_year"])
        if k not in merged:
            merged[k] = row
        else:
            merged[k].update({col: v for col, v in row.items() if v is not None})

    rows = list(merged.values())
    if not rows:
        raise ValueError(f"No players found for {season} {team} — check Lahman teamID")

    sb = get_supabase_client()
    CHUNK = 500
    for i in range(0, len(rows), CHUNK):
        upsert_players(sb, rows[i : i + CHUNK])
    n_h = sum(1 for r in rows if r.get("contact") is not None)
    n_p = sum(1 for r in rows if r.get("stuff")   is not None)
    print(f"    Ingested {len(rows)} rows  ({n_h} hitters, {n_p} pitchers)", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# Round-robin simulation
# ══════════════════════════════════════════════════════════════════════════════

def run_round_robin(
    rosters:   list,
    labels:    list[str],
    hist_wps:  list[float],
    n:         int,
    seed_base: str,
) -> dict[str, MidTierStats]:
    stats = {
        lbl: MidTierStats(lbl, hist_wps[i])
        for i, lbl in enumerate(labels)
    }

    ordered_pairs = list(itertools.permutations(range(len(labels)), 2))
    rng_sched     = random.Random(hash(seed_base) & 0xFFFF_FFFF)
    schedule: list[tuple[int, int]] = []
    while len(schedule) < n:
        rng_sched.shuffle(ordered_pairs)
        schedule.extend(ordered_pairs)
    schedule = schedule[:n]

    report_every = max(1, n // 10)
    t0 = time.perf_counter()

    for game_idx, (ai, hi) in enumerate(schedule):
        away_lbl = labels[ai]
        home_lbl = labels[hi]

        away_dict = rosters[ai].for_game_engine(rotation_index=game_idx)
        home_dict = rosters[hi].for_game_engine(rotation_index=game_idx)
        game_id   = f"{seed_base}-{away_lbl}-{home_lbl}-{game_idx:04d}"
        box       = simulate_game(away_dict, home_dict, game_id=game_id, verbose=False)

        away_rs  = box.final_score["away"]
        home_rs  = box.final_score["home"]
        away_won = away_rs > home_rs

        for ev in box.pa_events:
            if ev.half == "top":
                stats[away_lbl].bat(ev)
            else:
                stats[home_lbl].bat(ev)

        stats[away_lbl].end_game(away_rs, home_rs, away_won)
        stats[home_lbl].end_game(home_rs, away_rs, not away_won)

        if (game_idx + 1) % report_every == 0:
            elapsed = time.perf_counter() - t0
            rate    = (game_idx + 1) / elapsed
            eta     = (n - game_idx - 1) / rate
            print(
                f"  {game_idx + 1:>5}/{n}  {elapsed:>5.1f}s  ~{eta:>4.0f}s left",
                file=sys.stderr, flush=True,
            )

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def _chk(val: float, lo: float, hi: float) -> str:
    return "✓" if lo <= val <= hi else f"⚠ OUTSIDE [{lo:.3f}–{hi:.3f}]"


def print_report(
    stats:     dict[str, MidTierStats],
    labels:    list[str],
    n:         int,
    elapsed:   float,
    seed_base: str,
) -> None:
    W = 84
    sorted_teams = sorted(stats.values(), key=lambda t: (-t.wp, -t.rd))

    # ── League aggregates ────────────────────────────────────────────────────
    lg_games   = sum(t.games    for t in stats.values())
    lg_rs      = sum(t.rs       for t in stats.values())
    lg_pa      = sum(t.pa       for t in stats.values())
    lg_ab      = sum(t.ab       for t in stats.values())
    lg_h       = sum(t.h        for t in stats.values())
    lg_hr      = sum(t.hr       for t in stats.values())
    lg_tb      = sum(t.tb       for t in stats.values())
    lg_bb      = sum(t.bb       for t in stats.values())
    lg_hbp     = sum(t.hbp      for t in stats.values())
    lg_k       = sum(t.k        for t in stats.values())
    lg_bip_h   = sum(t.bip_h    for t in stats.values())
    lg_bip_ab  = sum(t.bip_ab   for t in stats.values())
    lg_bip_hr  = sum(t.bip_hr   for t in stats.values())

    lg_rpg    = lg_rs / lg_games        if lg_games else 0.0   # runs scored per team per game
    lg_avg    = lg_h   / lg_ab       if lg_ab    else 0.0
    lg_obp    = (lg_h + lg_bb + lg_hbp) / lg_pa if lg_pa else 0.0
    lg_slg    = lg_tb  / lg_ab       if lg_ab    else 0.0
    lg_babip  = lg_bip_h / (lg_bip_ab - lg_bip_hr) if (lg_bip_ab - lg_bip_hr) > 0 else 0.0
    lg_kpct   = lg_k   / lg_pa       if lg_pa    else 0.0
    lg_bbpct  = lg_bb  / lg_pa       if lg_pa    else 0.0

    print(f"\n{'═' * W}")
    print(f"  MID-TIER VALIDATION  —  {n:,} games  ({len(labels)} teams × balanced round robin)")
    print(f"  seed: '{seed_base}'")
    print(f"{'═' * W}")
    print(f"  {'Team':<12}  {'Real W%':>8}  {'Archetype'}")
    for season, team, lbl, hwp in TARGETS:
        print(f"  {lbl:<12}  {hwp:>8.3f}   {season} {team} — mid-tier historical record")

    # ── Standings ────────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'TEAM':<12}  {'W':>4}  {'L':>4}  {'SIM W%':>7}  {'REAL W%':>8}  "
          f"{'RS/G':>6}  {'RA/G':>6}  {'DIFF':>6}  {'HR/G':>5}")
    print(f"  {'─'*12}  {'─'*4}  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}")
    for ts in sorted_teams:
        print(f"  {ts.label:<12}"
              f"  {ts.wins:>4}  {ts.l:>4}"
              f"  {ts.wp:>7.3f}"
              f"  {ts.hist_wp:>8.3f}"
              f"  {ts.rpg:>6.2f}  {ts.rapg:>6.2f}"
              f"  {ts.rd:>+6}"
              f"  {ts.hrpg:>5.2f}")
    print(f"\n  {'LEAGUE AVG':<12}  {'':>4}  {'':>4}  {'':>7}  {'':>8}  "
          f"  {lg_rpg:>6.2f}  {lg_rpg:>6.2f}  {'':>6}  {lg_hr/lg_games:>5.2f}")

    # ── Batting Rates ─────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'TEAM':<12}  {'AVG':>6}  {'OBP':>6}  {'SLG':>6}  {'OPS':>6}  "
          f"{'BABIP':>6}  {'K%':>6}  {'BB%':>6}  {'HR/G':>6}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for ts in sorted_teams:
        print(f"  {ts.label:<12}"
              f"  {ts.avg:>6.3f}"
              f"  {ts.obp:>6.3f}"
              f"  {ts.slg:>6.3f}"
              f"  {ts.ops:>6.3f}"
              f"  {ts.babip:>6.3f}"
              f"  {ts.kpct:>5.1%}"
              f"  {ts.bbpct:>5.1%}"
              f"  {ts.hrpg:>6.2f}")
    print(f"\n  {'LEAGUE AVG':<12}"
          f"  {lg_avg:>6.3f}"
          f"  {lg_obp:>6.3f}"
          f"  {lg_slg:>6.3f}"
          f"  {lg_obp+lg_slg:>6.3f}"
          f"  {lg_babip:>6.3f}"
          f"  {lg_kpct:>5.1%}"
          f"  {lg_bbpct:>5.1%}"
          f"  {lg_hr/lg_games:>6.2f}")

    # ── W% distribution ──────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  WIN % DISTRIBUTION  (target: all teams within .380–.620)")
    print()
    BAR_W = 30
    wp_values = [ts.wp for ts in sorted_teams]
    wp_min, wp_max = min(wp_values), max(wp_values)
    for ts in sorted_teams:
        bar_fill = int(round(ts.wp * BAR_W))
        bar = "█" * bar_fill + "░" * (BAR_W - bar_fill)
        delta = ts.wp - ts.hist_wp
        print(f"  {ts.label:<12}  {bar}  {ts.wp:.3f}  (real {ts.hist_wp:.3f}  sim Δ{delta:+.3f})")
    print(f"\n  Range: {wp_min:.3f} – {wp_max:.3f}  "
          f"({'✓ within spec' if wp_max - wp_min <= 0.24 else '⚠ spread too wide'})")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  SANITY CHECKS")
    print()

    def _row(label: str, val: float, lo: float, hi: float, fmt: str = ".3f") -> None:
        status = "✓" if lo <= val <= hi else "⚠ OUTSIDE RANGE"
        print(f"  {label:<34}  {val:{fmt}}  [{lo:{fmt}}–{hi:{fmt}}]  {status}")

    _row("League BABIP",           lg_babip, 0.285, 0.320)
    _row("League runs/game",       lg_rpg,   4.0,   5.8,   ".2f")
    _row("League AVG",             lg_avg,   0.240, 0.295)
    _row("League OBP",             lg_obp,   0.300, 0.375)
    _row("League SLG",             lg_slg,   0.360, 0.510)
    _row("League K%",              lg_kpct,  0.07,  0.22,  ".3f")
    _row("League BB%",             lg_bbpct, 0.06,  0.13,  ".3f")
    _row("W% spread (max − min)",  wp_max - wp_min, 0.00, 0.240, ".3f")

    # Per-team W% bounds
    print()
    for ts in sorted_teams:
        ok = 0.380 <= ts.wp <= 0.620
        print(f"  {ts.label:<12}  W% = {ts.wp:.3f}  {'✓' if ok else '⚠ OUTSIDE .380–.620'}")

    # ── Overall verdict ────────────────────────────────────────────────────────
    checks = [
        0.285 <= lg_babip <= 0.320,
        4.0   <= lg_rpg   <= 5.8,
        0.240 <= lg_avg   <= 0.285,
        0.300 <= lg_obp   <= 0.370,
        wp_max - wp_min   <= 0.240,
        all(0.380 <= ts.wp <= 0.620 for ts in sorted_teams),
    ]
    passed = sum(checks)
    total  = len(checks)

    print(f"\n{'─' * W}")
    verdict = "PASS" if passed == total else f"PARTIAL ({passed}/{total})"
    print(f"  VALIDATION RESULT: {verdict}")
    if passed < total:
        print(f"  Review ⚠ flags above for calibration issues.")

    print(f"\n{'═' * W}")
    print(f"  {n:,} games  ·  seed='{seed_base}'  ·  "
          f"{elapsed:.1f}s  ·  {n / elapsed:.0f} games/sec")
    print(f"{'═' * W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mid-tier validation: 6 average teams, balanced round robin"
    )
    parser.add_argument("--n",           type=int, default=1000,
                        help="Total games to simulate (default: 1000)")
    parser.add_argument("--seed-base",   type=str, default="mid-tier-val-2026",
                        help="Seed prefix for game IDs")
    parser.add_argument("--no-dh",       action="store_true",
                        help="Pitcher bats 9th (pre-DH / NL rules)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip DB check and ingestion (all teams already loaded)")
    parser.add_argument("--min-pa",      type=int, default=1,
                        help="Minimum PA for hitter inclusion (default: 1)")
    parser.add_argument("--min-bf",      type=int, default=1,
                        help="Minimum BF for pitcher inclusion (default: 1)")
    args = parser.parse_args()

    use_dh = not args.no_dh

    # ── Step 1: Ensure all teams are ingested ─────────────────────────────────
    if not args.skip_ingest:
        print(f"\n  Checking which teams are in Supabase …", file=sys.stderr)
        in_db = _teams_in_db()
        for season, team, label, _ in TARGETS:
            if (season, team) in in_db:
                print(f"  {label}  →  already in DB", file=sys.stderr)
            else:
                print(f"  {label}  →  NOT in DB — ingesting …", file=sys.stderr)
                try:
                    _ingest_team(season, team, args.min_pa, args.min_bf)
                    print(f"  {label}  →  ingested ✓", file=sys.stderr)
                except Exception as exc:
                    print(f"  {label}  →  FAILED: {exc}", file=sys.stderr)
                    sys.exit(1)
    else:
        print(f"\n  --skip-ingest: assuming all teams are in DB.", file=sys.stderr)

    # ── Step 2: Load rosters ──────────────────────────────────────────────────
    print(f"\n  Loading rosters …", file=sys.stderr)
    rosters:   list = []
    labels:    list[str] = []
    hist_wps:  list[float] = []
    missing:   list[str] = []

    for season, team, label, hwp in TARGETS:
        try:
            r = load_team(season, team, use_dh=use_dh)
            rosters.append(r)
            labels.append(label)
            hist_wps.append(hwp)
            sp_rotation = [r.pitcher] + [p for p in r.bullpen if p.get("pitcher_role") == "SP"]
            print(f"  {label}  →  loaded  SP rotation ({len(sp_rotation)}): "
                  f"{', '.join(p['name'] for p in sp_rotation)}", file=sys.stderr)
        except Exception as exc:
            print(f"  {label}  →  FAILED to load: {exc}", file=sys.stderr)
            missing.append(label)

    if missing:
        print(f"\n  ERROR: Could not load {missing}. "
              f"Run without --skip-ingest to retry ingestion.", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Run simulation ─────────────────────────────────────────────────
    n_pairs = len(labels) * (len(labels) - 1)
    print(
        f"\n  Simulating {args.n:,} games across {len(labels)} teams "
        f"({n_pairs} ordered matchup pairs) …\n",
        file=sys.stderr,
    )

    t0     = time.perf_counter()
    stats  = run_round_robin(rosters, labels, hist_wps, args.n, args.seed_base)
    elapsed = time.perf_counter() - t0

    # ── Step 4: Report ─────────────────────────────────────────────────────────
    print_report(stats, labels, args.n, elapsed, args.seed_base)


if __name__ == "__main__":
    main()
