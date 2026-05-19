"""
reingest_roundrobin.py — Batch re-ingest for all cross-era simulation teams

Loads Lahman CSVs and computes the cross-era global z-score baselines ONCE,
then builds and upserts corrected player cards for all eight team-seasons:

    1927 NYA  ·  1927 PIT                     (dead-ball / early power era)
    1954 CLE  ·  1975 CIN  ·  1985 SLN        (post-war / Big Red Machine / speed era)
    2000 NYA  ·  2000 ARI  ·  2001 SEA        (modern power / high-K era)

Why a dedicated batch script?
  Running ingest_historical.py separately for each team triggers 8 full Lahman
  CSV loads and separate global_calibration computations.  This script does
  both expensive steps once (~60–90 s) then loops over all teams in ~2 s each.

Note on global_calibration.py:
  The global z-score reference pool is computed from ALL Lahman seasons
  (1871–2025), not just the teams in Supabase.  Adding more teams to the
  simulation pool does NOT change the global baselines — they are already
  cross-era.  The purpose of this script is to push correctly-calibrated
  player cards to the DB for each simulation team.

Usage:
    python3 reingest_roundrobin.py           # dry run — prints key trait values
    python3 reingest_roundrobin.py --push    # writes corrected rows to Supabase
    python3 reingest_roundrobin.py --push --min-pa 50 --min-bf 50
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Tuple

# ── Importing ingest_historical triggers _GLOBAL_STATS = get_global_stats()
# which loads ALL Lahman CSVs and computes cross-era z-score baselines ONCE.
# Subsequent calls to build_hitter_traits / build_pitcher_traits use the
# cached baselines — no second computation needed.
from ingest_historical import (
    pick_lahman_dir,
    load_lahman,
    build_people_lookup,
    agg_batting_player_season,
    agg_pitching_player_season,
    get_primary_teams_batting,
    get_primary_teams_pitching,
    compute_pitcher_roles,
    compute_primary_positions,
    league_context_from_batting,
    league_context_from_pitching,
    build_hitter_traits,
    build_pitcher_traits,
    upsert_players,
    get_supabase_client,
    _GLOBAL_STATS,
)

# ── Targets ───────────────────────────────────────────────────────────────────

TARGETS: list[tuple[int, str]] = [
    # Dead-ball / early power era
    (1927, "NYA"),
    (1927, "PIT"),
    # Post-war / integration / speed eras
    (1954, "CLE"),
    (1975, "CIN"),
    (1985, "SLN"),
    # Modern power / high-K era
    (2000, "NYA"),
    (2000, "ARI"),
    (2001, "SEA"),
]

# Key players to spotlight in dry-run output (playerID → display name)
SPOTLIGHT: dict[str, str] = {
    # 1927 NYA
    "ruthba01":   "Babe Ruth",
    "gehrilo01":  "Lou Gehrig",
    "hoytwa01":   "Waite Hoyt",
    "pennohe01":  "Herb Pennock",
    "pipgrage01": "George Pipgras",
    # 1927 PIT
    "wanerpa01":  "Paul Waner",
    "wanello01":  "Lloyd Waner",
    "meadowle01": "Lee Meadows",
    # 1954 CLE
    "dobyla01":   "Larry Doby",
    "lemonbo01":  "Bob Lemon",
    "wynnea01":   "Early Wynn",
    # 1975 CIN
    "benchjo01":  "Johnny Bench",
    "morgajo02":  "Joe Morgan",
    "rosepe01":   "Pete Rose",
    "perezto01":  "Tony Perez",
    "gulledo01":  "Don Gullett",
    # 1985 SLN
    "smithoz01":  "Ozzie Smith",
    "mcgeewi01":  "Willie McGee",
    "tudorjo01":  "John Tudor",
    "andujjo01":  "Joaquin Andujar",
    # 2000 NYA
    "jeterde01":  "Derek Jeter",
    "willibe02":  "Bernie Williams",
    "hernaorl01": "Orlando Hernandez",
    "pettian01":  "Andy Pettitte",
    # 2000 ARI
    "johnsra05":  "Randy Johnson",
    "schilcu01":  "Curt Schilling",
    "gonzalu01":  "Luis Gonzalez",
    # 2001 SEA
    "suzukic01":  "Ichiro Suzuki",
    "boonebr01":  "Bret Boone",
    "martied01":  "Edgar Martinez",
    "moyerja01":  "Jamie Moyer",
    "garcifr02":  "Freddy Garcia",
}


# ── Core builder ──────────────────────────────────────────────────────────────

def build_rows_for_team(
    season:                  int,
    team:                    str,
    bat_ps:                  "pd.DataFrame",
    pit_ps:                  "pd.DataFrame",
    ppl:                     "pd.DataFrame",
    batting_team_lookup:     dict,
    pitching_team_lookup:    dict,
    pitcher_role_lookup:     dict,
    primary_position_lookup: dict,
    ctx:                     "pd.DataFrame",
    pit_ctx:                 "pd.DataFrame",
    min_pa:                  int,
    min_bf:                  int,
) -> list[dict]:
    """Build corrected player rows for one team-season."""
    import pandas as pd  # local import to keep top-level light

    C = {}  # legacy parameter — not used inside build_*_traits

    bat_s = bat_ps[bat_ps["yearID"] == season].copy()
    pit_s = pit_ps[pit_ps["yearID"] == season].copy()
    ctx_s = ctx[ctx["yearID"] == season]
    if ctx_s.empty:
        raise ValueError(f"No league context for season {season}. Check Lahman Batting.csv.")
    lg_row = ctx_s.iloc[0].to_dict()

    # Use pitching-table league rates for pitcher traits (BF denominator, includes lg_era)
    pit_ctx_s = pit_ctx[pit_ctx["yearID"] == season]
    pit_lg_row = pit_ctx_s.iloc[0].to_dict() if not pit_ctx_s.empty else lg_row

    bat_s = bat_s.merge(ppl, on="playerID", how="left")
    pit_s = pit_s.merge(ppl, on="playerID", how="left")

    hitter_rows: list[dict] = []
    for _, r in bat_s.iterrows():
        if r["PA"] < min_pa:
            continue
        key     = (r["playerID"], int(r["yearID"]))
        team_id = batting_team_lookup.get(key)
        if team_id != team:
            continue
        traits = build_hitter_traits(r, lg_row, {}, C)
        hitter_rows.append({
            "player_id":        r["playerID"],
            "player_name":      r.get("player_name", r["playerID"]),
            "season_year":      int(r["yearID"]),
            "team":             team_id,
            "bats":             (str(r["bats"])   if r.get("bats")   not in (None, float("nan")) else None),
            "throws":           (str(r["throws"]) if r.get("throws") not in (None, float("nan")) else None),
            "pa":               int(r["PA"]),
            **traits,                               # contact, power, eye, ak, gap, speed
            "stuff":            None,
            "control":          None,
            "movement":         None,
            "ip":               None,
            "pitcher_role":     None,
            "primary_position": primary_position_lookup.get(key),
            "data_source":      "historical",
        })

    pitcher_rows: list[dict] = []
    for _, r in pit_s.iterrows():
        if r["BF"] < min_bf:
            continue
        key     = (r["playerID"], int(r["yearID"]))
        team_id = pitching_team_lookup.get(key)
        if team_id != team:
            continue
        traits   = build_pitcher_traits(r, pit_lg_row, C)
        ip_val   = float(r.get("IPouts", 0)) / 3.0
        g        = int(r.get("G", 1))
        gs       = int(r.get("GS", 0))
        pitcher_role = "SP" if g > 0 and gs / g >= 0.5 else "RP"
        pitcher_rows.append({
            "player_id":        r["playerID"],
            "player_name":      r.get("player_name", r["playerID"]),
            "season_year":      int(r["yearID"]),
            "team":             team_id,
            "bats":             (str(r["bats"])   if r.get("bats")   not in (None, float("nan")) else None),
            "throws":           (str(r["throws"]) if r.get("throws") not in (None, float("nan")) else None),
            "ip":               round(ip_val, 1),
            **traits,                               # stuff, control, movement
            "contact":          None,
            "power":            None,
            "eye":              None,
            "ak":               None,
            "gap":              None,
            "speed":            None,
            "pa":               None,
            "pitcher_role":     pitcher_role,
            "primary_position": None,
            "data_source":      "historical",
        })

    # Two-way players: merge hitter + pitcher rows by (player_id, season_year)
    merged: Dict[Tuple[str, int], dict] = {}
    for row in hitter_rows + pitcher_rows:
        k = (row["player_id"], row["season_year"])
        if k not in merged:
            merged[k] = row
        else:
            merged[k].update({col: v for col, v in row.items() if v is not None})

    return list(merged.values())


# ── Report helpers ────────────────────────────────────────────────────────────

def _print_trait_summary(season: int, team: str, rows: list[dict]) -> None:
    """Print a compact trait table for hitters and pitchers in one team-season."""
    hitters  = [r for r in rows if r.get("contact") is not None]
    pitchers = [r for r in rows if r.get("stuff")   is not None]

    def avg(lst, key):
        vals = [v for r in lst if (v := r.get(key)) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n  {season} {team}  ({len(hitters)} hitters  /  {len(pitchers)} pitchers)")

    # Hitter trait averages
    print(f"  {'':5}  {'CON':>5}  {'POW':>5}  {'EYE':>5}  {'AK':>5}  {'GAP':>5}")
    print(f"  {'avg':5}  "
          f"{avg(hitters,'contact'):>5.1f}  "
          f"{avg(hitters,'power'):>5.1f}  "
          f"{avg(hitters,'eye'):>5.1f}  "
          f"{avg(hitters,'ak'):>5.1f}  "
          f"{avg(hitters,'gap'):>5.1f}")

    # Pitcher trait averages
    print(f"  {'':5}  {'STF':>5}  {'CTL':>5}  {'CMD':>5}")
    print(f"  {'avg':5}  "
          f"{avg(pitchers,'stuff'):>5.1f}  "
          f"{avg(pitchers,'control'):>5.1f}  "
          f"{avg(pitchers,'movement'):>5.1f}")

    # Spotlight players
    spot = [r for r in rows if r["player_id"] in SPOTLIGHT]
    if spot:
        print()
        for r in sorted(spot, key=lambda x: SPOTLIGHT.get(x["player_id"], "")):
            name  = SPOTLIGHT.get(r["player_id"], r.get("player_name", r["player_id"]))
            if r.get("contact") is not None:
                print(f"    H  {name:<22}  "
                      f"CON={r['contact']:>3}  POW={r['power']:>3}  EYE={r['eye']:>3}  "
                      f"AK={r.get('ak',0):>3}  GAP={r.get('gap',0):>3}")
            elif r.get("stuff") is not None:
                print(f"    P  {name:<22}  "
                      f"STF={r['stuff']:>3}  CTL={r['control']:>3}  CMD={r['movement']:>3}  "
                      f"({r.get('pitcher_role','?')}  IP={r.get('ip',0):.0f})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch re-ingest 1927 NYA/PIT and 2000 NYA/ARI with cross-era calibration"
    )
    parser.add_argument("--push",   action="store_true",
                        help="Write corrected rows to Supabase (omit for dry run)")
    parser.add_argument("--min-pa", type=int, default=1,
                        help="Minimum PA to include a hitter row (default 1)")
    parser.add_argument("--min-bf", type=int, default=1,
                        help="Minimum BF to include a pitcher row (default 1)")
    parser.add_argument("--lahman-dir", default=None)
    args = parser.parse_args()

    print(f"\n  Global calibration baselines loaded:")
    print(f"    hitter seasons  : {_GLOBAL_STATS['n_hitter_seasons']:,}")
    print(f"    pitcher seasons : {_GLOBAL_STATS['n_pitcher_seasons']:,}")
    print(f"  (computed from ALL Lahman seasons — cross-era reference)\n")

    lahman_dir = pick_lahman_dir(args.lahman_dir)
    print(f"  Loading Lahman tables from '{lahman_dir}' …", flush=True)
    tables = load_lahman(lahman_dir)

    ppl                      = build_people_lookup(tables.people)
    bat_ps                   = agg_batting_player_season(tables.batting)
    pit_ps                   = agg_pitching_player_season(tables.pitching)
    batting_team_lookup      = get_primary_teams_batting(tables.batting)
    pitching_team_lookup     = get_primary_teams_pitching(tables.pitching)
    pitcher_role_lookup      = compute_pitcher_roles(tables.pitching)
    primary_position_lookup  = compute_primary_positions(tables.fielding)
    ctx                      = league_context_from_batting(bat_ps)
    pit_ctx                  = league_context_from_pitching(pit_ps)

    print("  Lahman tables ready.\n")

    # ── Build rows for all 4 team-seasons ─────────────────────────────────────
    all_rows: list[dict] = []
    for season, team in TARGETS:
        rows = build_rows_for_team(
            season, team, bat_ps, pit_ps, ppl,
            batting_team_lookup, pitching_team_lookup,
            pitcher_role_lookup, primary_position_lookup,
            ctx, pit_ctx, args.min_pa, args.min_bf,
        )
        all_rows.extend(rows)
        n_pit = sum(1 for r in rows if r.get("pitcher_role"))
        n_pos = sum(1 for r in rows if r.get("primary_position"))
        print(f"  {season} {team:<4}  {len(rows):>3} rows  "
              f"({n_pit} pitchers, {n_pos} with primary_position)")
        _print_trait_summary(season, team, rows)

    print(f"\n  Total rows across all 4 teams: {len(all_rows)}")

    # ── Push ──────────────────────────────────────────────────────────────────
    if not args.push:
        print("\n  Dry run — add --push to write to Supabase.")
        print("  Verify the trait values above look correct before pushing.")
        return

    print("\n  Pushing to Supabase …", flush=True)
    sb = get_supabase_client()

    # Upsert in chunks of 500 to avoid request size limits
    CHUNK = 500
    total_upserted = 0
    for i in range(0, len(all_rows), CHUNK):
        chunk = all_rows[i : i + CHUNK]
        resp  = upsert_players(sb, chunk)
        n     = len(resp.data) if resp.data else len(chunk)
        total_upserted += n
        print(f"    chunk {i // CHUNK + 1}: {n} rows upserted")

    print(f"\n  Done. {total_upserted} rows upserted across 4 team-seasons.")
    print("  Run `python3 calibration_report.py --n 50` to verify correction.\n")


if __name__ == "__main__":
    main()
