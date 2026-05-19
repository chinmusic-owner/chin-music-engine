"""
round_robin_sim.py — Cross-era 4-team Round Robin

Pits four archetypal teams from 1927 and 2000 against one another in a
balanced 500-game simulation, then prints a league-standings-style report.

Team archetypes
───────────────
  1927 NYA  Power / Legend Era     (Ruth, Gehrig)
  1927 PIT  Contact / Small Ball   (Waner brothers)
  2000 NYA  Modern Dynasty         (Jeter, Williams, Posada)
  2000 ARI  High-STF Pitching      (Randy Johnson, Schilling)
            ↳ falls back to 2000 NYN if ARI not in DB

Schedule
────────
All ordered pairs (A-away vs B-home) are generated and shuffled in random
blocks until N games are scheduled.  With 4 teams there are 12 ordered
pairs (4 × 3 permutations); each team plays ~N/2 games total.

DH Rules
────────
Universal DH (use_dh=True) by default so all teams field 9 position players.
Use --no-dh to put the pitcher in the 9th lineup slot for 1927 teams and NL
2000 teams.

Usage
─────
  python3 round_robin_sim.py [--n 500] [--seed-base rr2026] [--no-dh]
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from dataclasses import dataclass, field

from roster_manager import load_team, load_team_from_pilot
from game_engine import simulate_game


# ── Constants ────────────────────────────────────────────────────────────────

HITS     = {"Single", "Double", "Triple", "HR", "InfieldHit"}
_TB_VALS = {"InfieldHit": 1, "Single": 1, "Double": 2, "Triple": 3, "HR": 4}
PILOT_NYA27 = "pilot_1927_nya.json"

# ── Per-team aggregate accumulator ───────────────────────────────────────────

@dataclass
class TeamStats:
    """Accumulates batting + record stats for one team across all its games."""
    label: str
    games: int = 0
    wins:  int = 0
    rs:    int = 0      # runs scored
    ra:    int = 0      # runs allowed
    h:     int = 0
    tb:    int = 0      # total bases (for SLG)
    hr:    int = 0
    bb:    int = 0
    k:     int = 0
    pa:    int = 0
    ab:    int = 0
    hbp:   int = 0

    def bat(self, ev) -> None:
        """Tally one PAEvent for the batting side."""
        self.pa += 1
        o = ev.outcome
        if o not in ("BB", "HBP"):
            self.ab += 1
        if o == "BB":  self.bb  += 1
        if o == "HBP": self.hbp += 1
        if o == "K":   self.k   += 1
        if o in HITS:
            self.h  += 1
            self.tb += _TB_VALS.get(o, 1)
        if o == "HR":  self.hr  += 1

    def end_game(self, rs: int, ra: int, won: bool) -> None:
        self.games += 1
        self.wins  += int(won)
        self.rs    += rs
        self.ra    += ra

    # ── Derived rates ─────────────────────────────────────────────────────

    @property
    def l(self)     -> int:   return self.games - self.wins
    @property
    def wp(self)    -> float: return self.wins / self.games if self.games else 0.0
    @property
    def rd(self)    -> int:   return self.rs - self.ra
    @property
    def rpg(self)   -> float: return self.rs / self.games if self.games else 0.0
    @property
    def rapg(self)  -> float: return self.ra / self.games if self.games else 0.0
    @property
    def hrpg(self)  -> float: return self.hr / self.games if self.games else 0.0
    @property
    def ba(self)    -> float: return self.h  / self.ab   if self.ab   else 0.0
    @property
    def obp(self)   -> float:
        return (self.h + self.bb + self.hbp) / self.pa if self.pa else 0.0
    @property
    def slg(self)   -> float: return self.tb / self.ab   if self.ab   else 0.0
    @property
    def ops(self)   -> float: return self.obp + self.slg
    @property
    def kpct(self)  -> float: return self.k   / self.pa  if self.pa   else 0.0
    @property
    def bbpct(self) -> float: return self.bb  / self.pa  if self.pa   else 0.0


# ── Head-to-head tracking ─────────────────────────────────────────────────────
#
# h2h[A][B] tracks A's record in all games where A and B faced each other,
# regardless of home / away assignment.
#   [0] = A wins
#   [1] = total games played between A and B
#   [2] = A total runs scored
#   [3] = A total runs allowed

def _make_h2h(labels: list[str]) -> dict[str, dict[str, list[int]]]:
    return {
        a: {b: [0, 0, 0, 0] for b in labels if b != a}
        for a in labels
    }


def _update_h2h(
    h2h:      dict,
    away_lbl: str,
    home_lbl: str,
    away_rs:  int,
    home_rs:  int,
    away_won: bool,
) -> None:
    # Both teams see the same total game added to their opponent counter
    h2h[away_lbl][home_lbl][1] += 1
    h2h[home_lbl][away_lbl][1] += 1
    # Only the winner increments their own win count
    if away_won:
        h2h[away_lbl][home_lbl][0] += 1
    else:
        h2h[home_lbl][away_lbl][0] += 1
    # RS/RA from each team's perspective
    h2h[away_lbl][home_lbl][2] += away_rs
    h2h[away_lbl][home_lbl][3] += home_rs
    h2h[home_lbl][away_lbl][2] += home_rs
    h2h[home_lbl][away_lbl][3] += away_rs


# ── Simulation engine ────────────────────────────────────────────────────────

def run_round_robin(
    rosters:   list,          # list[TeamRoster], one per team
    labels:    list[str],     # display labels aligned with rosters
    n:         int,
    seed_base: str,
) -> tuple[dict[str, TeamStats], dict[str, dict[str, list[int]]]]:
    """
    Run N games in a balanced round robin.

    Schedule: ordered pairs (away_idx, home_idx) via itertools.permutations
    are shuffled in blocks and repeated until N games are filled.

    Returns:
        team_stats : {label: TeamStats}
        h2h        : nested dict as described in _make_h2h
    """
    team_stats: dict[str, TeamStats] = {lbl: TeamStats(lbl) for lbl in labels}
    h2h = _make_h2h(labels)

    # Generate balanced schedule (all ordered pairs, shuffled repeatedly)
    ordered_pairs = list(itertools.permutations(range(len(labels)), 2))
    rng_sched = random.Random(hash(seed_base) & 0xFFFF_FFFF)
    schedule: list[tuple[int, int]] = []
    while len(schedule) < n:
        rng_sched.shuffle(ordered_pairs)
        schedule.extend(ordered_pairs)
    schedule = schedule[:n]

    report_every = max(1, n // 10)
    t0 = time.perf_counter()

    for game_idx, (away_idx, home_idx) in enumerate(schedule):
        away_lbl = labels[away_idx]
        home_lbl = labels[home_idx]

        # for_game_engine() returns a fresh dict each call; rotation_index
        # cycles starters so all SPs get innings across the series.
        away_dict = rosters[away_idx].for_game_engine(rotation_index=game_idx)
        home_dict = rosters[home_idx].for_game_engine(rotation_index=game_idx)

        # Embed both team labels in the game_id for deterministic, unique seeds
        game_id = f"{seed_base}-{away_lbl}-{home_lbl}-{game_idx:04d}"
        box = simulate_game(away_dict, home_dict, game_id=game_id, verbose=False)

        away_rs  = box.final_score["away"]
        home_rs  = box.final_score["home"]
        away_won = away_rs > home_rs

        # Accumulate batting events
        for ev in box.pa_events:
            if ev.half == "top":
                team_stats[away_lbl].bat(ev)
            else:
                team_stats[home_lbl].bat(ev)

        # Game record
        team_stats[away_lbl].end_game(away_rs, home_rs, away_won)
        team_stats[home_lbl].end_game(home_rs, away_rs, not away_won)
        _update_h2h(h2h, away_lbl, home_lbl, away_rs, home_rs, away_won)

        if (game_idx + 1) % report_every == 0:
            elapsed = time.perf_counter() - t0
            rate    = (game_idx + 1) / elapsed
            eta     = (n - game_idx - 1) / rate
            print(
                f"  {game_idx + 1:>5}/{n}  {elapsed:>5.1f}s  ~{eta:>4.0f}s left",
                file=sys.stderr, flush=True,
            )

    return team_stats, h2h


# ── Report ───────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 18) -> str:
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)


def print_report(
    team_stats: dict[str, TeamStats],
    h2h:        dict[str, dict[str, list[int]]],
    labels:     list[str],
    n:          int,
    elapsed:    float,
    seed_base:  str,
    archetypes: dict[str, str],   # label → archetype description
) -> None:
    W = 80
    sorted_teams = sorted(team_stats.values(), key=lambda t: (-t.wp, -t.rd))

    print(f"\n{'═' * W}")
    print(f"  ROUND ROBIN STANDINGS  —  {n:,} games  ({len(labels)} teams)")
    print(f"{'═' * W}")
    for lbl, desc in archetypes.items():
        print(f"    {lbl:<14}  {desc}")

    # ── League Standings ─────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(
        f"  {'TEAM':<14}  {'W':>4}  {'L':>4}  {'W%':>6}  "
        f"{'RS/G':>5}  {'RA/G':>5}  {'DIFF':>5}  {'HR/G':>5}"
    )
    print(
        f"  {'─' * 14}  {'─' * 4}  {'─' * 4}  {'─' * 6}  "
        f"{'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 5}"
    )
    for ts in sorted_teams:
        print(
            f"  {ts.label:<14}"
            f"  {ts.wins:>4}"
            f"  {ts.l:>4}"
            f"  {ts.wp:>6.1%}"
            f"  {ts.rpg:>5.2f}"
            f"  {ts.rapg:>5.2f}"
            f"  {ts.rd:>+5}"
            f"  {ts.hrpg:>5.2f}"
        )

    # ── Batting Rates ─────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(
        f"  {'TEAM':<14}  {'AVG':>5}  {'OBP':>5}  {'SLG':>5}  {'OPS':>5}  "
        f"{'K%':>6}  {'BB%':>6}  {'HR%':>6}"
    )
    print(
        f"  {'─' * 14}  {'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 5}  "
        f"{'─' * 6}  {'─' * 6}  {'─' * 6}"
    )
    for ts in sorted_teams:
        print(
            f"  {ts.label:<14}"
            f"  {ts.ba:>5.3f}"
            f"  {ts.obp:>5.3f}"
            f"  {ts.slg:>5.3f}"
            f"  {ts.ops:>5.3f}"
            f"  {ts.kpct:>5.1%}"
            f"  {ts.bbpct:>5.1%}"
            f"  {ts.hrpg:>5.2f}"
        )

    # ── Head-to-Head Matrix ───────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  HEAD-TO-HEAD  (row = team, col = opponent;  format: W-L  RS/RA)")
    print()
    col_w = 15
    header = f"  {'':14}"
    for lbl in labels:
        header += f"  {lbl:<{col_w}}"
    print(header)
    print(f"  {'─' * 14}" + ("  " + "─" * col_w) * len(labels))
    for a in labels:
        row = f"  {a:<14}"
        for b in labels:
            if a == b:
                row += f"  {'—':<{col_w}}"
            else:
                rec = h2h[a][b]
                wins  = rec[0]
                games = rec[1]
                rs    = rec[2] / games if games else 0
                ra    = rec[3] / games if games else 0
                cell  = f"{wins}-{games - wins} {rs:.1f}/{ra:.1f}"
                row  += f"  {cell:<{col_w}}"
        print(row)

    # ── Cross-Era Spotlight ───────────────────────────────────────────────────
    era27 = [l for l in labels if "1927" in l]
    era00 = [l for l in labels if "2000" in l]

    if era27 and era00:
        print(f"\n{'─' * W}")
        print(f"  CROSS-ERA SPOTLIGHT: 1927 lineups vs 2000 pitching")
        print()
        for team27 in era27:
            ts27 = team_stats[team27]
            print(f"  {team27:<14}  archetype: {archetypes.get(team27, '')}")
            for team00 in era00:
                rec27 = h2h[team27][team00]
                games = rec27[1]
                if games:
                    wins = rec27[0]
                    rs   = rec27[2] / games
                    ra   = rec27[3] / games
                    bar  = _bar(wins / games, 16)
                    print(
                        f"    vs {team00:<12}  "
                        f"{wins:>3}W-{games-wins:<3}L  ({wins/games:.0%})  "
                        f"{bar}  "
                        f"RS/G {rs:.2f}  RA/G {ra:.2f}"
                    )
            print()

        print(f"  Contact era (1927 PIT) vs Power era (1927 NYA) — same 2000 opponents:")
        if len(era27) >= 2:
            for team00 in era00:
                rows: list[tuple[float, str]] = []
                for team27 in era27:
                    rec = h2h[team27][team00]
                    if rec[1]:
                        rs  = rec[2] / rec[1]
                        wp  = rec[0] / rec[1]
                        rows.append((wp, f"    {team27:<12} vs {team00}: {rec[0]}-{rec[1]-rec[0]}  W%={wp:.1%}  RS/G={rs:.2f}"))
                for _, line in sorted(rows, reverse=True):
                    print(line)
        print()

    # ── Runs-per-game distribution bars ──────────────────────────────────────
    MAX_R = 16
    print(f"{'─' * W}")
    print(f"  RUNS/GAME DISTRIBUTION")
    print()
    bar_w = 16
    hdr_lbl = f"  {'R':>3}  "
    for ts in sorted_teams:
        hdr_lbl += f"  {ts.label:<{bar_w + 5}}"
    print(hdr_lbl)
    # Build frequency dicts
    freq: dict[str, dict[int, int]] = {}
    for ts in sorted_teams:
        # Reconstruct from totals — we only have aggregate; skip per-game dist
        # (round-robin only stores aggregate, not per-game dist list)
        freq[ts.label] = {}

    # We need the run_dist. Let's carry it: re-collect from ts.rs
    # Actually we don't have it. Skip the histogram for aggregate-only tracking.
    # Show a text summary per team instead.
    print(f"  (Per-game run distribution not tracked in aggregate mode.)")
    print()
    for ts in sorted_teams:
        print(
            f"  {ts.label:<14}  "
            f"avg {ts.rpg:.2f} RS/g  avg {ts.rapg:.2f} RA/g  "
            f"W%={ts.wp:.1%}  run diff {ts.rd:+d}"
        )

    # ── Footer ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(
        f"  {n:,} games  ·  seed='{seed_base}'  ·  "
        f"{elapsed:.1f}s  ·  {n / elapsed:.0f} games/sec"
    )
    print(f"{'═' * W}\n")


# ── Roster loading with graceful fallbacks ────────────────────────────────────

def _load_with_fallback(
    season: int,
    team:   str,
    use_dh: bool,
    pilot:  str | None = None,
) -> "TeamRoster | None":
    """
    Try Supabase first.  If unavailable and a pilot path is given, fall back.
    Returns None if loading fails and no pilot fallback is available.
    """
    try:
        r = load_team(season, team, use_dh=use_dh)
        print(f"  {season} {team:>4}  →  Supabase (historical)", file=sys.stderr)
        return r
    except Exception as e:
        if pilot:
            print(
                f"  {season} {team:>4}  →  pilot JSON  (Supabase: {e})",
                file=sys.stderr,
            )
            try:
                return load_team_from_pilot(
                    pilot, team_id=f"{season}-{team}", use_dh=use_dh
                )
            except Exception as e2:
                print(f"           pilot also failed: {e2}", file=sys.stderr)
        else:
            print(
                f"  {season} {team:>4}  →  FAILED  ({e})",
                file=sys.stderr,
            )
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-era round robin: 4 archetypal teams, N games"
    )
    parser.add_argument("--n",         type=int, default=500,
                        help="Total games to simulate (default: 500)")
    parser.add_argument("--seed-base", type=str, default="rr-2026",
                        help="Seed prefix for game IDs (default: 'rr-2026')")
    parser.add_argument("--no-dh",     action="store_true",
                        help="Pitcher bats 9th for pre-DH / NL teams")
    args = parser.parse_args()

    use_dh = not args.no_dh

    # ── Team definitions ──────────────────────────────────────────────────────
    #
    # Each entry: (season, team_id, display_label, archetype_desc, pilot_fallback)
    # pilot_fallback is used when Supabase is unavailable for that team.
    #
    # ARI is the preferred 4th team (Randy Johnson era); NYN is the fallback.

    TEAM_DEFS: list[tuple[int, str, str, str, str | None]] = [
        (1927, "NYA", "1927 NYA", "Power / Legend Era  (Ruth, Gehrig)",      PILOT_NYA27),
        (1927, "PIT", "1927 PIT", "Contact / Small Ball  (Waner Brothers)",  None),
        (2000, "NYA", "2000 NYA", "Modern Dynasty  (Jeter, Williams)",        None),
        (2000, "ARI", "2000 ARI", "High-STF Pitching  (R. Johnson, Schilling)", None),
    ]

    print(f"\n  Loading rosters …", file=sys.stderr)
    rosters: list = []
    labels:  list[str] = []
    archetypes: dict[str, str] = {}
    missing: list[str] = []

    for season, team, label, desc, pilot in TEAM_DEFS:
        roster = _load_with_fallback(season, team, use_dh, pilot)
        if roster is None:
            # Try 2000 NYN as fallback for 2000 ARI
            if team == "ARI":
                print(
                    f"  2000 ARI not available — trying 2000 NYN as fallback …",
                    file=sys.stderr,
                )
                roster = _load_with_fallback(
                    2000, "NYN", use_dh, pilot=None
                )
                if roster is not None:
                    label = "2000 NYN"
                    desc  = "High-STF Pitching  (Leiter, Hampton fallback)"
                    print(f"           → using 2000 NYN instead", file=sys.stderr)
        if roster is None:
            missing.append(label)
        else:
            rosters.append(roster)
            labels.append(label)
            archetypes[label] = desc

    if missing:
        print(
            f"\n  WARNING: Could not load {missing}.\n"
            f"  Run ingest_historical.py for each missing team-season "
            f"then retry.\n"
            f"  Continuing with {len(rosters)} teams.\n",
            file=sys.stderr,
        )

    if len(rosters) < 2:
        print("  ERROR: Need at least 2 teams to run a simulation.", file=sys.stderr)
        sys.exit(1)

    # Show rotations
    def _show_rotation(roster, lbl: str) -> None:
        sp_rotation = [roster.pitcher] + [
            p for p in roster.bullpen if p.get("pitcher_role") == "SP"
        ]
        names = ", ".join(p["name"] for p in sp_rotation)
        print(f"  {lbl} SP rotation ({len(sp_rotation)}): {names}", file=sys.stderr)

    print(file=sys.stderr)
    for roster, lbl in zip(rosters, labels):
        _show_rotation(roster, lbl)

    print(
        f"\n  Simulating {args.n:,} games across {len(rosters)} teams "
        f"({len(rosters) * (len(rosters) - 1)} ordered matchups) …\n",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    team_stats, h2h = run_round_robin(rosters, labels, args.n, args.seed_base)
    elapsed = time.perf_counter() - t0

    print_report(
        team_stats  = team_stats,
        h2h         = h2h,
        labels      = labels,
        n           = args.n,
        elapsed     = elapsed,
        seed_base   = args.seed_base,
        archetypes  = archetypes,
    )


if __name__ == "__main__":
    main()
