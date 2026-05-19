"""
simulate_series.py  —  Monte Carlo series simulator

Runs N independent games between two fixed rosters, varying only the game_id
(which seeds every PA deterministically).  Reports aggregate stats.

Roster sources (in priority order):
    1. Supabase `players` table  (load_team)  — requires ingested historical data
    2. Pilot JSON fallback        (load_team_from_pilot)  — for 1927 NYA only

Usage:
    python3 simulate_series.py [--n 10000] [--seed-base LABEL]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field

from roster_manager import load_team, load_team_from_pilot
from game_engine import simulate_game

HITS = {"Single", "Double", "Triple", "HR", "InfieldHit"}
PILOT_JSON = "pilot_1927_nya.json"


# ── Per-team accumulator ──────────────────────────────────────────────────────

@dataclass
class Accum:
    """Accumulates per-PA and per-game stats for one side across N games."""
    name: str
    n:          int  = 0
    wins:       int  = 0
    runs_total: int  = 0
    h:          int  = 0
    hr:         int  = 0
    bb:         int  = 0
    k:          int  = 0
    hbp:        int  = 0
    err:        int  = 0
    pa:         int  = 0
    run_dist:   list = field(default_factory=list)   # one entry per game

    def record_event(self, ev) -> None:
        """Tally one PAEvent (already filtered to the correct half)."""
        self.pa += 1
        o = ev.outcome
        if o in HITS:    self.h   += 1
        if o == "HR":    self.hr  += 1
        if o == "BB":    self.bb  += 1
        if o == "K":     self.k   += 1
        if o == "HBP":   self.hbp += 1
        if o == "Error": self.err += 1

    def finalize_game(self, runs: int, won: bool) -> None:
        self.n          += 1
        self.wins       += int(won)
        self.runs_total += runs
        self.run_dist.append(runs)

    # ── Derived rates (computed over the whole sample) ───────────────────────

    @property
    def win_pct(self)  -> float: return self.wins / self.n if self.n else 0.0
    @property
    def ab(self)       -> int:   return self.pa - self.bb - self.hbp
    @property
    def ba(self)       -> float: return self.h / self.ab if self.ab else 0.0
    @property
    def obp(self)      -> float:
        return (self.h + self.bb + self.hbp) / self.pa if self.pa else 0.0
    @property
    def k_pct(self)    -> float: return self.k  / self.pa if self.pa else 0.0
    @property
    def bb_pct(self)   -> float: return self.bb / self.pa if self.pa else 0.0
    @property
    def hr_pct(self)   -> float: return self.hr / self.pa if self.pa else 0.0

    def pg(self, total: int) -> float:
        """Total stat divided by number of games (per-game average)."""
        return total / self.n if self.n else 0.0

    def percentiles(self) -> tuple[int, float, int, int]:
        """(Q1, median, Q3, max) of run totals."""
        if not self.run_dist:
            return (0, 0, 0, 0)
        qs = statistics.quantiles(self.run_dist, n=4)
        return int(qs[0]), qs[1], int(qs[2]), max(self.run_dist)


@dataclass
class StarterLine:
    """Per-starter aggregate across their rotation starts."""
    name:        str
    starts:      int  = 0
    wins:        int  = 0
    # Pitching stats (opponent's batting events while this pitcher was active)
    pa_allowed:  int  = 0
    h_allowed:   int  = 0
    hr_allowed:  int  = 0
    bb_allowed:  int  = 0
    k_recorded:  int  = 0
    runs_allowed: int = 0
    outs_rec:    int  = 0   # outs recorded (K + Out outcomes)

    def record_event(self, ev, team_won: bool) -> None:
        self.pa_allowed  += 1
        o = ev.outcome
        if o in HITS:   self.h_allowed  += 1
        if o == "HR":   self.hr_allowed += 1
        if o == "BB":   self.bb_allowed += 1
        if o in ("K", "Out"): self.outs_rec += 1
        if o == "K":    self.k_recorded += 1
        self.runs_allowed += ev.runs_scored

    @property
    def ip_str(self) -> str:
        return f"{self.outs_rec // 3}.{self.outs_rec % 3}"

    @property
    def era(self) -> float:
        ip = self.outs_rec / 3
        return (self.runs_allowed / ip * 9) if ip > 0 else 0.0

    @property
    def k9(self) -> float:
        ip = self.outs_rec / 3
        return (self.k_recorded / ip * 9) if ip > 0 else 0.0

    @property
    def bb9(self) -> float:
        ip = self.outs_rec / 3
        return (self.bb_allowed / ip * 9) if ip > 0 else 0.0

    @property
    def whip(self) -> float:
        ip = self.outs_rec / 3
        return ((self.h_allowed + self.bb_allowed) / ip) if ip > 0 else 0.0

    def pg(self, total: int) -> float:
        return total / self.starts if self.starts else 0.0


# ── Simulation loop ───────────────────────────────────────────────────────────

def run_series(
    nya:        "TeamRoster",
    bos:        "TeamRoster",
    n:          int,
    seed_base:  str,
) -> tuple[Accum, Accum, list[dict], dict, dict]:
    """
    Simulate N games with full rotation cycling.

    Both teams cycle through their SP rotation:
        game i  →  rotation_index = i  →  starter = sp_rotation[i % len(rotation)]

    Returns:
        (away_accum, home_accum, game_level_list,
         away_starter_lines, home_starter_lines)
    where *_starter_lines is {pitcher_id: StarterLine}.
    """
    away_acc = Accum(nya.team_id)
    home_acc = Accum(bos.team_id)
    game_log: list[dict] = []
    away_starters: dict[str, StarterLine] = {}
    home_starters: dict[str, StarterLine] = {}

    report_every = max(1, n // 10)
    t0 = time.perf_counter()

    for i in range(n):
        away_dict = nya.for_game_engine(rotation_index=i)
        home_dict = bos.for_game_engine(rotation_index=i)

        box = simulate_game(away_dict, home_dict,
                            game_id=f"{seed_base}-{i:05d}",
                            verbose=False)

        away_runs = box.final_score["away"]
        home_runs = box.final_score["home"]
        away_won  = away_runs > home_runs

        # Find the game-opening starting pitcher for each side (first PA's pitcher)
        away_sp_id = away_dict["pitcher"].get("player_id", "?")
        home_sp_id = home_dict["pitcher"].get("player_id", "?")

        # Ensure StarterLine objects exist
        if away_sp_id not in away_starters:
            away_starters[away_sp_id] = StarterLine(away_dict["pitcher"].get("name", away_sp_id))
        if home_sp_id not in home_starters:
            home_starters[home_sp_id] = StarterLine(home_dict["pitcher"].get("name", home_sp_id))

        away_starters[away_sp_id].starts += 1
        home_starters[home_sp_id].starts += 1
        if away_won:
            away_starters[away_sp_id].wins += 1
        else:
            home_starters[home_sp_id].wins += 1

        for ev in box.pa_events:
            if ev.half == "top":
                away_acc.record_event(ev)
                # Away bats top → home pitcher pitching → credit/charge the home SP
                # We only attribute to the game-opener starter (first pitcher)
                if ev.pitcher_id == home_sp_id:
                    home_starters[home_sp_id].record_event(ev, away_won)
            else:
                home_acc.record_event(ev)
                if ev.pitcher_id == away_sp_id:
                    away_starters[away_sp_id].record_event(ev, away_won)

        away_acc.finalize_game(away_runs, away_won)
        home_acc.finalize_game(home_runs, not away_won)

        game_log.append({
            "innings": box.innings_played,
            "walkoff": box.walk_off,
            "extras":  box.innings_played > 9,
        })

        if (i + 1) % report_every == 0:
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            eta     = (n - i - 1) / rate
            print(f"  {i + 1:>6,} / {n:,}   {elapsed:>6.1f}s elapsed   "
                  f"~{eta:>5.1f}s remaining",
                  file=sys.stderr, flush=True)

    return away_acc, home_acc, game_log, away_starters, home_starters


# ── Report printing ───────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)


def print_report(
    away:          Accum,
    home:          Accum,
    game_log:      list[dict],
    n:             int,
    elapsed:       float,
    seed_base:     str,
    away_lineup:   list,
    home_lineup:   list,
    away_starters: dict,
    home_starters: dict,
) -> None:
    W = 72

    n_walkoff = sum(g["walkoff"] for g in game_log)
    n_extras  = sum(g["extras"]  for g in game_log)
    avg_inn   = statistics.mean(g["innings"] for g in game_log)
    avg_gpa   = (away.pa + home.pa) / n

    print(f"\n{'═' * W}")
    print(f"  {away.name}  vs  {home.name}  —  {n:,} Monte Carlo games")
    print(f"{'═' * W}")

    # ── Rosters ──────────────────────────────────────────────────────────────
    a_names = ", ".join(b["name"] for b in away_lineup[:5]) + ("…" if len(away_lineup) > 5 else "")
    h_names = ", ".join(b["name"] for b in home_lineup[:5]) + ("…" if len(home_lineup) > 5 else "")
    print(f"\n  {'STARTING LINEUP (1–5)':24}  {away.name:<28}  {home.name}")
    print(f"  {'':24}  {a_names:<28}  {h_names}")

    # ── Win / loss record ─────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'TEAM RECORD':24}  {'WIN %':>7}  {'W':>7}  {'L':>7}")
    print(f"  {'─' * 24}  {'─' * 7}  {'─' * 7}  {'─' * 7}")
    print(f"  {away.name:<24}  {away.win_pct:>7.1%}  {away.wins:>7,}  {n - away.wins:>7,}")
    print(f"  {home.name:<24}  {home.win_pct:>7.1%}  {home.wins:>7,}  {n - home.wins:>7,}")
    print()
    print(f"  Walk-off games   {n_walkoff:>6,}   ({n_walkoff / n:.1%})")
    print(f"  Extra-inn games  {n_extras:>6,}   ({n_extras  / n:.1%})")
    print(f"  Avg game length  {avg_inn:>6.2f}   innings  ·  {avg_gpa:.1f} PA/game")

    # ── Per-game batting averages ─────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'PER-GAME AVERAGES':24}  {'Runs':>6}  {'H':>5}  {'HR':>4}  "
          f"{'BB':>4}  {'K':>5}  {'Err':>4}  {'PA':>5}")
    print(f"  {'─' * 24}  {'─' * 6}  {'─' * 5}  {'─' * 4}  "
          f"{'─' * 4}  {'─' * 5}  {'─' * 4}  {'─' * 5}")
    for acc in (away, home):
        print(
            f"  {acc.name:<24}"
            f"  {acc.pg(acc.runs_total):>6.2f}"
            f"  {acc.pg(acc.h):>5.2f}"
            f"  {acc.pg(acc.hr):>4.2f}"
            f"  {acc.pg(acc.bb):>4.2f}"
            f"  {acc.pg(acc.k):>5.2f}"
            f"  {acc.pg(acc.err):>4.2f}"
            f"  {acc.pg(acc.pa):>5.1f}"
        )

    # ── Batting rates ─────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  {'BATTING RATES':24}  {'BA':>5}  {'OBP':>5}  {'K%':>6}  {'BB%':>6}  {'HR%':>6}")
    print(f"  {'─' * 24}  {'─' * 5}  {'─' * 5}  {'─' * 6}  {'─' * 6}  {'─' * 6}")
    for acc in (away, home):
        print(
            f"  {acc.name:<24}"
            f"  {acc.ba:>5.3f}"
            f"  {acc.obp:>5.3f}"
            f"  {acc.k_pct:>5.1%}"
            f"  {acc.bb_pct:>5.1%}"
            f"  {acc.hr_pct:>5.1%}"
        )

    # ── Run distribution histogram ────────────────────────────────────────────
    BAR_W   = 20
    MAX_R   = 16
    a_n     = len(away.run_dist)
    h_n     = len(home.run_dist)
    a_cnt: dict[int, int] = {}
    h_cnt: dict[int, int] = {}
    for r in away.run_dist:
        k = min(r, MAX_R); a_cnt[k] = a_cnt.get(k, 0) + 1
    for r in home.run_dist:
        k = min(r, MAX_R); h_cnt[k] = h_cnt.get(k, 0) + 1

    print(f"\n{'─' * W}")
    print(f"  RUNS/GAME DISTRIBUTION")
    print(f"  {'─' * (W - 2)}")
    print(f"  {'R':>4}  {away.name:<{BAR_W + 7}}  {home.name:<{BAR_W + 7}}")
    for r in range(0, MAX_R + 1):
        label = f"{MAX_R}+" if r == MAX_R else str(r)
        a_c   = a_cnt.get(r, 0)
        h_c   = h_cnt.get(r, 0)
        a_pct = a_c / a_n if a_n else 0.0
        h_pct = h_c / h_n if h_n else 0.0
        print(
            f"  {label:>4}  {_bar(a_pct, BAR_W)} {a_pct:>4.1%}"
            f"   {_bar(h_pct, BAR_W)} {h_pct:>4.1%}"
        )

    # ── Percentile summary ────────────────────────────────────────────────────
    print()
    for acc in (away, home):
        q1, med, q3, mx = acc.percentiles()
        print(
            f"  {acc.name:<24}"
            f"  Q1={q1}  median={med:.1f}  Q3={q3}"
            f"  avg={acc.pg(acc.runs_total):.2f}  max={mx}"
        )

    # ── Per-starter pitching breakdown ────────────────────────────────────────
    def _print_starter_table(team_name: str, starters: dict) -> None:
        if not starters:
            return
        print(f"\n{'─' * W}")
        print(f"  {team_name} ROTATION BREAKDOWN  (PA attributed to starter only)")
        print(f"  {'─' * (W - 2)}")
        print(
            f"  {'PITCHER':<22}  {'GS':>3}  {'W':>3}  {'IP':>6}  "
            f"{'ERA':>5}  {'WHIP':>5}  {'K/9':>5}  {'BB/9':>5}  {'HR':>4}"
        )
        print(
            f"  {'─' * 22}  {'─' * 3}  {'─' * 3}  {'─' * 6}  "
            f"{'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 4}"
        )
        for sl in sorted(starters.values(), key=lambda s: s.starts, reverse=True):
            print(
                f"  {sl.name:<22}"
                f"  {sl.starts:>3}"
                f"  {sl.wins:>3}"
                f"  {sl.ip_str:>6}"
                f"  {sl.era:>5.2f}"
                f"  {sl.whip:>5.2f}"
                f"  {sl.k9:>5.1f}"
                f"  {sl.bb9:>5.1f}"
                f"  {sl.hr_allowed:>4}"
            )

    _print_starter_table(away.name, away_starters)
    _print_starter_table(home.name, home_starters)

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  seed base: '{seed_base}'  ·  {elapsed:.1f}s  ·  {n / elapsed:.0f} games/sec")
    print(f"{'═' * W}\n")


# ── Roster loading ────────────────────────────────────────────────────────────

def _load_nya(use_dh: bool = True):
    """Try Supabase first; fall back to pilot JSON."""
    try:
        r = load_team(1927, "NYA", use_dh=use_dh)
        print("  NYA  →  Supabase (historical)", file=sys.stderr)
        return r
    except Exception as e:
        print(f"  NYA  →  pilot JSON  (Supabase: {e})", file=sys.stderr)
        return load_team_from_pilot(PILOT_JSON, team_id="1927-NYA", use_dh=use_dh)


def _load_bos(use_dh: bool = True):
    """Try Supabase; if it fails, abort with a helpful message."""
    try:
        r = load_team(2000, "BOS", use_dh=use_dh)
        print("  BOS  →  Supabase (historical)", file=sys.stderr)
        return r
    except Exception as e:
        print(f"\n  ERROR loading 2000 BOS from Supabase: {e}", file=sys.stderr)
        print(
            "  Run:  python3 ingest_historical.py --season 2000 --team BOS --push\n"
            "  to populate the database, then retry.",
            file=sys.stderr,
        )
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo series: 1927 NYA vs 2000 BOS"
    )
    parser.add_argument("--n",          type=int,   default=10_000,
                        help="Number of games to simulate (default: 10,000)")
    parser.add_argument("--seed-base",  type=str,   default="nya27-bos00",
                        help="Seed prefix for game IDs (default: 'nya27-bos00')")
    parser.add_argument("--no-dh",      action="store_true",
                        help="Use no-DH rules (pitcher bats 9th)")
    args = parser.parse_args()

    use_dh = not args.no_dh

    print(f"\n  Loading rosters …", file=sys.stderr)
    nya = _load_nya(use_dh)
    bos = _load_bos(use_dh)

    # Show rotation for both teams before the run
    def _show_rotation(label: str, roster) -> None:
        sp_rotation = [roster.pitcher] + [
            p for p in roster.bullpen if p.get("pitcher_role") == "SP"
        ]
        names = ", ".join(p["name"] for p in sp_rotation)
        print(f"  {label} rotation ({len(sp_rotation)} SP): {names}", file=sys.stderr)

    _show_rotation("NYA", nya)
    _show_rotation("BOS", bos)

    print(f"\n  Simulating {args.n:,} games …\n", file=sys.stderr)
    t0 = time.perf_counter()
    away_acc, home_acc, game_log, away_starters, home_starters = run_series(
        nya, bos, args.n, args.seed_base
    )
    elapsed = time.perf_counter() - t0

    print_report(
        away          = away_acc,
        home          = home_acc,
        game_log      = game_log,
        n             = args.n,
        elapsed       = elapsed,
        seed_base     = args.seed_base,
        away_lineup   = nya.lineup,
        home_lineup   = bos.lineup,
        away_starters = away_starters,
        home_starters = home_starters,
    )


if __name__ == "__main__":
    main()
