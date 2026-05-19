"""
calibration_report.py — Cross-era trait and outcome calibration diagnostic

Loads all simulation team archetypes, prints their trait distributions with
cross-player z-scores, runs a calibration simulation, and reports outcome
statistics (K%, BB%, AVG, BABIP, contact-quality breakdown).

Teams (8 archetypes):
    1927 NYA  ·  1927 PIT                     (dead-ball / early power)
    1954 CLE  ·  1975 CIN  ·  1985 SLN        (post-war / speed / defense)
    2000 NYA  ·  2000 ARI  ·  2001 SEA        (modern power / high-K)

Usage:
    python3 calibration_report.py [--n 50] [--seed-base cal-2026]

--n is games per ordered matchup (n_teams*(n_teams-1) matchups × n = total).
Default 50 → 2,800 games total for 8 teams, runs in ~6s.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

from roster_manager import load_team, load_team_from_pilot, TeamRoster
from game_engine import simulate_game

PILOT_NYA27 = "pilot_1927_nya.json"
HITS        = {"Single", "Double", "Triple", "HR", "InfieldHit"}
BIP_SET     = {"Single", "Double", "Triple", "HR", "Out", "Error", "InfieldHit"}

_TB_VALS    = {"InfieldHit": 1, "Single": 1, "Double": 2, "Triple": 3, "HR": 4}


# ── Team definitions ──────────────────────────────────────────────────────────

TEAM_DEFS: list[tuple[int, str, str, str | None]] = [
    # Dead-ball / early power era
    (1927, "NYA", "1927 NYA", PILOT_NYA27),
    (1927, "PIT", "1927 PIT", None),
    # Post-war / integration / speed eras
    (1954, "CLE", "1954 CLE", None),
    (1975, "CIN", "1975 CIN", None),
    (1985, "SLN", "1985 SLN", None),
    # Modern power / high-K era
    (2000, "NYA", "2000 NYA", None),
    (2000, "ARI", "2000 ARI", None),
    (2001, "SEA", "2001 SEA", None),
]
ARI_FALLBACK = (2000, "NYN", "2000 NYN")


# ── Trait extraction ──────────────────────────────────────────────────────────

H_KEYS = ["CON", "POW", "EYE", "AK", "GAP", "SPD", "BIQ"]
P_KEYS = ["STF", "CTL", "CMD", "STA"]


def _all_hitter_traits(roster: TeamRoster) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for card in roster.lineup + roster.bench:
        if card.get("primary_role") == "Pitcher":
            continue
        t = card.get("traits", {})
        for k in H_KEYS:
            if k in t:
                out[k].append(float(t[k]))
    return dict(out)


def _all_pitcher_traits(roster: TeamRoster) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for card in ([roster.pitcher] + roster.bullpen):
        t = card.get("traits", {})
        for k in P_KEYS:
            if k in t:
                out[k].append(float(t[k]))
    return dict(out)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def _z(val: float, mu: float, sd: float) -> float:
    return (val - mu) / sd if sd > 0 else 0.0


# ── Outcome accumulator ───────────────────────────────────────────────────────

@dataclass
class OutcomeAccum:
    label: str
    pa:  int = 0
    ab:  int = 0
    h:   int = 0
    tb:  int = 0
    hr:  int = 0
    bb:  int = 0
    k:   int = 0
    hbp: int = 0
    bip: int = 0
    cq_weak:   int = 0
    cq_medium: int = 0
    cq_hard:   int = 0

    def record(self, ev) -> None:
        self.pa += 1
        o = ev.outcome
        is_hit = o in HITS
        if o not in ("BB", "HBP"):
            self.ab += 1
        if o == "BB":  self.bb  += 1
        if o == "HBP": self.hbp += 1
        if o == "K":   self.k   += 1
        if is_hit:
            self.h  += 1
            self.tb += _TB_VALS.get(o, 1)
        if o == "HR":  self.hr  += 1
        if o in BIP_SET and ev.raw_result:
            self.bip += 1
            cq = ((ev.raw_result or {}).get("contact") or {}).get("contact_quality", "")
            if   cq == "Weak":   self.cq_weak   += 1
            elif cq == "Medium": self.cq_medium += 1
            elif cq == "Hard":   self.cq_hard   += 1

    @property
    def avg(self)    -> float: return self.h / self.ab if self.ab else 0.0
    @property
    def obp(self)    -> float:
        return (self.h + self.bb + self.hbp) / self.pa if self.pa else 0.0
    @property
    def slg(self)    -> float: return self.tb / self.ab if self.ab else 0.0
    @property
    def ops(self)    -> float: return self.obp + self.slg
    @property
    def babip(self)  -> float:
        d = self.ab - self.k - self.hr
        return (self.h - self.hr) / d if d > 0 else 0.0
    @property
    def k_pct(self)  -> float: return self.k   / self.pa if self.pa else 0.0
    @property
    def bb_pct(self) -> float: return self.bb  / self.pa if self.pa else 0.0
    @property
    def hr_pct(self) -> float: return self.hr  / self.pa if self.pa else 0.0
    @property
    def cq_w(self)   -> float: return self.cq_weak   / self.bip if self.bip else 0.0
    @property
    def cq_m(self)   -> float: return self.cq_medium / self.bip if self.bip else 0.0
    @property
    def cq_h(self)   -> float: return self.cq_hard   / self.bip if self.bip else 0.0


# ── Calibration simulation ────────────────────────────────────────────────────

def run_calibration(
    rosters:       list[TeamRoster],
    labels:        list[str],
    n_per_matchup: int,
    seed_base:     str,
) -> dict[str, OutcomeAccum]:
    """Run n_per_matchup games for every ordered pair, accumulate PA events."""
    accums = {lbl: OutcomeAccum(lbl) for lbl in labels}
    pairs  = list(itertools.permutations(range(len(labels)), 2))
    total  = len(pairs) * n_per_matchup
    done   = 0
    t0     = time.perf_counter()

    for away_i, home_i in pairs:
        a_lbl = labels[away_i]
        h_lbl = labels[home_i]
        for i in range(n_per_matchup):
            away_d = rosters[away_i].for_game_engine(rotation_index=i)
            home_d = rosters[home_i].for_game_engine(rotation_index=i)
            gid    = f"{seed_base}-{a_lbl}-{h_lbl}-{i:04d}"
            box    = simulate_game(away_d, home_d, game_id=gid, verbose=False)
            for ev in box.pa_events:
                (accums[a_lbl] if ev.half == "top" else accums[h_lbl]).record(ev)
            done += 1
        elapsed = time.perf_counter() - t0
        rate    = done / elapsed if elapsed else 1
        print(
            f"  {done:>4}/{total}  {a_lbl} vs {h_lbl}  ({rate:.0f} g/s)",
            file=sys.stderr, flush=True,
        )
    return accums


# ── Report printing ───────────────────────────────────────────────────────────

def _sign(v: float) -> str:
    return f"({v:+.2f})" if v != 0.0 else "(    )"


def print_report(
    rosters:  list[TeamRoster],
    labels:   list[str],
    accums:   dict[str, OutcomeAccum],
    n_total:  int,
    elapsed:  float,
    seed_base: str,
) -> None:
    W = 96

    # ── Collect all individual trait values for cross-player z-scores ─────────
    all_h: dict[str, list[float]] = defaultdict(list)
    all_p: dict[str, list[float]] = defaultdict(list)
    h_by_team: dict[str, dict[str, list[float]]] = {}
    p_by_team: dict[str, dict[str, list[float]]] = {}

    for lbl, roster in zip(labels, rosters):
        hd = _all_hitter_traits(roster)
        pd = _all_pitcher_traits(roster)
        h_by_team[lbl] = hd
        p_by_team[lbl] = pd
        for k, vals in hd.items():
            all_h[k].extend(vals)
        for k, vals in pd.items():
            all_p[k].extend(vals)

    h_mu  = {k: _mean(v) for k, v in all_h.items()}
    h_sd  = {k: _std(v)  for k, v in all_h.items()}
    p_mu  = {k: _mean(v) for k, v in all_p.items()}
    p_sd  = {k: _std(v)  for k, v in all_p.items()}

    n_teams = len(labels)
    print(f"\n{'═' * W}")
    print(f"  CALIBRATION REPORT  —  {n_total:,} simulation games  ·  {n_teams} team archetypes")
    print(f"{'═' * W}")
    print()
    print("  z-scores are cross-player: (team avg − all-player mean) / all-player σ")
    print(f"  A z of +1.0 means the team avg is 1 std dev above the {n_teams}-team combined pool.")

    # ── Hitter traits ─────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  HITTER TRAITS   avg  (z)")
    col_w = 10
    hdr = f"  {'TEAM':<14}"
    for k in H_KEYS:
        hdr += f"  {k:>{col_w}}"
    hdr += f"  {'n':>3}"
    print(hdr)
    print(f"  {'─' * 14}" + f"  {'─' * col_w}" * len(H_KEYS) + "  ───")

    for lbl in labels:
        hd  = h_by_team[lbl]
        row = f"  {lbl:<14}"
        n   = len(hd.get("CON", []))
        for k in H_KEYS:
            vals = hd.get(k, [])
            if vals:
                mu = _mean(vals)
                z  = _z(mu, h_mu.get(k, mu), h_sd.get(k, 1.0))
                row += f"  {mu:>4.1f}{_sign(z):>6}"
            else:
                row += f"  {'—':>{col_w}}"
        row += f"  {n:>3}"
        print(row)

    # ── Pitcher traits ────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  PITCHER TRAITS   avg  (z)")
    hdr = f"  {'TEAM':<14}"
    for k in P_KEYS:
        hdr += f"  {k:>{col_w}}"
    hdr += f"  {'n':>3}"
    print(hdr)
    print(f"  {'─' * 14}" + f"  {'─' * col_w}" * len(P_KEYS) + "  ───")

    for lbl in labels:
        pd  = p_by_team[lbl]
        row = f"  {lbl:<14}"
        n   = len(pd.get("STF", []))
        for k in P_KEYS:
            vals = pd.get(k, [])
            if vals:
                mu = _mean(vals)
                z  = _z(mu, p_mu.get(k, mu), p_sd.get(k, 1.0))
                row += f"  {mu:>4.1f}{_sign(z):>6}"
            else:
                row += f"  {'—':>{col_w}}"
        row += f"  {n:>3}"
        print(row)

    # ── Batting outcome stats ─────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  BATTING OUTCOMES  (each team as the batting side, all opponents combined)")
    hdr = (
        f"  {'TEAM':<14}  {'AVG':>5}  {'OBP':>5}  {'SLG':>5}  "
        f"{'BABIP':>6}  {'K%':>6}  {'BB%':>6}  {'HR%':>6}  {'PA':>6}"
    )
    print(hdr)
    sep = (
        f"  {'─' * 14}  {'─' * 5}  {'─' * 5}  {'─' * 5}  "
        f"{'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 6}"
    )
    print(sep)

    # z-scores for outcome stats across the 4 teams
    oc_list = [accums[l] for l in labels]
    def oz(attr: str) -> tuple[float, float]:
        vals = [getattr(a, attr) for a in oc_list]
        return _mean(vals), _std(vals)

    avg_mu, avg_sd   = oz("avg")
    obp_mu, obp_sd   = oz("obp")
    slg_mu, slg_sd   = oz("slg")
    bab_mu, bab_sd   = oz("babip")
    k_mu,   k_sd     = oz("k_pct")
    bb_mu,  bb_sd    = oz("bb_pct")
    hr_mu,  hr_sd    = oz("hr_pct")

    print(sep)
    print(f"  {'TEAM':<14}  {'AVG (z)':>12}  {'OBP (z)':>12}  {'SLG (z)':>12}  {'BABIP (z)':>12}  {'K% (z)':>12}")
    print(f"  {'─' * 14}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
    for lbl in labels:
        a = accums[lbl]
        def fmtz(val: float, mu: float, sd: float, pct: bool = False) -> str:
            z = _z(val, mu, sd)
            v = f"{val:.1%}" if pct else f"{val:.3f}"
            return f"{v} ({z:+.2f})"
        print(
            f"  {lbl:<14}"
            f"  {fmtz(a.avg,   avg_mu, avg_sd):>12}"
            f"  {fmtz(a.obp,   obp_mu, obp_sd):>12}"
            f"  {fmtz(a.slg,   slg_mu, slg_sd):>12}"
            f"  {fmtz(a.babip, bab_mu, bab_sd):>12}"
            f"  {fmtz(a.k_pct, k_mu,   k_sd,  True):>12}"
        )

    # ── Contact quality distribution ──────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  CONTACT QUALITY  (% of balls in play)")
    print(
        f"  {'TEAM':<14}  {'WEAK':>8}  {'MEDIUM':>9}  {'HARD':>8}  "
        f"{'BIP':>6}  {'note'}"
    )
    print(f"  {'─' * 14}  {'─' * 8}  {'─' * 9}  {'─' * 8}  {'─' * 6}")
    for lbl in labels:
        a = accums[lbl]
        note = ""
        if   a.cq_h > 0.35: note = "← high hard-contact rate"
        elif a.cq_h < 0.15: note = "← low hard-contact rate"
        if   a.cq_w > 0.45: note += "  ← high weak-contact rate (contact problems)"
        print(
            f"  {lbl:<14}"
            f"  {a.cq_w:>7.1%}"
            f"  {a.cq_m:>9.1%}"
            f"  {a.cq_h:>8.1%}"
            f"  {a.bip:>6,}"
            f"  {note}"
        )

    # ── Cross-era comparison ──────────────────────────────────────────────────
    # Group teams by era decade for multi-era comparison
    era_groups: dict[str, list[str]] = {}
    for lbl in labels:
        year_str = lbl.split()[0]  # e.g. "1927" from "1927 NYA"
        decade   = year_str[:3] + "0s"  # "192" → "1920s", "200" → "2000s"
        era_groups.setdefault(decade, []).append(lbl)

    # Always show the 1927 vs latest-era comparison if both exist
    era27_labels = [l for l in labels if l.startswith("1927")]
    era_modern   = [l for l in labels if l.startswith("2000") or l.startswith("2001")]

    if len(era_groups) >= 2:
        print(f"\n{'─' * W}")
        print(f"  CROSS-ERA COMPARISON  (all era groups)")
        print()

        def era_avg_attr(lbls: list[str], attr: str) -> float:
            return sum(getattr(accums[l], attr) for l in lbls) / len(lbls)

        def era_trait_avg(lbls: list[str], by_team: dict, key: str) -> float:
            vals = []
            for lbl in lbls:
                v = by_team.get(lbl, {}).get(key, [])
                if v:
                    vals.append(_mean(v))
            return _mean(vals) if vals else 0.0

        # Build header from era groups sorted chronologically
        sorted_eras = sorted(era_groups.keys())
        col_w2 = 10
        hdr = f"  {'METRIC':<18}"
        for era in sorted_eras:
            hdr += f"  {era:>{col_w2}}"
        print(hdr)
        print(f"  {'─' * 18}" + f"  {'─' * col_w2}" * len(sorted_eras))

        metric_rows = [
            ("Hitter CON",  lambda lbls: era_trait_avg(lbls, h_by_team, "CON"),  False),
            ("Hitter POW",  lambda lbls: era_trait_avg(lbls, h_by_team, "POW"),  False),
            ("Hitter EYE",  lambda lbls: era_trait_avg(lbls, h_by_team, "EYE"),  False),
            ("Pitcher STF", lambda lbls: era_trait_avg(lbls, p_by_team, "STF"),  False),
            ("Pitcher CTL", lambda lbls: era_trait_avg(lbls, p_by_team, "CTL"),  False),
            ("Pitcher CMD", lambda lbls: era_trait_avg(lbls, p_by_team, "CMD"),  False),
            ("",            None, False),
            ("Batting AVG", lambda lbls: era_avg_attr(lbls, "avg"),   False),
            ("OBP",         lambda lbls: era_avg_attr(lbls, "obp"),   False),
            ("SLG",         lambda lbls: era_avg_attr(lbls, "slg"),   False),
            ("BABIP",       lambda lbls: era_avg_attr(lbls, "babip"), False),
            ("K%",          lambda lbls: era_avg_attr(lbls, "k_pct"), True),
            ("BB%",         lambda lbls: era_avg_attr(lbls, "bb_pct"),True),
            ("HR%",         lambda lbls: era_avg_attr(lbls, "hr_pct"),True),
            ("Hard CQ%",    lambda lbls: era_avg_attr(lbls, "cq_h"),  True),
            ("Weak CQ%",    lambda lbls: era_avg_attr(lbls, "cq_w"),  True),
        ]
        for metric, fn, pct in metric_rows:
            if not metric:
                print()
                continue
            row = f"  {metric:<18}"
            vals_for_era = []
            for era in sorted_eras:
                v = fn(era_groups[era])
                vals_for_era.append(v)
                fmt = f"{v:.1%}" if pct else f"{v:.3f}"
                row += f"  {fmt:>{col_w2}}"
            print(row)

        # Anchor comparison: earliest vs most-modern era
        if era27_labels and era_modern:
            print()
            print(f"  {'─' * (W // 2)}")
            print(f"  ANCHOR: 1927 avg vs Modern (2000s) avg")
            print()
            v27_avg = era_avg_attr(era27_labels, "avg")
            v00_avg = era_avg_attr(era_modern,   "avg")
            gap_pct = (v27_avg / v00_avg - 1) * 100 if v00_avg else 0
            print(f"  Batting AVG  —  1927: {v27_avg:.3f}  modern: {v00_avg:.3f}  "
                  f"gap: {gap_pct:+.1f}%"
                  + ("  ← within tolerance" if abs(gap_pct) < 8 else "  ⚠ still elevated"))
            stf27 = era_trait_avg(era27_labels, p_by_team, "STF")
            stf00 = era_trait_avg(era_modern,   p_by_team, "STF")
            print(f"  Pitcher STF  —  1927: {stf27:.1f}    modern: {stf00:.1f}    "
                  + ("  ✓ modern > 1927 (correct)" if stf00 > stf27 else "  ⚠ 1927 still higher"))

    # ── Diagnostic flags ──────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  DIAGNOSTIC FLAGS")
    print()

    flags: list[str] = []
    all_avgs = [accums[l].avg for l in labels]
    mu_avg   = _mean(all_avgs)
    sd_avg   = _std(all_avgs)
    all_kpcts = [accums[l].k_pct for l in labels]
    mu_k = _mean(all_kpcts)

    for lbl in labels:
        a  = accums[lbl]
        hd = h_by_team[lbl]
        pd = p_by_team[lbl]
        z_avg = _z(a.avg, mu_avg, sd_avg) if sd_avg else 0

        if z_avg > 1.5:
            flags.append(
                f"  ⚠ {lbl}: batting AVG z={z_avg:+.1f} — hitters significantly overrated "
                f"vs era pool. Check CON/EYE calibration in ingest_historical.py."
            )
        if z_avg < -1.5:
            flags.append(
                f"  ⚠ {lbl}: batting AVG z={z_avg:+.1f} — hitters significantly underrated "
                f"vs era pool. Verify ingest pulled correct season rows."
            )
        if a.k_pct > mu_k * 1.30 and hd.get("CON"):
            flags.append(
                f"  ⚠ {lbl}: K%={a.k_pct:.1%} ({a.k_pct/mu_k - 1:.0%} above mean) with "
                f"CON avg={_mean(hd['CON']):.1f} — high K% vs contact trait mismatch."
            )
        if pd.get("STF") and a.k_pct < mu_k * 0.70:
            stf_avg = _mean(pd["STF"])
            if stf_avg > _mean(all_p.get("STF", [50])):
                flags.append(
                    f"  ⚠ {lbl}: pitcher STF avg={stf_avg:.1f} but opponent K%={a.k_pct:.1%} "
                    f"is low — pitching may be underperforming relative to traits."
                )
        if a.cq_w > 0.48:
            flags.append(
                f"  ⚠ {lbl}: {a.cq_w:.1%} weak contact — batter CON/POW traits "
                f"producing too many soft outs. May be overpenalized in z-score calibration."
            )

    # Cross-era global flag
    def _era_trait_avg_diag(lbls: list[str], by_team: dict, key: str) -> float:
        vals = [_mean(by_team.get(l, {}).get(key, [])) for l in lbls
                if by_team.get(l, {}).get(key)]
        return _mean(vals) if vals else 0.0

    if era27_labels and era_modern:
        avg27 = _mean([accums[l].avg for l in era27_labels])
        avg00 = _mean([accums[l].avg for l in era_modern])
        con27 = _era_trait_avg_diag(era27_labels, h_by_team, "CON")
        con00 = _era_trait_avg_diag(era_modern,   h_by_team, "CON")
        stf27 = _era_trait_avg_diag(era27_labels, p_by_team, "STF")
        stf00 = _era_trait_avg_diag(era_modern,   p_by_team, "STF")

        if avg27 > avg00 * 1.08:
            flags.append(
                f"\n  ⚠ ERA CALIBRATION SKEW DETECTED:"
                f"\n    1927 batting AVG ({avg27:.3f}) > modern batting AVG ({avg00:.3f}) "
                f"by {(avg27/avg00 - 1)*100:.1f}%."
                f"\n    1927 CON avg={con27:.1f}  vs  modern CON avg={con00:.1f}."
                f"\n    This reflects genuine Murderers' Row / Waner-era contact dominance"
                f"\n    plus residual PA engine calibration (single K/BB curve across eras)."
            )
        if stf27 > stf00:
            flags.append(
                f"\n  ⚠ PITCHER STF INVERSION: 1927 avg STF ({stf27:.1f}) > "
                f"modern avg STF ({stf00:.1f})."
                f"\n    Expected: modern high-K era pitchers > dead-ball era pitchers."
                f"\n    Re-run reingest_roundrobin.py --push to correct pitcher calibration."
            )

    if flags:
        for f in flags:
            print(f)
    else:
        print("  ✓ No major calibration flags detected.")

    print(f"\n{'═' * W}")
    print(
        f"  {n_total:,} calibration games  ·  seed='{seed_base}'  ·  "
        f"{elapsed:.1f}s  ·  {n_total / elapsed:.0f} g/s"
    )
    print(f"{'═' * W}\n")


# ── Roster loader ─────────────────────────────────────────────────────────────

def _load(season: int, team: str, label: str, use_dh: bool,
          pilot: str | None = None) -> "TeamRoster | None":
    try:
        r = load_team(season, team, use_dh=use_dh)
        print(f"  {label}  →  Supabase", file=sys.stderr)
        return r
    except Exception as e:
        if pilot:
            print(f"  {label}  →  pilot JSON  (Supabase: {e})", file=sys.stderr)
            try:
                return load_team_from_pilot(pilot, team_id=label, use_dh=use_dh)
            except Exception as e2:
                print(f"           pilot failed: {e2}", file=sys.stderr)
        else:
            print(f"  {label}  →  FAILED  ({e})", file=sys.stderr)
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trait and outcome calibration report for 8 cross-era team archetypes"
    )
    parser.add_argument("--n",         type=int, default=20,
                        help="Games per ordered matchup (n_teams*(n_teams-1) pairs × n, default 20)")
    parser.add_argument("--seed-base", type=str, default="cal-2026")
    parser.add_argument("--no-dh",     action="store_true")
    args   = parser.parse_args()
    use_dh = not args.no_dh

    print("\n  Loading rosters …", file=sys.stderr)
    rosters: list = []
    labels:  list[str] = []

    for season, team, label, pilot in TEAM_DEFS:
        r = _load(season, team, label, use_dh, pilot)
        if r is None and team == "ARI":
            s2, t2, l2 = ARI_FALLBACK
            print(f"  Falling back to {l2} …", file=sys.stderr)
            r = _load(s2, t2, l2, use_dh)
            if r: label = l2
        if r:
            rosters.append(r)
            labels.append(label)

    if len(rosters) < 2:
        print("  ERROR: Need ≥2 teams.", file=sys.stderr)
        sys.exit(1)

    n_total = len(list(itertools.permutations(range(len(rosters)), 2))) * args.n
    print(
        f"\n  Running {n_total:,} calibration games "
        f"({len(rosters)} teams, {args.n} per matchup) …\n",
        file=sys.stderr,
    )

    t0     = time.perf_counter()
    accums = run_calibration(rosters, labels, args.n, args.seed_base)
    elapsed = time.perf_counter() - t0

    print_report(rosters, labels, accums, n_total, elapsed, args.seed_base)


if __name__ == "__main__":
    main()
