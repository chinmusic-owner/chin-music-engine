"""
trait_audit.py — THREE-WAY Trait Calibration Audit

Shows OLD (era-relative sigmoid/log), RAW-Z (pure global z-score on raw rates),
and HYBRID (era-adjusted plus stats globally z-scored) traits side by side
for six target players:

  Pedro Martinez  2000  — elite modern pitcher
  Lefty Grove     1927  — elite dead-ball/live-ball transition pitcher
  Walter Johnson  1913  — pre-live-ball era dominance
  Babe Ruth       1927  — all-time power outlier
  Lou Gehrig      1927  — consistent elite hitter
  Barry Bonds     2001  — peak modern power outlier

Success conditions (per PRD 02 Hybrid spec):
  • Pedro remains elite (STF 90+)
  • Grove / Johnson remain clearly elite (not average)
  • Ruth / Bonds remain elite power outliers
  • Trait distributions center near 50 overall

Run:
  python3 trait_audit.py
"""

import math
import os
import sys

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

LAHMAN_DIR = os.path.join(_HERE, "lahman_1871-2025_csv")

from global_calibration import get_global_stats, z_score_trait, sigmoid_trait

# ── Target players ───────────────────────────────────────────────────────────

PITCHER_TARGETS = [
    ("martipe02",  2000, "Pedro Martinez"),
    ("grovele01",  1927, "Lefty Grove"),
    ("johnswa01",  1913, "Walter Johnson"),
]

HITTER_TARGETS = [
    ("ruthba01",   1927, "Babe Ruth"),
    ("gehrilo01",  1927, "Lou Gehrig"),
    ("bondsba01",  2001, "Barry Bonds"),
]

# ── Lahman loaders ───────────────────────────────────────────────────────────

def _load_all():
    bat = pd.read_csv(os.path.join(LAHMAN_DIR, "Batting.csv"))
    pit = pd.read_csv(os.path.join(LAHMAN_DIR, "Pitching.csv"))
    bat = bat.fillna(0)
    pit = pit.fillna(0)

    # Aggregate multi-team seasons
    bat_agg_cols = [c for c in ["AB","H","2B","3B","HR","BB","SO","HBP","SF","SB","CS","IBB"]
                    if c in bat.columns]
    pit_agg_cols = [c for c in ["G","GS","H","HR","BB","SO","HBP","IPouts","BFP","ER"]
                    if c in pit.columns]

    bat = bat.groupby(["playerID","yearID"], as_index=False)[bat_agg_cols].sum()
    pit = pit.groupby(["playerID","yearID"], as_index=False)[pit_agg_cols].sum()

    return bat, pit


def _league_stats_year(bat, pit, year: int) -> dict:
    """Compute batting-side and pitching-side league averages for a given year."""
    b = bat[bat["yearID"] == year]
    p = pit[pit["yearID"] == year]

    # Batting side
    sf  = b["SF"].sum() if "SF" in b.columns else 0
    hbp = b["HBP"].sum() if "HBP" in b.columns else 0
    pa  = b["AB"].sum() + b["BB"].sum() + hbp + sf
    bip = b["AB"].sum() - b["SO"].sum() - b["HR"].sum() + sf
    ab  = b["AB"].sum()

    lg_k   = b["SO"].sum()  / max(pa, 1)
    lg_bb  = b["BB"].sum()  / max(pa, 1)
    lg_hr  = b["HR"].sum()  / max(pa, 1)
    lg_bab = (b["H"].sum() - b["HR"].sum()) / max(bip, 1)
    _2B = b["2B"].sum() if "2B" in b.columns else 0
    _3B = b["3B"].sum() if "3B" in b.columns else 0
    lg_iso = (_2B + 2 * _3B + 3 * b["HR"].sum()) / max(ab, 1)
    lg_xbh = (_2B + _3B) / max(pa, 1)

    # Pitching side
    phbp = p["HBP"].sum() if "HBP" in p.columns else 0
    if "BFP" in p.columns and p["BFP"].sum() > 0:
        pbf = p["BFP"].sum()
    else:
        pbf = p["IPouts"].sum() + p["H"].sum() + p["BB"].sum() + phbp

    pipo = p["IPouts"].sum()
    pbip = (pbf - p["SO"].sum() - p["BB"].sum() - phbp - p["HR"].sum())

    lg_pit_k  = p["SO"].sum() / max(pbf, 1)
    lg_pit_bb = p["BB"].sum() / max(pbf, 1)
    lg_pit_hr = p["HR"].sum() / max(pbf, 1)
    lg_era    = p["ER"].sum() * 27.0 / max(pipo, 1)
    lg_pit_bab= (p["H"].sum() - p["HR"].sum()) / max(pbip, 1)

    return {
        "lg_k": lg_k, "lg_bb": lg_bb, "lg_hr": lg_hr,
        "lg_babip": lg_bab, "lg_iso": lg_iso, "lg_xbh": lg_xbh,
        "lg_pit_k": lg_pit_k, "lg_pit_bb": lg_pit_bb, "lg_pit_hr": lg_pit_hr,
        "lg_era": lg_era, "lg_pit_babip": lg_pit_bab,
    }

# ── OLD trait logic (era-relative sigmoid/log) ──────────────────────────────

def _map_old(rel: float, kind: str) -> int:
    """
    Re-implementation of the legacy map_to_trait() logic from ratings.py v1.
    Mirrors the sigmoid/log formula used before hybrid calibration.
    """
    if rel <= 0 or not math.isfinite(rel):
        return 50
    log_rel = math.log(rel)
    if kind in ("STF", "AK"):
        trait = 50 + 43.65 * log_rel
    elif kind in ("CTL",):
        trait = 50 - 43.65 * log_rel
    elif kind == "CMD":
        trait = 50 - 34.0  * log_rel
    elif kind in ("CON", "EYE"):
        trait = 50 + 30.0  * log_rel
    elif kind == "POW":
        trait = 50 + 40.0  * log_rel
    elif kind == "GAP":
        trait = 50 + 38.0  * log_rel
    else:
        trait = 50 + 35.0  * log_rel
    return int(round(max(20.0, min(99.0, trait))))


def _old_pitcher(row, lg) -> dict:
    bf = float(row["BFP"]) if row["BFP"] > 0 else float(row["IPouts"]) + float(row["H"]) + float(row["BB"])
    k   = float(row["SO"]) / max(bf, 1)
    bb  = float(row["BB"]) / max(bf, 1)
    hr  = float(row["HR"]) / max(bf, 1)
    return {
        "STF": _map_old(k  / max(lg["lg_pit_k"],  1e-6), "STF"),
        "CTL": _map_old(bb / max(lg["lg_pit_bb"], 1e-6), "CTL"),
        "CMD": _map_old(hr / max(lg["lg_pit_hr"], 1e-6), "CMD"),
    }


def _old_hitter(row, lg) -> dict:
    pa  = float(row["AB"] + row["BB"] + row["HBP"] + row["SF"])
    bip = float(row["AB"] - row["SO"] - row["HR"] + row["SF"])
    ab  = float(row["AB"])
    k    = float(row["SO"])  / max(pa,  1)
    bb   = float(row["BB"])  / max(pa,  1)
    bab  = (float(row["H"]) - float(row["HR"])) / max(bip, 1)
    _2B  = float(row.get("2B", 0))
    _3B  = float(row.get("3B", 0))
    _HR  = float(row["HR"])
    iso  = (_2B + 2*_3B + 3*_HR) / max(ab, 1)
    xbh  = (_2B + _3B) / max(pa, 1)
    return {
        "CON": _map_old(bab / max(lg["lg_babip"], 1e-6), "CON"),
        "POW": _map_old(iso / max(lg["lg_iso"],   1e-6), "POW"),
        "EYE": _map_old(bb  / max(lg["lg_bb"],    1e-6), "EYE"),
        "AK":  _map_old(k   / max(lg["lg_k"],     1e-6), "AK"),   # inverted via _map_old CTL logic
        "GAP": _map_old(xbh / max(lg["lg_xbh"],   1e-6), "GAP"),
    }


# ── RAW-Z trait logic (pure global z-score, no era adjustment) ───────────────

def _raw_z_pitcher(row, gs) -> dict:
    gp = gs["raw_pitcher"]
    bf = float(row["BFP"]) if row["BFP"] > 0 else float(row["IPouts"]) + float(row["H"]) + float(row["BB"])
    k  = float(row["SO"]) / max(bf, 1)
    bb = float(row["BB"]) / max(bf, 1)
    hr = float(row["HR"]) / max(bf, 1)
    return {
        "STF": z_score_trait(k,  gp["k_rate"]["mean"],  gp["k_rate"]["std"]),
        "CTL": z_score_trait(bb, gp["bb_rate"]["mean"], gp["bb_rate"]["std"], invert=True),
        "CMD": z_score_trait(hr, gp["hr_rate"]["mean"], gp["hr_rate"]["std"], invert=True),
    }


def _raw_z_hitter(row, gs) -> dict:
    gh = gs["raw_hitter"]
    pa  = float(row["AB"] + row["BB"] + row["HBP"] + row["SF"])
    bip = float(row["AB"] - row["SO"] - row["HR"] + row["SF"])
    ab  = float(row["AB"])
    k   = float(row["SO"]) / max(pa,  1)
    bb  = float(row["BB"]) / max(pa,  1)
    bab = (float(row["H"]) - float(row["HR"])) / max(bip, 1)
    _2B = float(row.get("2B", 0))
    _3B = float(row.get("3B", 0))
    _HR = float(row["HR"])
    iso = (_2B + 2*_3B + 3*_HR) / max(ab, 1)
    xbh = (_2B + _3B) / max(pa, 1)
    return {
        "CON": z_score_trait(bab, gh["babip"]["mean"],   gh["babip"]["std"]),
        "POW": z_score_trait(iso, gh["iso"]["mean"],     gh["iso"]["std"]),
        "EYE": z_score_trait(bb,  gh["bb_rate"]["mean"], gh["bb_rate"]["std"]),
        "AK":  z_score_trait(k,   gh["k_rate"]["mean"],  gh["k_rate"]["std"],  invert=True),
        "GAP": z_score_trait(xbh, gh["babip"]["mean"],   gh["babip"]["std"]),  # proxy
    }


# ── HYBRID trait logic (era-adjusted plus stats, globally z-scored) ──────────

def _hybrid_pitcher(row, lg, gs) -> dict:
    _F = 1e-6
    gp = gs["pitcher"]
    bf = float(row["BFP"]) if row["BFP"] > 0 else float(row["IPouts"]) + float(row["H"]) + float(row["BB"])

    k  = float(row["SO"]) / max(bf, 1)
    bb = float(row["BB"]) / max(bf, 1)
    hr = float(row["HR"]) / max(bf, 1)

    ipo = float(row["IPouts"])
    er  = float(row["ER"])
    pit_era = (er * 27.0 / ipo) if ipo > 0 else lg["lg_era"]

    hbp = float(row["HBP"]) if "HBP" in row else 0.0
    bip_den = bf - float(row["SO"]) - float(row["BB"]) - hbp - float(row["HR"])
    pit_babip = ((float(row["H"]) - float(row["HR"])) / max(bip_den, _F)) if bip_den > 0 else lg["lg_pit_babip"]

    k_plus      = k  / max(lg["lg_pit_k"],   _F)
    bb_plus_inv = max(lg["lg_pit_bb"], _F) / max(bb, _F)
    era_ratio   = min(max(lg["lg_era"], _F) / max(pit_era, _F), 6.0)
    hr_plus_inv = min(max(lg["lg_pit_hr"], _F) / max(hr, _F), 6.0)
    bab_inv     = min(max(lg["lg_pit_babip"], _F) / max(pit_babip, _F), 3.0)
    cmd_comp    = 0.50 * era_ratio + 0.30 * hr_plus_inv + 0.20 * bab_inv

    stf = sigmoid_trait(k_plus,    gp["k_plus"]["mean"],        gp["k_plus"]["std"])
    ctl = sigmoid_trait(bb_plus_inv, gp["bb_plus_inv"]["mean"],  gp["bb_plus_inv"]["std"])
    cmd = sigmoid_trait(cmd_comp,  gp["cmd_composite"]["mean"], gp["cmd_composite"]["std"])

    return {
        "STF": stf, "CTL": ctl, "CMD": cmd,
        "_k_plus": round(k_plus, 3), "_bb_plus_inv": round(bb_plus_inv, 3),
        "_era_ratio": round(era_ratio, 3), "_hr_inv": round(hr_plus_inv, 3),
        "_bab_inv": round(bab_inv, 3), "_cmd_comp": round(cmd_comp, 3),
        "_pit_era": round(pit_era, 2), "_lg_era": round(lg["lg_era"], 2),
    }


def _hybrid_hitter(row, lg, gs) -> dict:
    _F = 1e-6
    gh = gs["hitter"]
    pa  = float(row["AB"] + row["BB"] + row["HBP"] + row["SF"])
    bip = float(row["AB"] - row["SO"] - row["HR"] + row["SF"])
    ab  = float(row["AB"])
    k   = float(row["SO"]) / max(pa,  1)
    bb  = float(row["BB"]) / max(pa,  1)
    bab = (float(row["H"]) - float(row["HR"])) / max(bip, 1)
    _2B = float(row.get("2B", 0))
    _3B = float(row.get("3B", 0))
    _HR = float(row["HR"])
    iso = (_2B + 2*_3B + 3*_HR) / max(ab, 1)
    xbh = (_2B + _3B) / max(pa, 1)

    bb_plus    = bb  / max(lg["lg_bb"],  _F)
    bab_plus   = bab / max(lg["lg_babip"], _F)
    iso_plus   = iso / max(lg["lg_iso"],  _F)
    xbh_plus   = xbh / max(lg["lg_xbh"], _F)
    k_plus_inv = max(lg["lg_k"], _F) / max(k, _F)  # inverted: lower K = higher AK

    pow_ = z_score_trait(iso_plus,   gh["iso_plus"]["mean"],   gh["iso_plus"]["std"])
    eye  = z_score_trait(bb_plus,    gh["bb_plus"]["mean"],    gh["bb_plus"]["std"])
    # k_plus_inv = lg_k / player_k (true inverse, higher = better AK). No invert flag.
    ak   = z_score_trait(k_plus_inv, gh["k_plus_inv"]["mean"], gh["k_plus_inv"]["std"])
    con  = z_score_trait(bab_plus,   gh["babip_plus"]["mean"], gh["babip_plus"]["std"])
    gap  = z_score_trait(xbh_plus,   gh["xbh_plus"]["mean"],  gh["xbh_plus"]["std"])

    return {
        "CON": con, "POW": pow_, "EYE": eye, "AK": ak, "GAP": gap,
        "_iso_plus": round(iso_plus, 3), "_bb_plus": round(bb_plus, 3),
        "_bab_plus": round(bab_plus, 3), "_xbh_plus": round(xbh_plus, 3),
        "_k_plus_inv": round(k_plus_inv, 3),
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _sigma(val: int) -> str:
    z = (val - 50) / 16.0
    return f"{z:+.1f}σ"


def _bar(val: int, width: int = 20) -> str:
    filled = int(round((val - 20) / 79 * width))
    return "█" * filled + "░" * (width - filled)


# ── Audit reporters ───────────────────────────────────────────────────────────

def print_pitcher_audit(bat, pit, gs):
    print(f"\n{'═'*86}")
    print(f"  PITCHER AUDIT — Old vs Raw-Z vs Hybrid")
    print(f"{'─'*86}")
    print(f"  {'Player / Season':<22}  {'Trait':<5}  {'OLD':>5}  {'RAW-Z':>6}  {'HYBRID':>7}  "
          f"{'Δ(H-O)':>7}  {'Δ(H-R)':>7}  {'σ(H)':>6}")
    print(f"{'─'*86}")

    for pid, yr, name in PITCHER_TARGETS:
        prow = pit[(pit["playerID"] == pid) & (pit["yearID"] == yr)]
        if prow.empty:
            print(f"  !! {name} {yr}: NOT FOUND in Pitching.csv (pid={pid})")
            continue
        prow = prow.iloc[0]

        lg = _league_stats_year(bat, pit, yr)

        old = _old_pitcher(prow, lg)
        raw = _raw_z_pitcher(prow, gs)
        hyb = _hybrid_pitcher(prow, lg, gs)

        print(f"\n  {name} {yr}")

        bf = int(prow["BFP"]) if prow["BFP"] > 0 else int(prow["IPouts"]) + int(prow["H"])
        so = int(prow["SO"])
        bb = int(prow["BB"])
        ipo= int(prow["IPouts"])
        er = int(prow["ER"])
        era_val = er * 27.0 / ipo if ipo > 0 else 0
        print(f"    BF={bf}  SO={so} ({so/max(bf,1)*100:.1f}%)  BB={bb} ({bb/max(bf,1)*100:.1f}%)  "
              f"ERA={era_val:.2f}  lg_ERA={lg['lg_era']:.2f}")
        print(f"    K_plus={hyb['_k_plus']:.3f}×  "
              f"BB_inv={hyb['_bb_plus_inv']:.3f}×  "
              f"ERA_ratio={hyb['_era_ratio']:.3f}×  "
              f"HR_inv={hyb['_hr_inv']:.3f}×  "
              f"BAB_inv={hyb['_bab_inv']:.3f}×  "
              f"CMD_comp={hyb['_cmd_comp']:.3f}")

        for trait in ["STF", "CTL", "CMD"]:
            o = old.get(trait, 50)
            r = raw.get(trait, 50)
            h = hyb.get(trait, 50)
            print(f"    {trait:<5}  {o:>5}  {r:>6}  {h:>7}  "
                  f"{h-o:>+7}  {h-r:>+7}  {_sigma(h):>6}  {_bar(h)}")

    print(f"\n{'═'*86}")


def print_hitter_audit(bat, pit, gs):
    print(f"\n{'═'*86}")
    print(f"  HITTER AUDIT — Old vs Raw-Z vs Hybrid")
    print(f"{'─'*86}")
    print(f"  {'Player / Season':<22}  {'Trait':<5}  {'OLD':>5}  {'RAW-Z':>6}  {'HYBRID':>7}  "
          f"{'Δ(H-O)':>7}  {'Δ(H-R)':>7}  {'σ(H)':>6}")
    print(f"{'─'*86}")

    for pid, yr, name in HITTER_TARGETS:
        brow = bat[(bat["playerID"] == pid) & (bat["yearID"] == yr)]
        if brow.empty:
            print(f"  !! {name} {yr}: NOT FOUND in Batting.csv (pid={pid})")
            continue
        brow = brow.iloc[0]

        lg = _league_stats_year(bat, pit, yr)

        old = _old_hitter(brow, lg)
        raw = _raw_z_hitter(brow, gs)
        hyb = _hybrid_hitter(brow, lg, gs)

        print(f"\n  {name} {yr}")
        pa  = brow["AB"] + brow["BB"] + brow["HBP"] + brow["SF"]
        ab  = brow["AB"]
        bip = brow["AB"] - brow["SO"] - brow["HR"] + brow["SF"]
        bab = (brow["H"] - brow["HR"]) / max(bip, 1)
        _2B = float(brow.get("2B", 0))
        _3B = float(brow.get("3B", 0))
        iso = (_2B + 2*_3B + 3*brow["HR"]) / max(float(ab), 1)
        xbh = (_2B + _3B) / max(float(pa), 1)
        print(f"    PA={pa}  K%={brow['SO']/max(pa,1)*100:.1f}%  BB%={brow['BB']/max(pa,1)*100:.1f}%  "
              f"ISO={iso:.3f}  BABIP={bab:.3f}  XBH%={xbh*100:.1f}%")
        print(f"    ISO_plus={hyb['_iso_plus']:.3f}×  "
              f"BB_plus={hyb['_bb_plus']:.3f}×  "
              f"BAB_plus={hyb['_bab_plus']:.3f}×  "
              f"XBH_plus={hyb['_xbh_plus']:.3f}×  "
              f"K_plus_inv={hyb['_k_plus_inv']:.3f}×")

        for trait in ["CON", "POW", "EYE", "AK", "GAP"]:
            o = old.get(trait, 50)
            r = raw.get(trait, 50)
            h = hyb.get(trait, 50)
            print(f"    {trait:<5}  {o:>5}  {r:>6}  {h:>7}  "
                  f"{h-o:>+7}  {h-r:>+7}  {_sigma(h):>6}  {_bar(h)}")

    print(f"\n{'═'*86}")


def print_distribution_check(bat, pit, gs):
    """Spot-check that the hybrid trait distribution centers near 50."""
    gp = gs["pitcher"]
    gh = gs["hitter"]
    print(f"\n{'═'*60}")
    print("  Hybrid Plus Stat Global Baselines (mean ± std)")
    print(f"{'─'*60}")
    print("  PITCHERS")
    for k, label in [("k_plus", "K_plus (STF)"),
                     ("bb_plus_inv", "BB_plus_inv (CTL)"),
                     ("cmd_composite", "CMD composite")]:
        m, s = gp[k]["mean"], gp[k]["std"]
        print(f"    {label:<26}  mean={m:.4f}  std={s:.4f}")
    print("  HITTERS")
    for k, label in [("iso_plus", "ISO_plus (POW)"),
                     ("bb_plus", "BB_plus (EYE)"),
                     ("k_plus_inv", "K_plus_inv (AK)"),
                     ("babip_plus", "BABIP_plus (CON)"),
                     ("xbh_plus", "XBH_plus (GAP)")]:
        m, s = gh[k]["mean"], gh[k]["std"]
        print(f"    {label:<26}  mean={m:.4f}  std={s:.4f}")
    n_h = gs.get("n_hitter_seasons", "?")
    n_p = gs.get("n_pitcher_seasons", "?")
    print(f"\n  Sample: {n_h:,} hitter-seasons  {n_p:,} pitcher-seasons")
    print(f"  (Global std computed on clipped [0.10, 3.50] distribution)")
    print(f"{'═'*60}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nLoading Lahman data and computing hybrid calibration…")
    bat, pit = _load_all()
    gs = get_global_stats()

    print_distribution_check(bat, pit, gs)
    print_pitcher_audit(bat, pit, gs)
    print_hitter_audit(bat, pit, gs)

    print("\nSUCCESS CONDITIONS CHECK:")
    print("  ✓ Pedro STF in 90s?  ← see STF row for Pedro above")
    print("  ✓ Grove / Johnson clearly elite (STF > 75)?  ← see above")
    print("  ✓ Ruth / Bonds elite power (POW > 90)?       ← see above")
    print("  ✓ Trait mean ≈ 50 overall?  ← see baseline table above")
    print()


if __name__ == "__main__":
    main()
