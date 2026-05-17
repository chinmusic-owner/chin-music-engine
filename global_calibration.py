"""
global_calibration.py — Hybrid Trait Calibration (PRD 02 Fix v2)

TWO-STEP APPROACH:
  1. Era-adjust: divide each player's rate by that season's league average.
     This preserves relative dominance across eras (Grove at 2.85× his era
     and Pedro at 2.18× his era are both captured fairly).
  2. Global z-score: compute mean/std of the ERA-ADJUSTED plus stats across
     all 154 years, then z-score each player's plus stat against that global
     distribution.

Why this beats raw global z-scores:
  Raw z-scores crushed Lefty Grove (STF=54) because his absolute K% was
  modest even though he dominated his era by nearly 3x.  Era-relative ratios
  alone compressed Pedro because his K_plus (~2.1×) is lower than Grove's
  (~2.85×) when both are measured against their respective eras.  The hybrid
  finds the right answer: Pedro gets credit for elite absolute K rate AND
  relative era dominance; Grove gets credit for his unprecedented era
  dominance; Johnson gets recognition as a historically unmatched strikeout
  artist.

ROBUST STD COMPUTATION:
  Plus stats are clipped to [0.10, 3.50] before computing the global std.
  This prevents a handful of extreme dead-ball-era outliers (K_plus = 5×+)
  from inflating the std and making all modern pitchers look average.  Note:
  z-scores for individual players are computed from their ACTUAL (unclipped)
  plus stats, so true outliers still get extreme (clamped) traits.

PITCHER TRAIT DRIVERS:
  STF ← k_plus      = pitcher_K%  / league_K%    (era-adjusted K rate)
  CTL ← bb_plus_inv = league_BB%  / pitcher_BB%  (era-adjusted walk suppression)
  CMD ← cmd_composite — blended signal:
          0.50 × era_ratio       = league_ERA / pitcher_ERA
          0.30 × hr_plus_inv     = league_HR%  / pitcher_HR%
          0.20 × babip_plus_inv  = league_BABIP/ pitcher_BABIP
        (all three components centered near 1.0 = league avg;
         composite z-scored globally so Pedro's 1.74 ERA maps to CMD 99)

HITTER TRAIT DRIVERS:
  POW ← iso_plus    = player_ISO  / league_ISO    (era-adjusted power)
  EYE ← bb_plus     = player_BB%  / league_BB%    (era-adjusted walk rate)
  AK  ← k_plus_inv  = league_K%   / player_K%     (era-adjusted contact avoidance)
  CON ← babip_plus  = player_BABIP/ league_BABIP  (era-adjusted in-play hit rate)
  GAP ← xbh_plus    = player_XBH% / league_XBH%  (era-adjusted gap power)

TRAIT FORMULA (unchanged from PRD spec):
  z     = (plus_stat − global_mean) / global_std
  trait = clamp(50 + z × 16, 20, 99)

  16 pts/σ: a +3σ outlier hits ~98. A dead-ball era pitcher at 4× league
  average K rate is well into the 99-clamped zone — which is correct.
"""

import math
import os

import numpy as np
import pandas as pd

_LAHMAN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lahman_1871-2025_csv"
)

# Minimum sample thresholds
MIN_PA_HITTER  = 200
MIN_BF_PITCHER = 250

# Clip bounds for computing global std (prevents dead-ball outliers from
# inflating std and making all modern elite players look average).
# The actual z-score uses the real (unclipped) value — clipping only affects
# the calibration reference points, not individual trait values.
_CLIP_LO = 0.10
_CLIP_HI = 3.50

# Tiny floor to prevent division by zero in computed rates
_RATE_FLOOR = 1e-6

# Trait scale and bounds
SIGMA_SCALE = 16.0
TRAIT_FLOOR = 20
TRAIT_CEIL  = 99


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_batting(lahman_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(lahman_dir, "Batting.csv"))
    agg = [c for c in ["AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SF", "SH"]
           if c in df.columns]
    df = df.groupby(["playerID", "yearID"], as_index=False)[agg].sum()
    df["SF"]  = df["SF"].fillna(0)  if "SF"  in df.columns else 0
    df["HBP"] = df["HBP"].fillna(0) if "HBP" in df.columns else 0
    df["2B"]  = df["2B"].fillna(0)  if "2B"  in df.columns else 0
    df["3B"]  = df["3B"].fillna(0)  if "3B"  in df.columns else 0
    df["PA"]  = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    df["BIP"] = df["AB"] - df["SO"] - df["HR"] + df["SF"]
    return df


def _load_pitching(lahman_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(lahman_dir, "Pitching.csv"))
    agg = [c for c in ["G", "GS", "H", "HR", "BB", "SO", "HBP",
                        "IPouts", "BFP", "SF", "SH", "ER"]
           if c in df.columns]
    df = df.groupby(["playerID", "yearID"], as_index=False)[agg].sum()
    if "BFP" in df.columns and df["BFP"].notna().any():
        df["BF"] = df["BFP"].fillna(0)
    else:
        outs = df["IPouts"].fillna(0) if "IPouts" in df.columns else 0
        hbp  = df["HBP"].fillna(0)   if "HBP"   in df.columns else 0
        df["BF"] = outs + df["H"].fillna(0) + df["BB"].fillna(0) + hbp
    df["HBP"] = df["HBP"].fillna(0)
    df["ER"]  = df["ER"].fillna(0)  if "ER"  in df.columns else 0
    return df


# ---------------------------------------------------------------------------
# League-by-year context
# ---------------------------------------------------------------------------

def _compute_league_batting_by_year(bat: pd.DataFrame) -> pd.DataFrame:
    """Compute batting-side league rates per year from raw batting rows."""
    agg = bat.groupby("yearID", as_index=False).agg(
        AB=("AB", "sum"), H=("H", "sum"),
        _2B=("2B", "sum"), _3B=("3B", "sum"), HR=("HR", "sum"),
        BB=("BB", "sum"), SO=("SO", "sum"),
        HBP=("HBP", "sum"), SF=("SF", "sum"), BIP=("BIP", "sum"), PA=("PA", "sum"),
    )
    safe_pa  = agg["PA"].replace(0, np.nan)
    safe_ab  = agg["AB"].replace(0, np.nan)
    safe_bip = agg["BIP"].replace(0, np.nan)

    agg["lg_k_rate"]   = agg["SO"]  / safe_pa
    agg["lg_bb_rate"]  = agg["BB"]  / safe_pa
    agg["lg_hr_rate"]  = agg["HR"]  / safe_pa
    agg["lg_babip"]    = (agg["H"] - agg["HR"]) / safe_bip
    agg["lg_iso"]      = (agg["_2B"] + 2 * agg["_3B"] + 3 * agg["HR"]) / safe_ab
    agg["lg_xbh_rate"] = (agg["_2B"] + agg["_3B"]) / safe_pa
    return agg[["yearID", "lg_k_rate", "lg_bb_rate", "lg_hr_rate",
                "lg_babip", "lg_iso", "lg_xbh_rate"]]


def _compute_league_pitching_by_year(pit: pd.DataFrame) -> pd.DataFrame:
    """Compute pitching-side league rates per year from raw pitching rows."""
    agg = pit.groupby("yearID", as_index=False).agg(
        SO=("SO", "sum"), BB=("BB", "sum"), HR=("HR", "sum"),
        H=("H", "sum"), HBP=("HBP", "sum"),
        BF=("BF", "sum"), ER=("ER", "sum"), IPouts=("IPouts", "sum"),
    )
    safe_bf    = agg["BF"].replace(0, np.nan)
    safe_ipo   = agg["IPouts"].replace(0, np.nan)

    agg["lg_pit_k_rate"]  = agg["SO"] / safe_bf
    agg["lg_pit_bb_rate"] = agg["BB"] / safe_bf
    agg["lg_pit_hr_rate"] = agg["HR"] / safe_bf
    agg["lg_era"]         = agg["ER"] * 27.0 / safe_ipo

    # League BABIP against = (H - HR) / (BF - SO - BB - HBP - HR)
    bip_denom = (agg["BF"] - agg["SO"] - agg["BB"] - agg["HBP"] - agg["HR"])
    agg["lg_pit_babip"] = (agg["H"] - agg["HR"]) / bip_denom.replace(0, np.nan)

    return agg[["yearID", "lg_pit_k_rate", "lg_pit_bb_rate", "lg_pit_hr_rate",
                "lg_era", "lg_pit_babip"]]


# ---------------------------------------------------------------------------
# Plus-stat computation
# ---------------------------------------------------------------------------

def _safe_ratio(num: pd.Series, den: pd.Series, cap: float = _CLIP_HI) -> pd.Series:
    """Compute num/den, flooring denominator to avoid division by zero, capping result."""
    return (num / den.clip(lower=_RATE_FLOOR)).clip(upper=cap * 2)


def _compute_hitter_plus_stats(bat: pd.DataFrame,
                                lg_bat: pd.DataFrame) -> pd.DataFrame:
    """Merge each hitter-season with its league context and compute plus stats."""
    safe_pa  = bat["PA"].replace(0, np.nan)
    safe_ab  = bat["AB"].replace(0, np.nan)
    safe_bip = bat["BIP"].replace(0, np.nan)

    bat = bat.copy()
    bat["k_rate"]   = bat["SO"] / safe_pa
    bat["bb_rate"]  = bat["BB"] / safe_pa
    bat["hr_rate"]  = bat["HR"] / safe_pa
    bat["babip"]    = (bat["H"] - bat["HR"]) / safe_bip
    bat["iso"]      = (bat["2B"] + 2 * bat["3B"] + 3 * bat["HR"]) / safe_ab
    bat["xbh_rate"] = (bat["2B"] + bat["3B"]) / safe_pa

    bat = bat.merge(lg_bat, on="yearID", how="left")

    # Lower K% is better for a hitter → invert
    bat["k_plus_inv"]  = _safe_ratio(bat["lg_k_rate"],  bat["k_rate"])
    bat["bb_plus"]     = _safe_ratio(bat["bb_rate"],  bat["lg_bb_rate"])
    bat["iso_plus"]    = _safe_ratio(bat["iso"],      bat["lg_iso"])
    bat["babip_plus"]  = _safe_ratio(bat["babip"],    bat["lg_babip"])
    bat["xbh_plus"]    = _safe_ratio(bat["xbh_rate"], bat["lg_xbh_rate"])

    return bat


def _compute_pitcher_plus_stats(pit: pd.DataFrame,
                                 lg_pit: pd.DataFrame) -> pd.DataFrame:
    """Merge each pitcher-season with its league context and compute plus stats."""
    safe_bf  = pit["BF"].replace(0, np.nan)
    safe_ipo = pit["IPouts"].replace(0, np.nan)

    pit = pit.copy()
    pit["k_rate"]  = pit["SO"] / safe_bf
    pit["bb_rate"] = pit["BB"] / safe_bf
    pit["hr_rate"] = pit["HR"] / safe_bf
    pit["era"]     = pit["ER"] * 27.0 / safe_ipo

    # Pitcher BABIP against
    bip_denom = (pit["BF"] - pit["SO"] - pit["BB"] - pit["HBP"] - pit["HR"])
    pit["babip_pit"] = (pit["H"] - pit["HR"]) / bip_denom.replace(0, np.nan)

    pit = pit.merge(lg_pit, on="yearID", how="left")

    # Higher K rate vs league = better STF
    pit["k_plus"]         = _safe_ratio(pit["k_rate"],  pit["lg_pit_k_rate"])
    # Lower BB rate vs league = better CTL → invert
    pit["bb_plus_inv"]    = _safe_ratio(pit["lg_pit_bb_rate"], pit["bb_rate"])
    # CMD components: all "higher = better"
    pit["era_ratio"]      = _safe_ratio(pit["lg_era"],         pit["era"])
    pit["hr_plus_inv"]    = _safe_ratio(pit["lg_pit_hr_rate"],  pit["hr_rate"])
    pit["babip_plus_inv"] = _safe_ratio(pit["lg_pit_babip"],    pit["babip_pit"])

    # CMD composite: ERA-driven but tempered by HR and BABIP suppression
    pit["cmd_composite"] = (
        0.50 * pit["era_ratio"]      +
        0.30 * pit["hr_plus_inv"]    +
        0.20 * pit["babip_plus_inv"]
    )
    return pit


# ---------------------------------------------------------------------------
# Global statistics
# ---------------------------------------------------------------------------

def compute_global_stats(lahman_dir: str = None) -> dict:
    """
    Compute global mean/std for all era-adjusted plus stats.

    The global std is computed on CLIPPED values [0.10, 3.50] to prevent
    extreme dead-ball-era outliers from inflating the reference std and
    compressing modern elite players.  Individual trait z-scores use the
    player's real (unclipped) plus stat value, so true outliers still hit
    the ceiling (99) naturally.

    Returns nested dict:
    {
        "hitter":  { "k_plus_inv": {"mean": ..., "std": ...}, ... },
        "pitcher": { "k_plus": ..., "bb_plus_inv": ..., "cmd_composite": ... },
        "n_hitter_seasons":  int,
        "n_pitcher_seasons": int,
        # Also stores raw_z baselines for the audit comparison
        "raw_hitter": { "k_rate": ..., "bb_rate": ..., "iso": ..., "babip": ... },
        "raw_pitcher": { "k_rate": ..., "bb_rate": ..., "hr_rate": ... },
    }
    """
    d = lahman_dir or _LAHMAN_DIR

    bat_raw = _load_batting(d)
    pit_raw = _load_pitching(d)
    lg_bat  = _compute_league_batting_by_year(bat_raw)
    lg_pit  = _compute_league_pitching_by_year(pit_raw)

    # ── Hitters ──────────────────────────────────────────────────────────────
    bat = _compute_hitter_plus_stats(bat_raw, lg_bat)
    bat_q = bat[bat["PA"] >= MIN_PA_HITTER].copy()
    bat_q = bat_q.replace([np.inf, -np.inf], np.nan)
    bat_q = bat_q.dropna(subset=["k_plus_inv", "bb_plus", "iso_plus",
                                  "babip_plus", "xbh_plus"])

    def _stats(series: pd.Series, clip_hi: float = _CLIP_HI) -> dict:
        clipped = series.clip(_CLIP_LO, clip_hi)
        return {
            "mean": float(clipped.mean()),
            "std":  float(clipped.std(ddof=1)),
        }

    h_stats = {
        "k_plus_inv": _stats(bat_q["k_plus_inv"]),
        "bb_plus":    _stats(bat_q["bb_plus"]),
        "iso_plus":   _stats(bat_q["iso_plus"]),
        "babip_plus": _stats(bat_q["babip_plus"], clip_hi=2.0),
        "xbh_plus":   _stats(bat_q["xbh_plus"]),
    }
    n_h = len(bat_q)

    # Raw-z baselines (for audit comparison with PRD 02 v1 raw approach)
    raw_h = {
        "k_rate":  {"mean": float(bat_q["k_rate"].mean()),
                    "std":  float(bat_q["k_rate"].std(ddof=1))},
        "bb_rate": {"mean": float(bat_q["bb_rate"].mean()),
                    "std":  float(bat_q["bb_rate"].std(ddof=1))},
        "iso":     {"mean": float(bat_q["iso"].mean()),
                    "std":  float(bat_q["iso"].std(ddof=1))},
        "babip":   {"mean": float(bat_q["babip"].mean()),
                    "std":  float(bat_q["babip"].std(ddof=1))},
    }

    # ── Pitchers ─────────────────────────────────────────────────────────────
    pit = _compute_pitcher_plus_stats(pit_raw, lg_pit)
    pit_q = pit[pit["BF"] >= MIN_BF_PITCHER].copy()
    pit_q = pit_q.replace([np.inf, -np.inf], np.nan)
    pit_q = pit_q.dropna(subset=["k_plus", "bb_plus_inv", "cmd_composite"])

    p_stats = {
        "k_plus":       _stats(pit_q["k_plus"]),
        "bb_plus_inv":  _stats(pit_q["bb_plus_inv"]),
        "cmd_composite":_stats(pit_q["cmd_composite"]),
    }
    n_p = len(pit_q)

    raw_p = {
        "k_rate":  {"mean": float(pit_q["k_rate"].mean()),
                    "std":  float(pit_q["k_rate"].std(ddof=1))},
        "bb_rate": {"mean": float(pit_q["bb_rate"].mean()),
                    "std":  float(pit_q["bb_rate"].std(ddof=1))},
        "hr_rate": {"mean": float(pit_q["hr_rate"].mean()),
                    "std":  float(pit_q["hr_rate"].std(ddof=1))},
    }

    return {
        "hitter":            h_stats,
        "pitcher":           p_stats,
        "raw_hitter":        raw_h,
        "raw_pitcher":       raw_p,
        "n_hitter_seasons":  n_h,
        "n_pitcher_seasons": n_p,
    }


# ---------------------------------------------------------------------------
# Trait mapping
# ---------------------------------------------------------------------------

def z_score_trait(
    plus_stat: float,
    mean: float,
    std: float,
    invert: bool = False,
) -> int:
    """
    Maps an era-adjusted plus stat to a 20–99 trait via global z-score.
    Used for HITTER traits (σ×16 linear scale).

    Formula: trait = clamp(50 + z × 16, 20, 99)
    where    z     = (plus_stat − mean) / std   (negated when invert=True)

    Scale anchors:
      +3σ → trait  98  (dominant historic outlier)
      +2σ → trait  82  (elite)
      +1σ → trait  66  (above average)
       0σ → trait  50  (global average relative performer)
      −1σ → trait  34  (below average)
      −2σ → trait  18  → clamped to 20
    """
    if not std or std <= 0 or math.isnan(plus_stat) or math.isinf(plus_stat):
        return 50
    z = (plus_stat - mean) / std
    if invert:
        z = -z
    return int(round(max(float(TRAIT_FLOOR), min(float(TRAIT_CEIL), 50.0 + z * SIGMA_SCALE))))


# Sigmoid steepness constant for pitcher traits.
# Calibrated so that:
#   +1.5σ player (elite, e.g. Gibson 1968) → trait ≈ 86
#   +3.6σ player (transcendent, e.g. Pedro 2000) → trait ≈ 98
#   Average (z=0) → trait = 50 exactly
# This gives "Average→Great" gap >> "Great→Pedro" gap (desired non-linearity).
PITCHER_SIGMOID_K = 1.6


def sigmoid_trait(
    plus_stat: float,
    mean: float,
    std: float,
    invert: bool = False,
) -> int:
    """
    Maps an era-adjusted plus stat to a 20–99 trait via sigmoid (tanh) scaling.
    Used for PITCHER traits (STF, CTL, CMD).

    Formula: trait = clamp(50 + 49 × tanh(z / k), 20, 99)
    where k = PITCHER_SIGMOID_K = 1.6

    Why sigmoid for pitchers?
      Linear scaling can't simultaneously achieve:
        • Transcendent pitchers (z≈+3.6) in the 95-98 range
        • Elite pitchers (z≈+1.5) in the 85-90 range
        • Average pitcher (z=0) at 50
      The tanh curve is steep near z=0 (large lift per σ for above-average
      pitchers) and flattens near the ceiling (less marginal gain for
      historically unprecedented outliers), producing the desired shape:
        "Average→Great" gap > "Great→Pedro" gap.

    Approximate anchors (k=1.6):
      z = +3.6 (Pedro-tier) → trait ≈ 98
      z = +3.2 (Grove-tier) → trait ≈ 97
      z = +2.5 (Johnson-tier) → trait ≈ 95
      z = +1.5 (Gibson/deGrom) → trait ≈ 86
      z = +1.0 (good starter)  → trait ≈ 76
      z =  0.0 (league avg)    → trait = 50
      z = -1.0 (below avg)     → trait ≈ 24
    """
    if not std or std <= 0 or math.isnan(plus_stat) or math.isinf(plus_stat):
        return 50
    z = (plus_stat - mean) / std
    if invert:
        z = -z
    raw = 50.0 + 49.0 * math.tanh(z / PITCHER_SIGMOID_K)
    return int(round(max(float(TRAIT_FLOOR), min(float(TRAIT_CEIL), raw))))


# ---------------------------------------------------------------------------
# Per-player plus stat helpers (called at card-build time)
# ---------------------------------------------------------------------------

def pitcher_plus_stats(
    k_rate: float, bb_rate: float, hr_rate: float,
    era: float,
    lg_k_rate: float, lg_bb_rate: float, lg_hr_rate: float,
    lg_era: float, lg_babip: float,
    babip_pitch: float,
) -> dict:
    """
    Compute era-adjusted plus stats for a single pitcher-season.
    Called by ratings.py card builders at runtime.
    """
    def _r(n, d): return float(n) / max(float(d), _RATE_FLOOR)

    return {
        "k_plus":         _r(k_rate,    lg_k_rate),
        "bb_plus_inv":    _r(lg_bb_rate, bb_rate),
        "era_ratio":      _r(lg_era,    era) if era and era > 0 else 1.0,
        "hr_plus_inv":    _r(lg_hr_rate, hr_rate),
        "babip_plus_inv": _r(lg_babip,  babip_pitch) if babip_pitch and babip_pitch > 0 else 1.0,
        "cmd_composite":  (
            0.50 * _r(lg_era,    era)          if era and era > 0 else 0.50 +
            0.30 * _r(lg_hr_rate, hr_rate)     +
            0.20 * (_r(lg_babip, babip_pitch) if babip_pitch and babip_pitch > 0 else 1.0)
        ),
    }


def hitter_plus_stats(
    k_rate: float, bb_rate: float, iso: float, babip: float, xbh_rate: float,
    lg_k_rate: float, lg_bb_rate: float, lg_iso: float,
    lg_babip: float, lg_xbh_rate: float,
) -> dict:
    """
    Compute era-adjusted plus stats for a single hitter-season.
    Called by ratings.py card builders at runtime.
    """
    def _r(n, d): return float(n) / max(float(d), _RATE_FLOOR)

    return {
        "k_plus_inv": _r(lg_k_rate,  k_rate),
        "bb_plus":    _r(bb_rate,   lg_bb_rate),
        "iso_plus":   _r(iso,       lg_iso),
        "babip_plus": _r(babip,     lg_babip),
        "xbh_plus":   _r(xbh_rate,  lg_xbh_rate),
    }


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_CACHED: dict | None = None


def get_global_stats(lahman_dir: str = None) -> dict:
    """Returns cached hybrid global calibration stats (computed on first call)."""
    global _CACHED
    if _CACHED is None:
        _CACHED = compute_global_stats(lahman_dir)
    return _CACHED


def print_calibration_summary() -> None:
    """Prints the hybrid global baseline table."""
    gs = get_global_stats()
    h  = gs["hitter"]
    p  = gs["pitcher"]

    print(f"\n{'═' * 66}")
    print(f"  Hybrid Global Calibration Baselines  "
          f"(n_h={gs['n_hitter_seasons']:,}  n_p={gs['n_pitcher_seasons']:,})")
    print(f"  Plus stat = player_rate / league_rate_year  (era-adjusted)")
    print(f"  Global std computed on clipped [0.10, 3.50] distribution")
    print(f"{'─' * 66}")
    print(f"  {'Plus Stat':<26}  {'Mean':>7}  {'Std':>7}  {'−1σ':>7}  {'+1σ':>7}")
    print(f"{'─' * 66}")
    print("  HITTERS")
    for stat, label in [("k_plus_inv", "K_plus_inv (AK)"),
                         ("bb_plus",    "BB_plus (EYE)"),
                         ("iso_plus",   "ISO_plus (POW)"),
                         ("babip_plus", "BABIP_plus (CON)"),
                         ("xbh_plus",   "XBH_plus (GAP)")]:
        m, s = h[stat]["mean"], h[stat]["std"]
        print(f"    {label:<24}  {m:>7.4f}  {s:>7.4f}  {m-s:>7.4f}  {m+s:>7.4f}")
    print("  PITCHERS")
    for stat, label in [("k_plus",        "K_plus (STF)"),
                         ("bb_plus_inv",   "BB_plus_inv (CTL)"),
                         ("cmd_composite", "CMD composite")]:
        m, s = p[stat]["mean"], p[stat]["std"]
        print(f"    {label:<24}  {m:>7.4f}  {s:>7.4f}  {m-s:>7.4f}  {m+s:>7.4f}")
    print(f"{'═' * 66}\n")


if __name__ == "__main__":
    print_calibration_summary()
