"""
ratings.py — Stage 3: Trait Conversion (PRD 02 — Global Z-Score Calibration)

Converts player rate stats into 20–99 traits using global z-score normalization
anchored to the full Lahman database (1871–2025).  This replaces the previous
era-relative sigmoid/log approach, which compressed historic outliers by
dividing against each era's own league average.

The z-score formula preserves the true magnitude of cross-era dominance:
  z     = (player_stat − global_mean) / global_std
  trait = clamp(50 + z × 16, 20, 99)

  16 pts/σ: a +3σ outlier hits trait 98 (near ceiling).
  Negative stats (hitter K%, pitcher BB%, pitcher HR%) invert z so that
  better performance always equals a higher trait number.

Trait → driver mapping:
  Hitters:  CON ← BABIP    POW ← ISO    EYE ← BB%    AK ← K% (inverted)
            GAP ← XBH relative (era-relative, unchanged — not a PRD core driver)
  Pitchers: STF ← K%       CTL ← BB% (inverted)     CMD ← HR% (inverted)
            STA ← IP/start (era-relative, unchanged — usage pattern, not rate)
"""
import json
import math
import datetime
from dataclasses import dataclass, field

from global_calibration import get_global_stats, z_score_trait, sigmoid_trait


BUILD_VERSION  = "2026.02"
SOURCE_VERSION = "lahman_1871-2025"

# Era baseline for STA (average IP per start).
# Dynamic per-year computation planned for a future version.
_AVG_IP_PER_START = 7.0


# ─── Legacy mapping (kept for audit comparison) ───────────────────────────────
# Logarithmic scale: 1.0x lg avg → 50, 2.5x → 90, 5.0x → 99 (clamped).
# Used only by trait_audit.py to show old vs new side-by-side.
_K_LEGACY = 40.0 / math.log(2.5)

# Cross-era POW baselines (legacy pipeline only).
HIST_AVG_HR_RATE = 0.0191
HIST_AVG_ISO     = 0.1256
HIST_AVG_XBH_PCT = 0.0671


def _map_to_trait_legacy(relative_score: float, trait_type: str = None) -> int:
    """
    Legacy era-relative mapping (PRD 02 v1).
    Kept for audit comparison only — NOT used by current card builders.
    Maps player_rate / lg_avg_rate to a 1–99 trait via log scale.
    """
    try:
        if relative_score is None or math.isnan(relative_score) or math.isinf(relative_score):
            return 50
    except TypeError:
        return 50
    if relative_score <= 0:
        return 1
    raw = 50.0 + _K_LEGACY * math.log(relative_score)
    return int(round(max(1.0, min(99.0, raw))))


# Keep public alias so existing callers don't break immediately.
map_to_trait = _map_to_trait_legacy


def _reliability(denominator: int, shrinkage: int = 300) -> float:
    """Empirical Bayes style reliability weight: 0.0–0.99 based on sample size."""
    return round(min(0.99, denominator / (denominator + shrinkage)), 2)


def _safe(val, fallback: float = float("nan")) -> float:
    """Returns val if it is a finite float, else fallback."""
    try:
        f = float(val)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


# ─── Player Card ─────────────────────────────────────────────────────────────

@dataclass
class PlayerCard:
    # Identity
    player_id:    str
    season:       int
    name:         str
    bats:         str
    throws:       str
    team_id:      str  = ""
    primary_role: str  = "Hitter"     # Hitter | Pitcher
    pitcher_role: str  = None         # SP | RP | None
    # Hitter traits
    CON: int = 50
    GAP: int = 50
    POW: int = 50
    EYE: int = 50
    AK:  int = 50
    BNT: int = 50
    # Pitcher traits
    STF: int = 50
    CTL: int = 50
    CMD: int = 50
    STA: int = 50
    # Fielder traits
    RNG: int = 50
    HND: int = 50
    ARM: int = 50
    # Provenance & audit
    normalized_rates: dict = field(default_factory=dict)
    reliability:      dict = field(default_factory=dict)
    build_version:    str  = BUILD_VERSION
    source_version:   str  = SOURCE_VERSION

    def trait_dict(self) -> dict:
        """Returns only the traits relevant to this card's primary role."""
        if self.primary_role == "Hitter":
            return {"CON": self.CON, "GAP": self.GAP, "POW": self.POW,
                    "EYE": self.EYE, "AK": self.AK,  "BNT": self.BNT}
        return {"STF": self.STF, "CTL": self.CTL,
                "CMD": self.CMD, "STA": self.STA}

    def to_engine_input(self) -> dict:
        """Returns the runtime dict consumed by PRD 01 PA Engine."""
        player_type = "batter" if self.primary_role == "Hitter" else "pitcher"
        return {
            "player_id":   self.player_id,
            "player_type": player_type,
            "handedness":  self.bats,
            "traits":      self.trait_dict(),
        }

    def print_card(self):
        w = 50
        role_line = f"{self.primary_role}"
        if self.pitcher_role:
            role_line += f" · {self.pitcher_role}"
        print(f"┌{'─' * w}┐")
        print(f"│  {'Chin Music Player Card':<{w-2}}│")
        print(f"├{'─' * w}┤")
        print(f"│  {self.name:<{w-2}}│")
        print(f"│  {self.player_id:<12} {self.season}  ·  {role_line:<{w-28}}│")
        print(f"│  Bats: {self.bats}  ·  Throws: {self.throws:<{w-18}}│")
        print(f"├{'─' * w}┤")
        if self.primary_role == "Hitter":
            print(f"│  CON {self.CON:>3}   GAP {self.GAP:>3}   POW {self.POW:>3}   "
                  f"EYE {self.EYE:>3}   AK {self.AK:>3}  │")
        else:
            print(f"│  STF {self.STF:>3}   CTL {self.CTL:>3}   CMD {self.CMD:>3}   "
                  f"STA {self.STA:>3}              │")
        print(f"└{'─' * w}┘")


# ─── Card Builders ────────────────────────────────────────────────────────────

def build_hitter_card(row, name: str, bats: str, throws: str, team_id: str = "") -> PlayerCard:
    """
    Builds a hitter PlayerCard using the hybrid era-adjusted + global z-score system.

    Trait drivers (PRD 02 Hybrid Calibration):
      POW ← iso_plus   = player_ISO / league_ISO    (era-adjusted power)
      EYE ← bb_plus    = player_BB% / league_BB%    (era-adjusted discipline)
      AK  ← k_plus_inv = league_K% / player_K%      (era-adjusted contact avoidance)
      CON ← babip_plus = player_BABIP / league_BABIP (era-adjusted contact quality)
      GAP ← xbh_plus   = player_XBH% / league_XBH%  (era-adjusted gap power)

    Plus-stat columns expected in row (from ingestion.py::normalize_player_stats):
      rel_bb (BB_plus), rel_k (K_plus), rel_babip (BABIP_plus),
      iso_plus, xbh_plus
    """
    gs = get_global_stats()
    gh = gs["hitter"]

    # Era-adjusted plus stats (player_rate / league_rate for this season)
    # rel_bb = BB% / lg_BB% = BB_plus (higher → better plate discipline)
    # rel_k  = K%  / lg_K%  = K_plus  (higher → worse contact avoidance → invert below)
    bb_plus    = _safe(row.get("rel_bb"),    1.0)   # BB_plus  (higher = better EYE)
    k_plus     = _safe(row.get("rel_k"),     1.0)   # K_plus   (higher = worse AK)
    babip_plus = _safe(row.get("rel_babip"), 1.0)   # BABIP_plus
    iso_plus   = _safe(row.get("iso_plus"),  1.0)   # ISO_plus
    xbh_plus   = _safe(row.get("xbh_plus"),  1.0)  # XBH_plus

    def _ok(v: float) -> float:
        return v if (math.isfinite(v) and v > 0) else 1.0

    bb_plus    = _ok(bb_plus)
    k_plus     = _ok(k_plus)
    babip_plus = _ok(babip_plus)
    iso_plus   = _ok(iso_plus)
    xbh_plus   = _ok(xbh_plus)

    # For AK: global distribution stores k_plus_inv = lg_K% / player_K% (higher=better).
    # rel_k gives K_plus = player_K% / lg_K% — invert it to match the stored distribution.
    k_plus_inv = 1.0 / k_plus  # now: higher value = lower K rate = better AK

    pa  = int(row.get("PA", 0))
    rel = _reliability(pa)
    hitter_traits = ["CON", "GAP", "POW", "EYE", "AK", "BNT"]

    card = PlayerCard(
        player_id    = str(row["playerID"]),
        season       = int(row["yearID"]),
        name         = name,
        bats         = str(bats),
        throws       = str(throws),
        team_id      = team_id,
        primary_role = "Hitter",
        POW = z_score_trait(iso_plus,   gh["iso_plus"]["mean"],   gh["iso_plus"]["std"]),
        EYE = z_score_trait(bb_plus,    gh["bb_plus"]["mean"],    gh["bb_plus"]["std"]),
        AK  = z_score_trait(k_plus_inv, gh["k_plus_inv"]["mean"], gh["k_plus_inv"]["std"]),
        CON = z_score_trait(babip_plus, gh["babip_plus"]["mean"], gh["babip_plus"]["std"]),
        GAP = z_score_trait(xbh_plus,   gh["xbh_plus"]["mean"],  gh["xbh_plus"]["std"]),
        normalized_rates = {
            "k_plus":     round(k_plus,     4),  # K_plus (player/league); lower = better AK
            "k_plus_inv": round(k_plus_inv, 4),  # lg/player; used for AK trait
            "bb_plus":    round(bb_plus,    4),
            "iso_plus":   round(iso_plus,   4),
            "babip_plus": round(babip_plus, 4),
            "xbh_plus":   round(xbh_plus,   4),
            "global_h_iso_mean":   round(gh["iso_plus"]["mean"],   4),
            "global_h_bb_mean":    round(gh["bb_plus"]["mean"],    4),
            "global_h_k_mean":     round(gh["k_plus_inv"]["mean"], 4),
            "global_h_babip_mean": round(gh["babip_plus"]["mean"], 4),
        },
        reliability = {t: rel for t in hitter_traits},
    )
    return card


def build_pitcher_card(row, name: str, bats: str, throws: str, team_id: str = "") -> PlayerCard:
    """
    Builds a pitcher PlayerCard using the hybrid era-adjusted + global z-score system.

    Trait drivers (PRD 02 Hybrid Calibration):
      STF ← k_plus      = pitcher_K% / league_K%    (era-adjusted strikeout dominance)
      CTL ← bb_plus_inv = league_BB% / pitcher_BB%  (era-adjusted walk suppression)
      CMD ← cmd_composite — blended:
              0.50 × era_ratio      (league_ERA / pitcher_ERA)
              0.30 × hr_plus_inv    (league_HR% / pitcher_HR%)
              0.20 × babip_pit_plus_inv (league_BABIP / pitcher_BABIP)
      STA ← IP/start vs era baseline (era-contextual, unchanged)

    Plus-stat columns expected in row (from ingestion.py::normalize_pitcher_stats):
      rel_p_k (K_plus), rel_p_bb (BB_plus),
      cmd_composite (pre-computed blend), rel_sta
    """
    gs = get_global_stats()
    gp = gs["pitcher"]

    # Era-adjusted plus stats
    # rel_p_k  = pitcher_K%  / league_K%  = K_plus  (higher → better STF)
    # rel_p_bb = pitcher_BB% / league_BB% = BB_plus  (lower  → better CTL)
    k_plus        = _safe(row.get("rel_p_k"),        1.0)  # K_plus (higher = better)
    bb_plus       = _safe(row.get("rel_p_bb"),       1.0)  # BB_plus (lower = better)
    cmd_composite = _safe(row.get("cmd_composite"),  1.0)  # blended ERA/HR/BABIP

    rel_sta = _safe(row.get("rel_sta"), 1.0)
    if not math.isfinite(rel_sta) or rel_sta <= 0:
        rel_sta = 1.0

    def _ok(v: float) -> float:
        return v if (math.isfinite(v) and v > 0) else 1.0

    k_plus        = _ok(k_plus)
    bb_plus       = _ok(bb_plus)
    cmd_composite = _ok(cmd_composite)

    # For CTL: global distribution stores bb_plus_inv = lg_BB% / pitcher_BB% (higher=better).
    # rel_p_bb gives BB_plus = pitcher_BB% / lg_BB% — invert to match stored distribution.
    bb_plus_inv = 1.0 / bb_plus  # higher value = lower BB rate = better CTL

    g      = int(row.get("G",  1))
    gs_val = int(row.get("GS", 0))
    pitcher_role = "SP" if gs_val / max(g, 1) >= 0.5 else "RP"

    bfp = int(row.get("BFP", 0))
    rel = _reliability(bfp, shrinkage=400)
    pitcher_traits = ["STF", "CTL", "CMD", "STA"]

    card = PlayerCard(
        player_id    = str(row["playerID"]),
        season       = int(row["yearID"]),
        name         = name,
        bats         = str(bats),
        throws       = str(throws),
        team_id      = team_id,
        primary_role = "Pitcher",
        pitcher_role = pitcher_role,
        STF = sigmoid_trait(k_plus,        gp["k_plus"]["mean"],        gp["k_plus"]["std"]),
        CTL = sigmoid_trait(bb_plus_inv,   gp["bb_plus_inv"]["mean"],   gp["bb_plus_inv"]["std"]),
        CMD = sigmoid_trait(cmd_composite, gp["cmd_composite"]["mean"], gp["cmd_composite"]["std"]),
        STA = _map_to_trait_legacy(rel_sta, "STA"),
        normalized_rates = {
            "k_plus":        round(k_plus,        4),
            "bb_plus":       round(bb_plus,        4),
            "cmd_composite": round(cmd_composite,  4),
            "rel_sta":       round(rel_sta,        3),
            "era_ratio":     round(_safe(row.get("era_ratio"),    1.0), 3),
            "hr_plus_inv":   round(_safe(row.get("hr_plus_inv"),  1.0), 3),
            "global_p_k_mean":   round(gp["k_plus"]["mean"],        4),
            "global_p_bb_mean":  round(gp["bb_plus_inv"]["mean"],   4),
            "global_p_cmd_mean": round(gp["cmd_composite"]["mean"], 4),
        },
        reliability = {t: rel for t in pitcher_traits},
    )
    return card


# ─── Export ──────────────────────────────────────────────────────────────────

def export_to_json(cards: list[PlayerCard], filename: str) -> None:
    """
    Exports a list of PlayerCards to JSON using the PRD 02 §6A schema.
    Includes trait_provenance for Live-proofing (§6A-i).
    """
    build_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    records = []

    for card in cards:
        record = {
            "player_id":    card.player_id,
            "card_id":      f"{card.player_id}|{card.season}",
            "season":       card.season,
            "team_id":      card.team_id,
            "name":         card.name,
            "bats":         card.bats,
            "throws":       card.throws,
            "primary_role": card.primary_role,
            "pitcher_role": card.pitcher_role,
            "traits":       card.trait_dict(),
            "normalized_rates": card.normalized_rates,
            "reliability":      card.reliability,
            "trait_provenance": {
                "mode":          "historical",
                "build_version": card.build_version,
                "source_version": card.source_version,
                "components": [
                    {"name": "season", "season": card.season, "weight": 1.0}
                ],
            },
            "build_version":    card.build_version,
            "source_version":   card.source_version,
            "build_timestamp":  build_timestamp,
        }
        records.append(record)

    with open(filename, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Exported {len(records)} cards → {filename}")
