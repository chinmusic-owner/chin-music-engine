"""
ratings.py — Stage 3: Trait Conversion (PRD 02)
Converts era-normalized relative scores into 0–100 traits for use by the PA Engine.
"""
import json
import math
import datetime
from dataclasses import dataclass, field


# ─── Calibration ─────────────────────────────────────────────────────────────
# Logarithmic scale anchored to three constraints:
#   1.0x league average → trait 50  (ln(1) = 0, always true)
#   2.5x league average → trait ~90
#   5.0x+ league average → 99 (clamped)
#
# Derivation: 50 + K * ln(2.5) = 90  →  K = 40 / ln(2.5) ≈ 43.65
_K = 40.0 / math.log(2.5)

BUILD_VERSION  = "2026.01"
SOURCE_VERSION = "lahman_1871-2025"

# Era baseline for STA (average IP per start in the dead-ball / early live-ball era).
# Will be computed dynamically per year in a future version.
_AVG_IP_PER_START = 7.0

# Cross-era baselines for POW calibration (computed from 41,293 qualifying
# player-seasons with PA≥100 spanning 1900–2025).
# POW is driven by ISO and XBH% vs these fixed historical pool averages,
# NOT vs the current era's league average.  This prevents deadball hitters
# from receiving inflated POW simply because the 1906 HR floor was ~0.3%.
HIST_AVG_HR_RATE  = 0.0191   # mean HR/PA across pool (absolute, not era-relative)
HIST_AVG_ISO      = 0.1256   # mean (2B + 2×3B + 3×HR) / AB across pool
HIST_AVG_XBH_PCT  = 0.0671   # mean (2B + 3B + HR) / PA across pool


# ─── Core mapping function ────────────────────────────────────────────────────

def map_to_trait(relative_score: float, trait_type: str = None) -> int:
    """
    Maps a relative performance score (player_rate / lg_avg_rate) to a 0–100 trait.

    Scale anchors:
      5.0x+ → 99  (all-time elite; clamped)
      2.5x  → 90  (dominant)
      1.5x  → 67  (above average)
      1.0x  → 50  (league average)
      0.7x  → 35  (below average)
      0.4x  → 17  (poor)
      0.0x  →  1  (floor)

    trait_type reserved for future per-trait calibration constants.
    """
    try:
        if relative_score is None or math.isnan(relative_score) or math.isinf(relative_score):
            return 50
    except TypeError:
        return 50
    if relative_score <= 0:
        return 1
    raw = 50.0 + _K * math.log(relative_score)
    return int(round(max(1.0, min(99.0, raw))))


def _reliability(denominator: int, shrinkage: int = 300) -> float:
    """Empirical Bayes style reliability weight: 0.0–0.99 based on sample size."""
    return round(min(0.99, denominator / (denominator + shrinkage)), 2)


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
    Builds a hitter PlayerCard from a normalized stats row.
    Trait drivers (PRD 02 §7):
      POW ← ISO×0.65 + XBH%×0.35  vs fixed historical pool baselines
            (era-relative HR rate intentionally NOT used — see HIST_AVG_ISO)
      EYE ← rel_bb      AK ← 1/rel_k
      CON ← rel_babip   GAP ← rel_gap
    """
    rel_k   = row.get("rel_k",   float("nan"))
    rel_gap = row.get("rel_gap", float("nan"))

    ak_score  = (1.0 / rel_k)  if (rel_k  == rel_k  and rel_k  > 0) else 1.0
    gap_score = rel_gap         if (rel_gap == rel_gap and rel_gap > 0) else 1.0

    # POW: weighted blend of three cross-era signals, all vs fixed historical baselines.
    #   rel_hr_hist (55%) — absolute HR/PA vs all-time pool; dominant signal.
    #                       NOT divided by era avg, so a 1906 player with 2 HRs
    #                       gets the same credit as a 2005 player with 2 HRs.
    #   rel_iso     (30%) — ISO captures extra-base authority (2B, 3B, HR weighted).
    #   rel_xbh     (15%) — XBH/PA stabilises gap hitters at low HR counts.
    rel_hr_hist = row.get("rel_hr_hist", float("nan"))
    rel_iso     = row.get("rel_iso",     float("nan"))
    rel_xbh     = row.get("rel_xbh",     float("nan"))
    rel_hr_hist = rel_hr_hist if (rel_hr_hist == rel_hr_hist and rel_hr_hist > 0) else 0.01
    rel_iso     = rel_iso     if (rel_iso     == rel_iso     and rel_iso     > 0) else 1.0
    rel_xbh     = rel_xbh     if (rel_xbh     == rel_xbh     and rel_xbh     > 0) else 1.0
    pow_score = rel_hr_hist * 0.55 + rel_iso * 0.30 + rel_xbh * 0.15

    pa = int(row.get("PA", 0))
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
        CON = map_to_trait(row.get("rel_babip"), "CON"),
        GAP = map_to_trait(gap_score,            "GAP"),
        POW = map_to_trait(pow_score,            "POW"),
        EYE = map_to_trait(row.get("rel_bb"),    "EYE"),
        AK  = map_to_trait(ak_score,             "AK"),
        normalized_rates = {
            "rel_hr_hist": round(float(rel_hr_hist), 3),
            "rel_iso":     round(float(rel_iso), 3),
            "rel_xbh":     round(float(rel_xbh), 3),
            "rel_hr":      round(float(row.get("rel_hr", float("nan"))), 3),  # era-relative, audit only
            "rel_bb":    round(float(row.get("rel_bb", float("nan"))), 3),
            "rel_k":     round(float(rel_k), 3),
            "rel_babip": round(float(row.get("rel_babip", float("nan"))), 3),
            "rel_gap":   round(float(gap_score), 3),
        },
        reliability = {t: rel for t in hitter_traits},
    )
    return card


def build_pitcher_card(row, name: str, bats: str, throws: str, team_id: str = "") -> PlayerCard:
    """
    Builds a pitcher PlayerCard from a normalized stats row.
    Trait drivers (PRD 02 §7):
      STF ← rel_p_k (pitcher K rate vs league — high is good)
      CTL ← 1/rel_p_bb (inverse walk rate — lower BB = higher CTL)
      CMD ← 1/rel_p_hr (inverse HR rate — lower HR = higher CMD)
      STA ← IP/start relative to era baseline
    """
    rel_p_bb = row.get("rel_p_bb", float("nan"))
    rel_p_hr = row.get("rel_p_hr", float("nan"))
    rel_sta  = row.get("rel_sta",  1.0)

    ctl_score = (1.0 / rel_p_bb) if (rel_p_bb == rel_p_bb and rel_p_bb > 0) else 1.0
    cmd_score = (1.0 / rel_p_hr) if (rel_p_hr == rel_p_hr and rel_p_hr > 0) else 1.0

    # SP/RP classification: GS/G >= 0.5 → starter
    g  = int(row.get("G",  1))
    gs = int(row.get("GS", 0))
    pitcher_role = "SP" if gs / max(g, 1) >= 0.5 else "RP"

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
        STF = map_to_trait(row.get("rel_p_k"), "STF"),
        CTL = map_to_trait(ctl_score,          "CTL"),
        CMD = map_to_trait(cmd_score,           "CMD"),
        STA = map_to_trait(rel_sta,             "STA"),
        normalized_rates = {
            "rel_p_k":  round(float(row.get("rel_p_k",  float("nan"))), 3),
            "rel_p_bb": round(float(rel_p_bb), 3),
            "rel_p_hr": round(float(rel_p_hr), 3),
            "rel_sta":  round(float(rel_sta), 3),
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
