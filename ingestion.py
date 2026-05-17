import pandas as pd
import os

LAHMAN_DIR = os.path.join(os.path.dirname(__file__), "lahman_1871-2025_csv")

BATTING_COLS  = ["playerID", "yearID", "teamID", "AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SF"]
PITCHING_COLS = ["playerID", "yearID", "teamID", "G", "GS", "IPouts", "BFP", "H", "HR", "BB", "SO", "HBP", "ER"]
PEOPLE_COLS   = ["playerID", "nameFirst", "nameLast", "bats", "throws"]


class LahmanAdapter:

    def load_batting(self, team_id: str, year: int) -> pd.DataFrame:
        path = os.path.join(LAHMAN_DIR, "Batting.csv")
        df = pd.read_csv(path)
        df = df[(df["teamID"] == team_id) & (df["yearID"] == year)]
        df = df[BATTING_COLS].copy()
        return df.reset_index(drop=True)

    def load_pitching(self, team_id: str, year: int) -> pd.DataFrame:
        path = os.path.join(LAHMAN_DIR, "Pitching.csv")
        df = pd.read_csv(path)
        df = df[(df["teamID"] == team_id) & (df["yearID"] == year)]
        df = df[PITCHING_COLS].copy()
        return df.reset_index(drop=True)

    def load_people(self) -> pd.DataFrame:
        path = os.path.join(LAHMAN_DIR, "People.csv")
        df = pd.read_csv(path, usecols=PEOPLE_COLS)
        return df.reset_index(drop=True)

    def load_league_context(self, year: int) -> dict:
        """
        Aggregates team-level counting stats across all teams in a given year
        to produce league-wide rate baselines used for era normalization.
        SF is filled with 0 for years where it wasn't tracked.

        Extended for hybrid calibration (PRD 02 Fix v2):
          lg_avg_iso      — league ISO (for era-adjusted POW)
          lg_pit_k_rate   — pitcher-side K% (from Pitching.csv)
          lg_pit_bb_rate  — pitcher-side BB%
          lg_pit_hr_rate  — pitcher-side HR%
          lg_era          — league ERA (for era-adjusted CMD)
          lg_pit_babip    — league BABIP against (for era-adjusted CMD)
        """
        path = os.path.join(LAHMAN_DIR, "Teams.csv")
        df = pd.read_csv(path)
        df = df[df["yearID"] == year].copy()
        df["SF"]  = df["SF"].fillna(0)
        df["HBP"] = df["HBP"].fillna(0)
        df["2B"]  = df["2B"].fillna(0) if "2B" in df.columns else 0
        df["3B"]  = df["3B"].fillna(0) if "3B" in df.columns else 0

        t = df[["AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SF"]].sum()

        lg_pa  = t["AB"] + t["BB"] + t["HBP"] + t["SF"]
        lg_bip = t["AB"] - t["SO"] - t["HR"] + t["SF"]

        context = {
            "year":             year,
            "lg_pa":            lg_pa,
            "lg_avg_k_rate":    t["SO"]  / lg_pa,
            "lg_avg_bb_rate":   t["BB"]  / lg_pa,
            "lg_avg_hr_rate":   t["HR"]  / lg_pa,
            "lg_avg_babip":     (t["H"] - t["HR"]) / max(float(lg_bip), 1.0),
            "lg_avg_xbh_rate":  (t["2B"] + t["3B"]) / lg_pa,
            "lg_avg_iso":       (t["2B"] + 2 * t["3B"] + 3 * t["HR"]) / max(float(t["AB"]), 1.0),
        }

        # Pitcher-side league stats from Pitching.csv
        try:
            pit_path = os.path.join(LAHMAN_DIR, "Pitching.csv")
            pit = pd.read_csv(pit_path)
            pit = pit[pit["yearID"] == year].copy()
            pit["HBP"] = pit["HBP"].fillna(0) if "HBP" in pit.columns else 0
            pit["ER"]  = pit["ER"].fillna(0)  if "ER"  in pit.columns else 0

            if "BFP" in pit.columns and pit["BFP"].notna().any():
                pit["BF"] = pit["BFP"].fillna(0)
            else:
                outs = pit["IPouts"].fillna(0) if "IPouts" in pit.columns else 0
                pit["BF"] = outs + pit["H"].fillna(0) + pit["BB"].fillna(0) + pit["HBP"]

            p = pit[["SO", "BB", "HR", "H", "BF", "ER", "IPouts", "HBP"]].sum()
            safe_bf  = max(float(p["BF"]),    1.0)
            safe_ipo = max(float(p["IPouts"]), 1.0)

            bip_den = float(p["BF"]) - float(p["SO"]) - float(p["BB"]) - float(p["HBP"]) - float(p["HR"])
            bip_den = max(bip_den, 1.0)

            context["lg_pit_k_rate"]  = float(p["SO"]) / safe_bf
            context["lg_pit_bb_rate"] = float(p["BB"]) / safe_bf
            context["lg_pit_hr_rate"] = float(p["HR"]) / safe_bf
            context["lg_era"]         = float(p["ER"])  * 27.0 / safe_ipo
            context["lg_pit_babip"]   = (float(p["H"]) - float(p["HR"])) / bip_den
        except Exception:
            # Fallback to batting-side proxies if Pitching.csv unavailable
            context["lg_pit_k_rate"]  = context["lg_avg_k_rate"]
            context["lg_pit_bb_rate"] = context["lg_avg_bb_rate"]
            context["lg_pit_hr_rate"] = context["lg_avg_hr_rate"]
            context["lg_era"]         = 4.00
            context["lg_pit_babip"]   = context["lg_avg_babip"]

        return context


def normalize_pitcher_stats(pitcher_df: pd.DataFrame, league_context: dict) -> pd.DataFrame:
    """
    Computes per-pitcher rate stats and era-adjusted plus stats for hybrid calibration.

    Plus stats (all → higher = better):
      rel_p_k   = p_k_rate  / lg_pit_k_rate   (K_plus — STF driver)
      rel_p_bb  = p_bb_rate / lg_pit_bb_rate   (BB_plus — CTL driver, inverted in card builder)
      rel_p_hr  = p_hr_rate / lg_pit_hr_rate   (HR_plus — CMD component, inverted)
      era_ratio = lg_era    / p_era            (ERA-ratio — primary CMD driver)
      babip_pit_plus_inv = lg_pit_babip / p_babip_pit   (CMD component)
      cmd_composite = 0.50*era_ratio + 0.30*(1/rel_p_hr) + 0.20*babip_pit_plus_inv

    STA uses IP/start relative to a fixed era baseline (_AVG_IP_PER_START in ratings.py).
    """
    from ratings import _AVG_IP_PER_START

    _FLOOR = 1e-6

    df = pitcher_df.copy()
    df["BFP"]    = df["BFP"].fillna(0)
    df["HBP"]    = df["HBP"].fillna(0)
    df["ER"]     = df["ER"].fillna(0) if "ER" in df.columns else 0.0
    df["IPouts"] = df["IPouts"].fillna(0)
    df["IP"]     = df["IPouts"] / 3.0

    safe_bfp = df["BFP"].replace(0, float("nan"))
    safe_ipo = df["IPouts"].replace(0, float("nan"))

    df["p_k_rate"]  = df["SO"] / safe_bfp
    df["p_bb_rate"] = df["BB"] / safe_bfp
    df["p_hr_rate"] = df["HR"] / safe_bfp
    df["p_era"]     = df["ER"] * 27.0 / safe_ipo

    # Pitcher BABIP against = (H - HR) / (BF - SO - BB - HBP - HR)
    bip_den = (df["BFP"] - df["SO"] - df["BB"] - df["HBP"] - df["HR"]).replace(0, float("nan"))
    df["p_babip_pit"] = (df["H"] - df["HR"]) / bip_den

    # Pitcher-side league context (falls back to batting-side if missing)
    lg_k  = league_context.get("lg_pit_k_rate",  league_context["lg_avg_k_rate"])
    lg_bb = league_context.get("lg_pit_bb_rate", league_context["lg_avg_bb_rate"])
    lg_hr = league_context.get("lg_pit_hr_rate", league_context["lg_avg_hr_rate"])
    lg_era= league_context.get("lg_era",         4.00)
    lg_bp = league_context.get("lg_pit_babip",   league_context["lg_avg_babip"])

    df["rel_p_k"]  = df["p_k_rate"]  / max(lg_k,  _FLOOR)
    df["rel_p_bb"] = df["p_bb_rate"] / max(lg_bb, _FLOOR)
    df["rel_p_hr"] = df["p_hr_rate"] / max(lg_hr, _FLOOR)

    # ERA ratio: higher = pitcher much better than league average
    df["era_ratio"] = lg_era / df["p_era"].replace(0, float("nan"))
    df["era_ratio"] = df["era_ratio"].clip(upper=6.0)  # cap extreme outliers

    # HR suppression plus stat (inverted: lower pitcher HR = higher value)
    df["hr_plus_inv"] = max(lg_hr, _FLOOR) / df["p_hr_rate"].replace(0, float("nan"))
    df["hr_plus_inv"] = df["hr_plus_inv"].clip(upper=6.0)

    # BABIP suppression (inverted: lower pitcher BABIP = higher value)
    df["babip_pit_plus_inv"] = max(lg_bp, _FLOOR) / df["p_babip_pit"].replace(0, float("nan"))
    df["babip_pit_plus_inv"] = df["babip_pit_plus_inv"].clip(upper=3.0)

    # CMD composite: era_ratio dominates (50%), tempered by HR and BABIP suppression
    df["cmd_composite"] = (
        0.50 * df["era_ratio"].fillna(1.0)          +
        0.30 * df["hr_plus_inv"].fillna(1.0)        +
        0.20 * df["babip_pit_plus_inv"].fillna(1.0)
    )

    # STA: IP per start for starters; IP per appearance for relievers
    is_starter = df["GS"] / df["G"].clip(lower=1) >= 0.5
    df["ip_per_outing"] = df.apply(
        lambda r: r["IP"] / r["GS"] if (r["GS"] > 0 and is_starter[r.name]) else r["IP"] / r["G"],
        axis=1,
    )
    df["rel_sta"] = df["ip_per_outing"] / _AVG_IP_PER_START

    return df


def normalize_player_stats(player_df: pd.DataFrame, league_context: dict) -> pd.DataFrame:
    """
    Computes per-player rate stats and relative scores for use by build_hitter_card.
    PA = AB + BB + HBP + SF (SF filled to 0 if missing).

    POW is driven by rel_iso and rel_xbh (fixed cross-era baselines from ratings.py),
    NOT by rel_hr vs the current era average.  This prevents deadball hitters from
    receiving inflated POW because their era's league HR floor was near zero.
    rel_hr is still computed and stored in normalized_rates for audit purposes only.
    """
    # Lazy import avoids circular dependency (ratings imports ingestion in __main__)
    from ratings import HIST_AVG_HR_RATE, HIST_AVG_ISO, HIST_AVG_XBH_PCT

    df = player_df.copy()
    df["SF"] = df["SF"].fillna(0)
    df["HBP"] = df["HBP"].fillna(0)

    df["PA"]  = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    df["BIP"] = df["AB"] - df["SO"] - df["HR"] + df["SF"]   # balls in play

    safe_pa = df["PA"].replace(0, float("nan"))
    safe_ab = df["AB"].replace(0, float("nan"))
    safe_bip = df["BIP"].replace(0, float("nan"))

    # Raw player rates
    df["k_rate"]  = df["SO"] / safe_pa
    df["bb_rate"] = df["BB"] / safe_pa
    df["hr_rate"] = df["HR"] / safe_pa
    df["babip"]   = (df["H"] - df["HR"]) / safe_bip

    # Era-relative scores (used by all traits except POW)
    df["xbh_rate"] = (df["2B"] + df["3B"]) / safe_pa

    df["rel_k"]    = df["k_rate"]  / league_context["lg_avg_k_rate"]
    df["rel_bb"]   = df["bb_rate"] / league_context["lg_avg_bb_rate"]
    df["rel_hr"]   = df["hr_rate"] / league_context["lg_avg_hr_rate"]   # audit only
    df["rel_babip"]= df["babip"]   / league_context["lg_avg_babip"]
    df["rel_gap"]  = df["xbh_rate"]/ league_context["lg_avg_xbh_rate"]

    # ERA-adjusted plus stats for hybrid calibration (PRD 02 Fix v2):
    #   iso_plus  = player_ISO / league_ISO for this year
    #   xbh_plus  = player_XBH% / league_XBH% for this year  (GAP driver)
    # These replace the fixed HIST_AVG_* baselines for the primary trait drivers.
    df["iso"]      = (df["2B"] + 2*df["3B"] + 3*df["HR"]) / safe_ab
    df["xbh_pct"]  = (df["2B"] + df["3B"]  +   df["HR"]) / safe_pa
    df["xbh_rate"] = (df["2B"] + df["3B"]) / safe_pa      # non-HR XBHs for GAP

    lg_iso     = league_context.get("lg_avg_iso",     0.130)
    lg_xbh_bat = league_context.get("lg_avg_xbh_rate",0.065)

    df["iso_plus"] = df["iso"]      / max(lg_iso,      1e-6)
    df["xbh_plus"] = df["xbh_rate"] / max(lg_xbh_bat, 1e-6)

    # Legacy fixed-baseline fields kept for backward compat and audit comparison
    df["xbh_pct"]     = (df["2B"] + df["3B"]  +   df["HR"]) / safe_pa
    df["rel_hr_hist"] = df["hr_rate"]  / HIST_AVG_HR_RATE
    df["rel_iso"]     = df["iso"]      / HIST_AVG_ISO
    df["rel_xbh"]     = df["xbh_pct"] / HIST_AVG_XBH_PCT

    return df


if __name__ == "__main__":
    from ratings import build_hitter_card, build_pitcher_card, export_to_json

    TEAM   = "NYA"
    YEAR   = 1927
    OUT    = f"pilot_{YEAR}_{TEAM.lower()}.json"

    adapter = LahmanAdapter()
    people  = adapter.load_people()
    lg      = adapter.load_league_context(YEAR)

    # ── Hitter cards (min 100 PA) ──────────────────────────────────────────
    batting = adapter.load_batting(TEAM, YEAR)
    batting = batting.merge(people, on="playerID", how="left")
    batting["name"] = batting["nameFirst"] + " " + batting["nameLast"]
    normed_bat = normalize_player_stats(batting, lg)

    hitter_cards = []
    for _, row in normed_bat.iterrows():
        if row["PA"] < 100:
            continue
        card = build_hitter_card(row, row["name"], str(row["bats"]),
                                 str(row["throws"]), team_id=TEAM)
        hitter_cards.append(card)

    # ── Pitcher cards (min 30 BFP) ─────────────────────────────────────────
    pitching = adapter.load_pitching(TEAM, YEAR)
    pitching = pitching.merge(people, on="playerID", how="left")
    pitching["name"] = pitching["nameFirst"] + " " + pitching["nameLast"]
    normed_pit = normalize_pitcher_stats(pitching, lg)

    pitcher_cards = []
    for _, row in normed_pit.iterrows():
        if row["BFP"] < 30:
            continue
        card = build_pitcher_card(row, row["name"], str(row["bats"]),
                                  str(row["throws"]), team_id=TEAM)
        pitcher_cards.append(card)

    all_cards = hitter_cards + pitcher_cards

    # ── Print preview ──────────────────────────────────────────────────────
    print(f"=== {YEAR} {TEAM} — Hitters ===")
    print(f"{'Name':<22}  PA   POW  EYE   AK  CON  GAP")
    print("─" * 52)
    for c in sorted(hitter_cards, key=lambda x: -x.POW):
        row = normed_bat[normed_bat["playerID"] == c.player_id].iloc[0]
        print(f"{c.name:<22} {int(row['PA']):>4}  "
              f"{c.POW:>3}  {c.EYE:>3}  {c.AK:>3}  {c.CON:>3}  {c.GAP:>3}")

    print(f"\n=== {YEAR} {TEAM} — Pitchers ===")
    print(f"{'Name':<22} Role  BFP   STF  CTL  CMD  STA")
    print("─" * 52)
    for c in sorted(pitcher_cards, key=lambda x: -x.STF):
        row = normed_pit[normed_pit["playerID"] == c.player_id].iloc[0]
        print(f"{c.name:<22}  {c.pitcher_role:<2}  {int(row['BFP']):>4}  "
              f"{c.STF:>3}  {c.CTL:>3}  {c.CMD:>3}  {c.STA:>3}")

    print()

    # ── Spotlight: Ruth card ───────────────────────────────────────────────
    ruth = next(c for c in hitter_cards if c.player_id == "ruthba01")
    ruth.print_card()
    print()

    # ── Export ────────────────────────────────────────────────────────────
    export_to_json(all_cards, OUT)
