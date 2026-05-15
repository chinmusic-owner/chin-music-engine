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
        """
        path = os.path.join(LAHMAN_DIR, "Teams.csv")
        df = pd.read_csv(path)
        df = df[df["yearID"] == year].copy()
        df["SF"] = df["SF"].fillna(0)
        df["HBP"] = df["HBP"].fillna(0)

        t = df[["AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SF"]].sum()

        lg_pa    = t["AB"] + t["BB"] + t["HBP"] + t["SF"]
        lg_bip   = t["AB"] - t["SO"] - t["HR"] + t["SF"]

        return {
            "year":             year,
            "lg_pa":            lg_pa,
            "lg_avg_k_rate":    t["SO"]  / lg_pa,
            "lg_avg_bb_rate":   t["BB"]  / lg_pa,
            "lg_avg_hr_rate":   t["HR"]  / lg_pa,
            "lg_avg_babip":     (t["H"] - t["HR"]) / lg_bip,
            "lg_avg_xbh_rate":  (t["2B"] + t["3B"]) / lg_pa,
        }


def normalize_pitcher_stats(pitcher_df: pd.DataFrame, league_context: dict) -> pd.DataFrame:
    """
    Computes per-pitcher rate stats and relative scores vs the league batting baseline.
    Since pitchers and batters face the same opponent pool, the batting-side league
    averages serve as the correct denominator for pitcher rate comparisons.

    STA uses IP/start relative to a fixed era baseline (_AVG_IP_PER_START in ratings.py).
    """
    from ratings import _AVG_IP_PER_START

    df = pitcher_df.copy()
    df["BFP"]  = df["BFP"].fillna(0)
    df["HBP"]  = df["HBP"].fillna(0)
    df["IP"]   = df["IPouts"] / 3.0

    safe_bfp = df["BFP"].replace(0, float("nan"))

    df["p_k_rate"]  = df["SO"] / safe_bfp
    df["p_bb_rate"] = df["BB"] / safe_bfp
    df["p_hr_rate"] = df["HR"] / safe_bfp

    # Higher pitcher K rate vs league avg = better STF
    df["rel_p_k"]  = df["p_k_rate"]  / league_context["lg_avg_k_rate"]
    # Higher pitcher BB rate vs league avg = worse CTL (inverted in build_pitcher_card)
    df["rel_p_bb"] = df["p_bb_rate"] / league_context["lg_avg_bb_rate"]
    # Higher pitcher HR rate vs league avg = worse CMD (inverted in build_pitcher_card)
    df["rel_p_hr"] = df["p_hr_rate"] / league_context["lg_avg_hr_rate"]

    # STA: IP per start for starters; IP per appearance for relievers
    safe_gs = df["GS"].replace(0, float("nan"))
    safe_g  = df["G"].replace(0, float("nan"))
    is_starter = df["GS"] / df["G"].clip(lower=1) >= 0.5

    df["ip_per_outing"] = df.apply(
        lambda r: r["IP"] / r["GS"] if (r["GS"] > 0 and is_starter[r.name]) else r["IP"] / r["G"],
        axis=1,
    )
    df["rel_sta"] = df["ip_per_outing"] / _AVG_IP_PER_START

    return df


def normalize_player_stats(player_df: pd.DataFrame, league_context: dict) -> pd.DataFrame:
    """
    Computes per-player rate stats and divides each by the league average
    to produce a Relative Score (1.0 = league average, 2.0 = twice average, etc.).
    PA = AB + BB + HBP + SF (SF filled to 0 if missing).
    """
    df = player_df.copy()
    df["SF"] = df["SF"].fillna(0)
    df["HBP"] = df["HBP"].fillna(0)

    df["PA"] = df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    df["BIP"] = df["AB"] - df["SO"] - df["HR"] + df["SF"]   # balls in play

    # Raw player rates (guard against divide-by-zero for zero-PA rows)
    df["k_rate"]  = df["SO"]  / df["PA"].replace(0, float("nan"))
    df["bb_rate"] = df["BB"]  / df["PA"].replace(0, float("nan"))
    df["hr_rate"] = df["HR"]  / df["PA"].replace(0, float("nan"))
    df["babip"]   = (df["H"] - df["HR"]) / df["BIP"].replace(0, float("nan"))

    # Relative scores vs league average (1.0 = league avg)
    df["xbh_rate"] = (df["2B"] + df["3B"]) / df["PA"].replace(0, float("nan"))

    df["rel_k"]    = df["k_rate"]   / league_context["lg_avg_k_rate"]
    df["rel_bb"]   = df["bb_rate"]  / league_context["lg_avg_bb_rate"]
    df["rel_hr"]   = df["hr_rate"]  / league_context["lg_avg_hr_rate"]
    df["rel_babip"]= df["babip"]    / league_context["lg_avg_babip"]
    df["rel_gap"]  = df["xbh_rate"] / league_context["lg_avg_xbh_rate"]

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
