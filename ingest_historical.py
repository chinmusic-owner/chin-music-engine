import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Optional: if you already have database.py, you can import your client from there instead.
from supabase import create_client


DEFAULT_CONSTANTS = {
    # shrinkage strength: larger => more regression to league avg
    "shrink_pa": 200.0,   # for batter rates
    "shrink_bf": 250.0,   # for pitcher rates

    # trait mapping sensitivity (bigger => more extreme traits for same relative performance)
    "alpha_eye": 1.2,
    "alpha_con": 1.1,
    "alpha_pow": 1.3,
    "alpha_spd": 1.0,
    "alpha_stf": 1.2,
    "alpha_ctl": 1.2,
    "alpha_mov": 1.2,

    # minimum playing time for "reliable" (still ingests below this, but traits are closer to 50)
    "min_pa_for_full_weight": 450,
    "min_bf_for_full_weight": 700,
}

# ---- Helpers ----

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d and d != 0 else 0.0

def shrink_rate(player_rate: float, lg_rate: float, denom: float, shrink: float) -> float:
    # Empirical-Bayes-ish shrinkage toward league average
    w = denom / (denom + shrink) if denom > 0 else 0.0
    return w * player_rate + (1.0 - w) * lg_rate

def trait_from_relative(rel: float, alpha: float, center: float = 50.0, scale: float = 25.0) -> int:
    """
    Monotonic mapping:
      rel = 1.0 => 50
      rel > 1.0 => >50
      rel < 1.0 => <50
    Uses tanh(log(rel)) for diminishing returns.
    """
    rel = max(rel, 1e-6)
    x = math.tanh(alpha * math.log(rel))
    val = center + scale * x
    return int(round(clamp(val, 0, 100)))

def pick_lahman_dir(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if os.path.isdir("lahman_1871-2025_csv"):
        return "lahman_1871-2025_csv"
    return "."

def require_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}\n"
            f"Found columns: {list(df.columns)[:30]} ...\n"
            f"If your Lahman export uses different names, tell me and I'll adjust the adapter."
        )

# ---- Lahman load + aggregate ----

@dataclass
class LahmanTables:
    people: pd.DataFrame
    batting: pd.DataFrame
    pitching: pd.DataFrame

def load_lahman(lahman_dir: str) -> LahmanTables:
    people_path = os.path.join(lahman_dir, "People.csv")
    batting_path = os.path.join(lahman_dir, "Batting.csv")
    pitching_path = os.path.join(lahman_dir, "Pitching.csv")

    if not os.path.exists(people_path):
        raise FileNotFoundError(f"Missing {people_path}")
    if not os.path.exists(batting_path):
        raise FileNotFoundError(f"Missing {batting_path}")
    if not os.path.exists(pitching_path):
        raise FileNotFoundError(f"Missing {pitching_path}")

    people = pd.read_csv(people_path)
    batting = pd.read_csv(batting_path)
    pitching = pd.read_csv(pitching_path)

    # Validate expected Lahman columns (standard Lahman)
    require_cols(people, ["playerID", "nameFirst", "nameLast"], "People.csv")
    require_cols(batting, ["playerID", "yearID", "AB", "H", "2B", "3B", "HR", "BB", "SO"], "Batting.csv")
    require_cols(pitching, ["playerID", "yearID", "G", "GS", "H", "HR", "BB", "SO"], "Pitching.csv")

    return LahmanTables(people=people, batting=batting, pitching=pitching)

def compute_pa(bat: pd.DataFrame) -> pd.Series:
    # Lahman often has HBP, SF, SH; fall back gracefully
    ab = bat["AB"].fillna(0)
    bb = bat["BB"].fillna(0)
    hbp = bat["HBP"].fillna(0) if "HBP" in bat.columns else 0
    sf = bat["SF"].fillna(0) if "SF" in bat.columns else 0
    sh = bat["SH"].fillna(0) if "SH" in bat.columns else 0
    return ab + bb + hbp + sf + sh

def compute_bip(bat: pd.DataFrame) -> pd.Series:
    ab = bat["AB"].fillna(0)
    so = bat["SO"].fillna(0)
    hr = bat["HR"].fillna(0)
    sf = bat["SF"].fillna(0) if "SF" in bat.columns else 0
    # Approximation: AB-SO-HR plus SF as a ball in play event
    return (ab - so - hr) + sf

def agg_batting_player_season(batting: pd.DataFrame) -> pd.DataFrame:
    # aggregate across stints/teams
    sum_cols = [c for c in ["AB","H","2B","3B","HR","BB","SO","HBP","SF","SH","SB","CS"] if c in batting.columns]
    grp = batting.groupby(["playerID","yearID"], as_index=False)[sum_cols].sum()
    grp["PA"] = compute_pa(grp)
    grp["BIP"] = compute_bip(grp)
    grp["1B"] = grp["H"] - grp.get("2B", 0) - grp.get("3B", 0) - grp.get("HR", 0)
    grp["XBH"] = grp.get("2B", 0) + grp.get("3B", 0)
    return grp

def estimate_bf(pit: pd.DataFrame) -> pd.Series:
    # Prefer BFP if present
    if "BFP" in pit.columns:
        return pit["BFP"].fillna(0)
    # Else estimate from outs + events (rough but usable)
    outs = pit["IPouts"].fillna(0) if "IPouts" in pit.columns else 0
    h = pit["H"].fillna(0)
    bb = pit["BB"].fillna(0)
    hbp = pit["HBP"].fillna(0) if "HBP" in pit.columns else 0
    sf = pit["SF"].fillna(0) if "SF" in pit.columns else 0
    sh = pit["SH"].fillna(0) if "SH" in pit.columns else 0
    return outs + h + bb + hbp + sf + sh

def agg_pitching_player_season(pitching: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [c for c in ["G","GS","H","HR","BB","SO","HBP","IPouts","BFP","SF","SH"] if c in pitching.columns]
    grp = pitching.groupby(["playerID","yearID"], as_index=False)[sum_cols].sum()
    grp["BF"] = estimate_bf(grp)
    return grp

def build_people_lookup(people: pd.DataFrame) -> pd.DataFrame:
    ppl = people.copy()
    ppl["player_name"] = (ppl["nameFirst"].fillna("").astype(str).str.strip() + " " +
                          ppl["nameLast"].fillna("").astype(str).str.strip()).str.strip()
    # optional handedness if you want it later
    if "bats" not in ppl.columns:
        ppl["bats"] = None
    if "throws" not in ppl.columns:
        ppl["throws"] = None
    return ppl[["playerID", "player_name", "bats", "throws"]]

# ---- League context + traits ----

def league_context_from_batting(bat_ps: pd.DataFrame) -> pd.DataFrame:
    # League context by season only (v1). If you want lgID splits later, we can add Teams.csv join.
    ctx = bat_ps.groupby(["yearID"], as_index=False).agg({
        "PA": "sum",
        "SO": "sum",
        "BB": "sum",
        "HR": "sum",
        "H": "sum",
        "1B": "sum",
        "XBH": "sum",
        "BIP": "sum",
        "HBP": "sum" if "HBP" in bat_ps.columns else "sum",
        "SB": "sum" if "SB" in bat_ps.columns else "sum",
        "CS": "sum" if "CS" in bat_ps.columns else "sum",
    })

    ctx["k_rate"] = ctx["SO"] / ctx["PA"]
    ctx["bb_rate"] = ctx["BB"] / ctx["PA"]
    ctx["hr_rate"] = ctx["HR"] / ctx["PA"]
    # BABIP proxy = (H - HR) / BIP
    ctx["babip"] = (ctx["H"] - ctx["HR"]) / ctx["BIP"].replace(0, pd.NA)
    ctx["babip"] = ctx["babip"].fillna(0.0)

    # Speed proxy: SB per PA adjusted by success
    sb = ctx["SB"] if "SB" in ctx.columns else 0
    cs = ctx["CS"] if "CS" in ctx.columns else 0
    attempts = (sb + cs).replace(0, pd.NA)
    success = (sb / attempts).fillna(0.7)  # default-ish if no attempts
    ctx["sb_index"] = (sb / ctx["PA"]) * (0.5 + 0.5 * success)

    return ctx[["yearID","k_rate","bb_rate","hr_rate","babip","sb_index"]]

def global_baseline(ctx_by_year: pd.DataFrame) -> Dict[str, float]:
    # simple baseline = mean across years (could be weighted by PA later)
    return {
        "k_rate": float(ctx_by_year["k_rate"].mean()),
        "bb_rate": float(ctx_by_year["bb_rate"].mean()),
        "hr_rate": float(ctx_by_year["hr_rate"].mean()),
        "babip": float(ctx_by_year["babip"].mean()),
        "sb_index": float(ctx_by_year["sb_index"].mean()),
    }

def build_hitter_traits(row, lg_row, base, C) -> Dict[str, int]:
    pa = float(row["PA"])
    # raw rates
    k = safe_div(row["SO"], pa)
    bb = safe_div(row["BB"], pa)
    hr = safe_div(row["HR"], pa)
    babip = safe_div((row["H"] - row["HR"]), max(row["BIP"], 0.0))

    # speed index
    sb = float(row["SB"]) if "SB" in row and not pd.isna(row["SB"]) else 0.0
    cs = float(row["CS"]) if "CS" in row and not pd.isna(row["CS"]) else 0.0
    att = sb + cs
    success = safe_div(sb, att) if att > 0 else 0.0
    sb_index = safe_div(sb, pa) * (0.5 + 0.5 * (success if att > 0 else 0.7))

    # shrink toward league-year
    k_adj = shrink_rate(k, lg_row["k_rate"], pa, C["shrink_pa"])
    bb_adj = shrink_rate(bb, lg_row["bb_rate"], pa, C["shrink_pa"])
    hr_adj = shrink_rate(hr, lg_row["hr_rate"], pa, C["shrink_pa"])
    babip_adj = shrink_rate(babip, lg_row["babip"], max(row["BIP"], 0.0), C["shrink_pa"])
    spd_adj = shrink_rate(sb_index, lg_row["sb_index"], pa, C["shrink_pa"])

    # normalize to global baseline (relative performance)
    rel_bb = safe_div(bb_adj, lg_row["bb_rate"]) if lg_row["bb_rate"] > 0 else 1.0
    rel_hr = safe_div(hr_adj, lg_row["hr_rate"]) if lg_row["hr_rate"] > 0 else 1.0
    rel_babip = safe_div(babip_adj, lg_row["babip"]) if lg_row["babip"] > 0 else 1.0
    rel_spd = safe_div(spd_adj, lg_row["sb_index"]) if lg_row["sb_index"] > 0 else 1.0

    return {
        "contact": trait_from_relative(rel_babip, C["alpha_con"]),
        "power": trait_from_relative(rel_hr, C["alpha_pow"]),
        "eye": trait_from_relative(rel_bb, C["alpha_eye"]),
        "speed": trait_from_relative(rel_spd, C["alpha_spd"]),
    }

def build_pitcher_traits(row, pit_lg, C) -> Dict[str, int]:
    bf = float(row["BF"])
    k = safe_div(row["SO"], bf)
    bb = safe_div(row["BB"], bf)
    hr = safe_div(row["HR"], bf)

    # shrink to league-year
    k_adj = shrink_rate(k, pit_lg["k_rate"], bf, C["shrink_bf"])
    bb_adj = shrink_rate(bb, pit_lg["bb_rate"], bf, C["shrink_bf"])
    hr_adj = shrink_rate(hr, pit_lg["hr_rate"], bf, C["shrink_bf"])

    rel_k = safe_div(k_adj, pit_lg["k_rate"]) if pit_lg["k_rate"] > 0 else 1.0            # higher better
    rel_bb_supp = safe_div(pit_lg["bb_rate"], bb_adj) if bb_adj > 0 else 1.0              # lower better
    rel_hr_supp = safe_div(pit_lg["hr_rate"], hr_adj) if hr_adj > 0 else 1.0              # lower better

    return {
        "stuff": trait_from_relative(rel_k, C["alpha_stf"]),
        "control": trait_from_relative(rel_bb_supp, C["alpha_ctl"]),
        "movement": trait_from_relative(rel_hr_supp, C["alpha_mov"]),
    }

# ---- Supabase push ----

def get_supabase_client():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) in .env.\n"
            "For ingestion writes, service role is recommended (kept local, never committed)."
        )
    return create_client(url, key)

def upsert_players(sb, rows):
    # Requires unique index on (player_id, season_year)
    resp = sb.table("players").upsert(rows, on_conflict="player_id,season_year").execute()
    return resp

# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lahman_dir", default=None)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--min_pa", type=int, default=1)
    parser.add_argument("--min_bf", type=int, default=1)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    C = DEFAULT_CONSTANTS
    lahman_dir = pick_lahman_dir(args.lahman_dir)
    tables = load_lahman(lahman_dir)

    ppl = build_people_lookup(tables.people)
    bat_ps = agg_batting_player_season(tables.batting)
    pit_ps = agg_pitching_player_season(tables.pitching)

    # League context from batting; v1 uses same ctx for pitching (good enough to start)
    ctx = league_context_from_batting(bat_ps)
    base = global_baseline(ctx)

    # Filter season
    bat_s = bat_ps[bat_ps["yearID"] == args.season].copy()
    pit_s = pit_ps[pit_ps["yearID"] == args.season].copy()
    ctx_s = ctx[ctx["yearID"] == args.season]
    if ctx_s.empty:
        raise ValueError(f"No league context computed for season {args.season}. Is Batting.csv populated?")
    lg_row = ctx_s.iloc[0].to_dict()

    # Join names
    bat_s = bat_s.merge(ppl, on="playerID", how="left")
    pit_s = pit_s.merge(ppl, on="playerID", how="left")

    # Build hitter cards
    hitter_rows = []
    for _, r in bat_s.iterrows():
        if r["PA"] < args.min_pa:
            continue
        traits = build_hitter_traits(r, lg_row, base, C)
        hitter_rows.append({
            "player_id": r["playerID"],
            "player_name": r["player_name"],
            "season_year": int(r["yearID"]),
            "team": None,
            "position": None,
            **traits,
            # pitcher traits left null
            "stuff": None,
            "control": None,
            "movement": None,
        })

    # Build pitcher cards
    pitcher_rows = []
    for _, r in pit_s.iterrows():
        if r["BF"] < args.min_bf:
            continue
        traits = build_pitcher_traits(r, lg_row, C)
        pitcher_rows.append({
            "player_id": r["playerID"],
            "player_name": r["player_name"],
            "season_year": int(r["yearID"]),
            "team": None,
            "position": None,
            # hitter traits left null
            "contact": None,
            "power": None,
            "eye": None,
            "speed": None,
            **traits,
        })

    # Merge hitter+pitcher rows by (player_id, season_year) so two-way ends up in one row
    merged: Dict[Tuple[str,int], Dict] = {}
    for row in hitter_rows + pitcher_rows:
        key = (row["player_id"], row["season_year"])
        if key not in merged:
            merged[key] = row
        else:
            merged[key].update({k: v for k, v in row.items() if v is not None})

    out_rows = list(merged.values())
    print(f"Season {args.season}: built {len(out_rows)} player-season rows")

    if not args.push:
        print("Dry run only (no push). Add --push to write to Supabase.")
        print("Sample row:", out_rows[0] if out_rows else None)
        return

    sb = get_supabase_client()
    resp = upsert_players(sb, out_rows)
    print("Upsert complete.")
    # supabase-py returns object-ish; print minimal
    try:
        print(f"Inserted/updated rows: {len(resp.data) if resp.data else 0}")
    except Exception:
        print("Done.")

if __name__ == "__main__":
    main()
