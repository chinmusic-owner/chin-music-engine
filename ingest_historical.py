import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Optional: if you already have database.py, you can import your client from there instead.
from supabase import create_client

from global_calibration import get_global_stats, z_score_trait, sigmoid_trait

# Load hybrid global calibration once at module level.
_GLOBAL_STATS = get_global_stats()
_GH = _GLOBAL_STATS["hitter"]   # plus stat distributions for hitters
_GP = _GLOBAL_STATS["pitcher"]  # plus stat distributions for pitchers

# ---- Helpers ----

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d and d != 0 else 0.0

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
    fielding: pd.DataFrame

def load_lahman(lahman_dir: str) -> LahmanTables:
    people_path   = os.path.join(lahman_dir, "People.csv")
    batting_path  = os.path.join(lahman_dir, "Batting.csv")
    pitching_path = os.path.join(lahman_dir, "Pitching.csv")
    fielding_path = os.path.join(lahman_dir, "Fielding.csv")

    for path in [people_path, batting_path, pitching_path, fielding_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")

    people   = pd.read_csv(people_path)
    batting  = pd.read_csv(batting_path)
    pitching = pd.read_csv(pitching_path)
    fielding = pd.read_csv(fielding_path)

    # Validate expected Lahman columns (standard Lahman)
    require_cols(people,   ["playerID", "nameFirst", "nameLast"], "People.csv")
    require_cols(batting,  ["playerID", "yearID", "AB", "H", "2B", "3B", "HR", "BB", "SO"], "Batting.csv")
    require_cols(pitching, ["playerID", "yearID", "G", "GS", "H", "HR", "BB", "SO"], "Pitching.csv")
    require_cols(fielding, ["playerID", "yearID", "POS", "G"], "Fielding.csv")

    return LahmanTables(people=people, batting=batting, pitching=pitching, fielding=fielding)


# ---- Position / role derivation ----

# Lahman raw position → Chin Music scarcity group.
# Lahman uses 'OF' for all outfield (not split LF/CF/RF); LF/CF/RF included
# as a forward-compat guard for any Lahman versions that do split them.
_LAHMAN_TO_CM_POS: dict[str, str] = {
    "C":  "C",
    "1B": "1B",
    "2B": "IF",
    "3B": "IF",
    "SS": "IF",
    "OF": "OF",
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "DH": "UTIL",
}
# Positions that are events, not defensive slots — excluded from primary_position logic.
_NON_DEFENSIVE_POS = {"P", "PH", "PR"}


def compute_pitcher_roles(pitching: pd.DataFrame) -> dict[tuple, str]:
    """
    Returns {(playerID, yearID): 'SP' | 'RP'} for every pitcher-season.

    Logic: aggregate G and GS across stints, then SP if GS/G >= 0.5.
    A pitcher who never started (GS=0) is always RP.
    """
    sum_cols = [c for c in ["G", "GS"] if c in pitching.columns]
    agg = pitching.groupby(["playerID", "yearID"], as_index=False)[sum_cols].sum()
    agg["GS"] = agg["GS"].fillna(0)

    result: dict[tuple, str] = {}
    for _, row in agg.iterrows():
        g  = float(row["G"])
        gs = float(row["GS"])
        role = "SP" if (g > 0 and gs / g >= 0.5) else "RP"
        result[(row["playerID"], int(row["yearID"]))] = role
    return result


def compute_primary_positions(fielding: pd.DataFrame) -> dict[tuple, str]:
    """
    Returns {(playerID, yearID): CM_position_group} for every hitter-season.

    Logic:
      1. Exclude non-defensive appearances (P, PH, PR).
      2. Aggregate games by (playerID, yearID, POS) across stints.
      3. Pick the position with the most games for each player-season.
      4. Map the winning Lahman POS to a Chin Music scarcity group.
      5. Anything not in the map → 'UTIL'.
    """
    df = fielding[~fielding["POS"].isin(_NON_DEFENSIVE_POS)].copy()
    if df.empty:
        return {}

    agg = df.groupby(["playerID", "yearID", "POS"], as_index=False)["G"].sum()
    # argmax of G per player-season → dominant position
    idx  = agg.groupby(["playerID", "yearID"])["G"].idxmax()
    best = agg.loc[idx]

    result: dict[tuple, str] = {}
    for _, row in best.iterrows():
        cm_pos = _LAHMAN_TO_CM_POS.get(row["POS"], "UTIL")
        result[(row["playerID"], int(row["yearID"]))] = cm_pos
    return result

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
    """
    Builds hitter traits using the hybrid era-adjusted + global z-score system.

    Drivers (era-adjusted plus stats, globally z-scored):
      contact ← BABIP_plus  = player_BABIP / league_BABIP
      power   ← ISO_plus    = player_ISO   / league_ISO
      eye     ← BB_plus     = player_BB%   / league_BB%
      speed   ← SB index (era-relative, contextual)
    """
    _FLOOR = 1e-6
    pa  = float(row["PA"])
    bip = float(max(row["BIP"], 0.0))

    bb_rate = safe_div(row["BB"], pa)
    babip   = safe_div(row["H"] - row["HR"], bip) if bip > 0 else 0.0
    ab = float(row.get("AB", pa - row.get("BB", 0)))
    iso = safe_div(
        float(row.get("2B", 0)) + 2.0 * float(row.get("3B", 0)) + 3.0 * float(row.get("HR", 0)),
        ab,
    )

    # Era-adjust via league context
    lg_bb   = max(float(lg_row.get("bb_rate", 0.085)),  _FLOOR)
    lg_babip= max(float(lg_row.get("babip",   0.300)),  _FLOOR)
    lg_iso  = max(float(lg_row.get("lg_avg_iso", 0.130)), _FLOOR)

    bb_plus   = bb_rate  / lg_bb
    babip_plus= babip    / lg_babip
    iso_plus  = iso      / lg_iso

    # Speed: SB-based index, era-relative (contextual)
    sb = float(row["SB"]) if "SB" in row and not pd.isna(row["SB"]) else 0.0
    cs = float(row["CS"]) if "CS" in row and not pd.isna(row["CS"]) else 0.0
    att = sb + cs
    success  = safe_div(sb, att) if att > 0 else 0.0
    sb_index = safe_div(sb, pa) * (0.5 + 0.5 * (success if att > 0 else 0.7))
    spd_adj  = (sb_index / lg_row["sb_index"]) if lg_row.get("sb_index", 0) > 0 else 1.0
    spd_trait = int(round(clamp(50.0 + 25.0 * math.tanh(math.log(max(spd_adj, 1e-6))), 20, 99)))

    return {
        "contact": z_score_trait(babip_plus, _GH["babip_plus"]["mean"], _GH["babip_plus"]["std"]),
        "power":   z_score_trait(iso_plus,   _GH["iso_plus"]["mean"],   _GH["iso_plus"]["std"]),
        "eye":     z_score_trait(bb_plus,    _GH["bb_plus"]["mean"],    _GH["bb_plus"]["std"]),
        "speed":   spd_trait,
    }


def build_pitcher_traits(row, pit_lg, C) -> Dict[str, int]:
    """
    Builds pitcher traits using the hybrid era-adjusted + global z-score system.

    Drivers (era-adjusted plus stats, globally z-scored):
      stuff    ← K_plus       = pitcher_K%  / league_K%
      control  ← BB_plus_inv  = league_BB%  / pitcher_BB%  (inverted)
      movement ← CMD composite = 0.50*era_ratio + 0.30*hr_plus_inv + 0.20*babip_plus_inv
    """
    _FLOOR = 1e-6
    bf = float(row["BF"])

    k_rate  = safe_div(row["SO"], bf)
    bb_rate = safe_div(row["BB"], bf)
    hr_rate = safe_div(row["HR"], bf)

    # Era-adjust (pit_lg comes from league_context_from_batting which has k_rate/bb_rate)
    lg_k  = max(float(pit_lg.get("k_rate",  0.14)),  _FLOOR)
    lg_bb = max(float(pit_lg.get("bb_rate", 0.083)), _FLOOR)
    lg_hr = max(float(pit_lg.get("hr_rate", 0.019)), _FLOOR)
    lg_era  = max(float(pit_lg.get("lg_era",     4.00)), _FLOOR)
    lg_babip= max(float(pit_lg.get("babip",      0.300)), _FLOOR)

    k_plus     = k_rate  / lg_k
    bb_plus_inv= lg_bb   / max(bb_rate, _FLOOR)

    # Pitcher ERA (need ER and IPouts)
    outs = float(row.get("IPouts", 0)) if "IPouts" in row else 0.0
    er   = float(row.get("ER",     0)) if "ER"    in row else 0.0
    pit_era = (er * 27.0 / outs) if outs > 0 else lg_era
    era_ratio = lg_era / max(pit_era, _FLOOR)
    era_ratio = min(era_ratio, 6.0)  # cap extreme outliers

    hr_plus_inv  = lg_hr   / max(hr_rate, _FLOOR)
    hr_plus_inv  = min(hr_plus_inv, 6.0)

    # Pitcher BABIP: (H-HR) / (BF-SO-BB-HBP-HR)
    hbp = float(row.get("HBP", 0)) if "HBP" in row else 0.0
    bip_den = bf - float(row["SO"]) - float(row["BB"]) - hbp - float(row["HR"])
    if bip_den > 0:
        pit_babip = (float(row["H"]) - float(row["HR"])) / bip_den
        babip_inv = min(lg_babip / max(pit_babip, _FLOOR), 3.0)
    else:
        babip_inv = 1.0

    cmd_composite = 0.50 * era_ratio + 0.30 * hr_plus_inv + 0.20 * babip_inv

    return {
        "stuff":    sigmoid_trait(k_plus,        _GP["k_plus"]["mean"],        _GP["k_plus"]["std"]),
        "control":  sigmoid_trait(bb_plus_inv,   _GP["bb_plus_inv"]["mean"],   _GP["bb_plus_inv"]["std"]),
        "movement": sigmoid_trait(cmd_composite, _GP["cmd_composite"]["mean"], _GP["cmd_composite"]["std"]),
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

    ppl    = build_people_lookup(tables.people)
    bat_ps = agg_batting_player_season(tables.batting)
    pit_ps = agg_pitching_player_season(tables.pitching)

    # Build position / role lookup tables (full-history, season-keyed)
    pitcher_role_lookup    = compute_pitcher_roles(tables.pitching)
    primary_position_lookup = compute_primary_positions(tables.fielding)

    # League context still used for speed trait (SB era-relative) but no longer drives
    # the four primary traits — those now use global z-scores from _GLOBAL_STATS.
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

    print(f"  [calibration] Using global z-score baselines "
          f"(n_h={_GLOBAL_STATS['n_hitter_seasons']:,} / "
          f"n_p={_GLOBAL_STATS['n_pitcher_seasons']:,} qualifying seasons)")

    # Build hitter cards
    hitter_rows = []
    for _, r in bat_s.iterrows():
        if r["PA"] < args.min_pa:
            continue
        # C and base still passed for speed/legacy; primary traits use global z-scores
        traits  = build_hitter_traits(r, lg_row, base, C)
        key     = (r["playerID"], int(r["yearID"]))
        hitter_rows.append({
            "player_id":        r["playerID"],
            "player_name":      r["player_name"],
            "season_year":      int(r["yearID"]),
            "team":             None,
            "position":         None,
            **traits,
            # pitcher traits left null
            "stuff":            None,
            "control":          None,
            "movement":         None,
            # new fields
            "pitcher_role":     None,
            "primary_position": primary_position_lookup.get(key),
        })

    # Build pitcher cards
    pitcher_rows = []
    for _, r in pit_s.iterrows():
        if r["BF"] < args.min_bf:
            continue
        traits  = build_pitcher_traits(r, lg_row, C)
        key     = (r["playerID"], int(r["yearID"]))
        pitcher_rows.append({
            "player_id":        r["playerID"],
            "player_name":      r["player_name"],
            "season_year":      int(r["yearID"]),
            "team":             None,
            "position":         None,
            # hitter traits left null
            "contact":          None,
            "power":            None,
            "eye":              None,
            "speed":            None,
            **traits,
            # new fields
            "pitcher_role":     pitcher_role_lookup.get(key),
            "primary_position": None,
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

    n_with_role = sum(1 for r in out_rows if r.get("pitcher_role"))
    n_with_pos  = sum(1 for r in out_rows if r.get("primary_position"))
    print(f"Season {args.season}: built {len(out_rows)} player-season rows "
          f"({n_with_role} with pitcher_role, {n_with_pos} with primary_position)")

    if not args.push:
        print("Dry run only (no push). Add --push to write to Supabase.")
        sample = out_rows[0] if out_rows else None
        if sample:
            print(f"  Sample hitter: pitcher_role={sample.get('pitcher_role')!r}  "
                  f"primary_position={sample.get('primary_position')!r}")
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
