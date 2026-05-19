"""
ingest_fielding_traits.py — Lahman Fielding → RNG / HND / ARM on Supabase `players`.

Uses position- and season-specific z-scores on proxy rates, innings-based shrinkage
toward 50, then innings-weighted collapse across positions.

Does not modify pa_engine constants or batting/pitching traits. Upsert merges full
rows fetched from Supabase so other columns are preserved.

Schema: run migrations/add_fielding_traits.sql once.

Usage:
  python ingest_fielding_traits.py --migrate              # Postgres URI in .env (SUPABASE_DB_URL or DATABASE_URL, …)
  python ingest_fielding_traits.py --migrate --push         # migrate then upsert fielding traits
  python ingest_fielding_traits.py --lahman_dir lahman_1871-2025_csv [--dry_run]
  python ingest_fielding_traits.py --lahman_dir lahman_1871-2025_csv --push
  python ingest_fielding_traits.py --push --seasons 1927,2000

  The Postgres URI is not the same as SUPABASE_SERVICE_ROLE_KEY — use Dashboard → Database → URI.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# ── Config (tuning lives here only — not in pa_engine) ─────────────────────
MIN_INNINGS_LOW_RELIABILITY = 50   # reporting threshold
SHRINKK_INN_DENOM = 300.0          # w = innings / (innings + SHRINKK_INN_DENOM)
TRAIT_SLOPE = 10.0                 # trait = 50 + z * TRAIT_SLOPE (before shrink)
TRAIT_LO, TRAIT_HI = 20, 99
UPSERT_CHUNK = 200

# Non-defensive or non-rate positions — exclude from fielding trait pipeline
_SKIP_POS = {"DH", "PH", "PR"}


def pick_lahman_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.isdir("lahman_1871-2025_csv"):
        return "lahman_1871-2025_csv"
    return "."


def get_supabase_client():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY (or SERVICE_ROLE) in .env.")
    return create_client(url, key)


def resolve_postgres_url() -> str | None:
    """
    Postgres URI for DDL (not the Supabase JWT).
    Checks env after load_dotenv; does not override vars already set in the process environment.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)
    for key in ("SUPABASE_DB_URL", "DATABASE_URL", "POSTGRES_URL", "PGURL"):
        raw = os.getenv(key)
        if raw and str(raw).strip():
            return str(raw).strip().strip('"').strip("'")
    return None


def apply_fielding_migration_db() -> None:
    """Run migrations/add_fielding_traits.sql via direct Postgres (pooler/direct URI)."""
    db_url = resolve_postgres_url()
    if not db_url:
        raise RuntimeError(
            "No Postgres URI found. Set one of these in your environment or .env file:\n"
            "  SUPABASE_DB_URL   (recommended)\n"
            "  DATABASE_URL\n"
            "  POSTGRES_URL\n"
            "  PGURL\n\n"
            "Get the URI from Supabase Dashboard → Project Settings → Database → Connection string → URI.\n"
            "Prefer the session pooler (IPv4) if direct db.* host is IPv6-only:\n"
            "  postgresql://postgres.[ref]:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres"
        )
    try:
        import psycopg2
    except ImportError:
        raise RuntimeError("pip install psycopg2-binary to run --migrate")

    mig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations", "add_fielding_traits.sql")
    with open(mig_path, encoding="utf-8") as f:
        sql_text = f.read()

    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql_text)
    finally:
        conn.close()
    print("  Migration applied: rng, hnd, arm columns on players (IF NOT EXISTS).")


def fielding_columns_ok(path: str) -> list[str]:
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def load_aggregate_fielding(lahman_dir: str) -> pd.DataFrame:
    path = os.path.join(lahman_dir, "Fielding.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    need = ["playerID", "yearID", "POS", "G"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Fielding.csv missing required column {c!r}")
    for c in ("PO", "A", "E", "InnOuts", "GS"):
        if c not in df.columns:
            df[c] = 0
    for c in ("PO", "A", "E", "InnOuts", "G", "GS"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    agg = (
        df.groupby(["playerID", "yearID", "POS"], as_index=False)
        .agg(PO=("PO", "sum"), A=("A", "sum"), E=("E", "sum"), InnOuts=("InnOuts", "sum"), G=("G", "sum"))
    )
    return agg


def row_innings(r: pd.Series) -> float:
    """InnOuts/3 when present; else fall back to G (games) as innings proxy."""
    inn = float(r["InnOuts"]) / 3.0 if r["InnOuts"] > 0 else float(r["G"])
    return max(inn, 0.0)


def build_position_level_frame(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season-POS with rates, eligibility flags."""
    rows = []
    for _, r in agg.iterrows():
        pos = str(r["POS"]).strip()
        if pos in _SKIP_POS:
            continue
        po, a, e = float(r["PO"]), float(r["A"]), float(r["E"])
        chances = po + a + e
        inn = row_innings(r)
        if chances <= 0 or inn <= 0:
            continue
        range_rate = chances / inn
        error_rate = e / chances
        arm_rate = a / inn
        low_rel = inn < MIN_INNINGS_LOW_RELIABILITY
        rows.append(
            {
                "playerID": r["playerID"],
                "yearID": int(r["yearID"]),
                "POS": pos,
                "innings": inn,
                "chances": chances,
                "range_rate": range_rate,
                "error_rate": error_rate,
                "arm_rate": arm_rate,
                "low_reliability": low_rel,
            }
        )
    return pd.DataFrame(rows)


def add_league_zscores(f: pd.DataFrame) -> pd.DataFrame:
    """Per (yearID, POS): z-score range_rate, error_rate, arm_rate."""
    f = f.copy()

    def _z(grp: pd.DataFrame, col: str) -> pd.Series:
        mu = grp[col].mean()
        sd = grp[col].std(ddof=0)
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1e-6
        return (grp[col] - mu) / sd

    z_parts = []
    for (_, _), grp in f.groupby(["yearID", "POS"]):
        g2 = grp.copy()
        g2["z_range"] = _z(g2, "range_rate")
        g2["z_err"] = _z(g2, "error_rate")
        g2["z_hands"] = -g2["z_err"]
        g2["z_arm"] = _z(g2, "arm_rate")
        z_parts.append(g2)
    out = pd.concat(z_parts, ignore_index=True)
    return out


def z_to_trait(z: float) -> float:
    return float(np.clip(50.0 + z * TRAIT_SLOPE, TRAIT_LO, TRAIT_HI))


def apply_shrinkage(trait: float, innings: float) -> float:
    w = innings / (innings + SHRINKK_INN_DENOM)
    return w * trait + (1.0 - w) * 50.0


def collapse_player_season(f: pd.DataFrame) -> pd.DataFrame:
    """Innings-weighted RNG, HND, ARM per (playerID, yearID)."""
    f = f.copy()
    f["RNG_raw"] = f["z_range"].apply(z_to_trait)
    f["HND_raw"] = f["z_hands"].apply(z_to_trait)
    f["ARM_raw"] = f["z_arm"].apply(z_to_trait)
    f["RNG_s"] = f.apply(lambda r: apply_shrinkage(r["RNG_raw"], r["innings"]), axis=1)
    f["HND_s"] = f.apply(lambda r: apply_shrinkage(r["HND_raw"], r["innings"]), axis=1)
    f["ARM_s"] = f.apply(lambda r: apply_shrinkage(r["ARM_raw"], r["innings"]), axis=1)

    rows: list[dict[str, Any]] = []
    for (pid, yr), grp in f.groupby(["playerID", "yearID"]):
        w = grp["innings"].to_numpy(dtype=float)
        ws = float(w.sum())
        if ws <= 0:
            continue
        rng = int(round(np.average(grp["RNG_s"], weights=w)))
        hnd = int(round(np.average(grp["HND_s"], weights=w)))
        arm = int(round(np.average(grp["ARM_s"], weights=w)))
        rng = int(np.clip(rng, TRAIT_LO, TRAIT_HI))
        hnd = int(np.clip(hnd, TRAIT_LO, TRAIT_HI))
        arm = int(np.clip(arm, TRAIT_LO, TRAIT_HI))
        low_rel = bool((grp["innings"] < MIN_INNINGS_LOW_RELIABILITY).any()) or ws < MIN_INNINGS_LOW_RELIABILITY
        rows.append(
            {
                "playerID": pid,
                "yearID": int(yr),
                "rng": rng,
                "hnd": hnd,
                "arm": arm,
                "innings": ws,
                "low_rel": low_rel,
            }
        )
    return pd.DataFrame(rows)


def fetch_supabase_seasons(sb) -> list[int]:
    seasons: set[int] = set()
    page, size = 0, 1000
    while True:
        resp = sb.table("players").select("season_year").range(page * size, (page + 1) * size - 1).execute()
        batch = resp.data or []
        for r in batch:
            if r.get("season_year") is not None:
                seasons.add(int(r["season_year"]))
        if len(batch) < size:
            break
        page += 1
    return sorted(seasons)


def check_defense_columns(sb) -> bool:
    try:
        sb.table("players").select("rng,hnd,arm").limit(1).execute()
        return True
    except Exception as exc:
        if "PGRST204" in str(exc) or "column" in str(exc).lower():
            return False
        raise


def fetch_players_defense_populated(sb, seasons: list[int] | None) -> list[dict]:
    """Rows with non-null rng (post-ingest validation)."""
    rows: list[dict] = []
    page, size = 0, 1000
    while True:
        q = (
            sb.table("players")
            .select("player_id,player_name,season_year,primary_position,rng,hnd,arm")
            .not_.is_("player_id", "null")
            .not_.is_("rng", "null")
        )
        if seasons is not None:
            q = q.in_("season_year", seasons)
        resp = q.range(page * size, (page + 1) * size - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < size:
            break
        page += 1
    return rows


def report_supabase_validation(sb, seasons: list[int] | None) -> None:
    print("\n=== Step 8 — Supabase validation (rows with rng NOT NULL) ===")
    rows = fetch_players_defense_populated(sb, seasons)
    if not rows:
        print("  No rows with rng set.")
        return
    df = pd.DataFrame(rows)
    df["primary_position"] = df["primary_position"].fillna("(null)")
    print(f"  Population: {len(df)} player-seasons with defensive traits")
    print("\n  Mean / std by primary_position:")
    g = df.groupby("primary_position")[["rng", "hnd", "arm"]].agg(["mean", "std", "count"]).round(2)
    with pd.option_context("display.max_rows", 25, "display.width", 120):
        print(g)
    for col in ("rng", "hnd", "arm"):
        print(f"  Overall {col}: mean={df[col].mean():.2f}  std={df[col].std():.2f}")

    hi = df.loc[df["rng"].idxmax()]
    lo = df.loc[df["rng"].idxmin()]
    print("\n  Examples (by RNG):")
    print(f"    Elite: {hi.get('player_name')} ({hi.get('player_id')}) {hi.get('season_year')}  RNG={hi['rng']} HND={hi['hnd']} ARM={hi['arm']} pos={hi.get('primary_position')}")
    print(f"    Poor:  {lo.get('player_name')} ({lo.get('player_id')}) {lo.get('season_year')}  RNG={lo['rng']} HND={lo['hnd']} ARM={lo['arm']} pos={lo.get('primary_position')}")


def fetch_players_season(sb, season: int) -> list[dict]:
    rows: list[dict] = []
    page, size = 0, 1000
    while True:
        resp = (
            sb.table("players")
            .select("*")
            .eq("season_year", season)
            .not_.is_("player_id", "null")
            .range(page * size, (page + 1) * size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < size:
            break
        page += 1
    return rows


def validate_report(
    f_pos: pd.DataFrame,
    collapsed: pd.DataFrame,
    lahman_dir: str,
) -> None:
    """Print mean/std by POS (position-level pre-collapse traits) and summary stats."""
    print("\n=== Validation (local, pre-Supabase) ===")
    if f_pos.empty:
        print("  No fielding rows.")
        return
    work = f_pos.copy()
    work["RNG_t"] = work["z_range"].apply(z_to_trait)
    work["HND_t"] = work["z_hands"].apply(z_to_trait)
    work["ARM_t"] = work["z_arm"].apply(z_to_trait)
    work["RNG_shr"] = work.apply(lambda r: apply_shrinkage(r["RNG_t"], r["innings"]), axis=1)
    work["HND_shr"] = work.apply(lambda r: apply_shrinkage(r["HND_t"], r["innings"]), axis=1)
    work["ARM_shr"] = work.apply(lambda r: apply_shrinkage(r["ARM_t"], r["innings"]), axis=1)

    print("\n  Mean / std of shrunk traits by Lahman POS (all years in file):")
    g = work.groupby("POS")[["RNG_shr", "HND_shr", "ARM_shr"]].agg(["mean", "std"]).round(2)
    with pd.option_context("display.max_rows", 30, "display.width", 100):
        print(g)

    n_low = int(collapsed["low_rel"].sum())
    n_tot = len(collapsed)
    pct = 100.0 * n_low / n_tot if n_tot else 0.0
    print(f"\n  Player-seasons with any position (or total) below {MIN_INNINGS_LOW_RELIABILITY} inn. proxy: "
          f"{n_low} / {n_tot} ({pct:.1f}%) flagged low_reliability")

    # Example elite / poor by collapsed rng (join Lahman names when available)
    c = collapsed.dropna(subset=["rng"])
    if len(c) >= 2:
        i_max = c["rng"].idxmax()
        i_min = c["rng"].idxmin()
        ppl_path = os.path.join(lahman_dir, "People.csv")
        names: dict[str, str] = {}
        if os.path.isfile(ppl_path):
            ppl = pd.read_csv(ppl_path, usecols=["playerID", "nameFirst", "nameLast"])
            for _, pr in ppl.iterrows():
                pid = str(pr["playerID"])
                names[pid] = f"{pr.get('nameFirst', '')} {pr.get('nameLast', '')}".strip()
        print("\n  Example (global collapsed, all seasons): lowest / highest RNG")
        p_lo, p_hi = c.loc[i_min, "playerID"], c.loc[i_max, "playerID"]
        print(
            f"    Poor (min RNG):  {names.get(p_lo, p_lo)} {int(c.loc[i_min, 'yearID'])}  "
            f"RNG={int(c.loc[i_min, 'rng'])}"
        )
        print(
            f"    Elite (max RNG): {names.get(p_hi, p_hi)} {int(c.loc[i_max, 'yearID'])}  "
            f"RNG={int(c.loc[i_max, 'rng'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lahman_dir", default=None)
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Apply migrations/add_fielding_traits.sql using Postgres URI (SUPABASE_DB_URL or DATABASE_URL, etc.).",
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--validate_remote",
        action="store_true",
        help="Only query Supabase for rng/hnd/arm summary (after migration + optional prior push).",
    )
    parser.add_argument("--seasons", default=None, help="Comma years; default = all seasons present in Supabase")
    args = parser.parse_args()

    if args.migrate:
        apply_fielding_migration_db()
        print("  (--migrate complete)")
        if not args.push:
            print("\nDone.")
            return

    seasons_filter: list[int] | None = None
    if args.seasons:
        seasons_filter = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]

    if args.validate_remote:
        sb = get_supabase_client()
        if not check_defense_columns(sb):
            print("Columns rng/hnd/arm missing — run migration first.")
            sys.exit(1)
        seas = seasons_filter or fetch_supabase_seasons(sb)
        report_supabase_validation(sb, seas)
        print("\nDone.")
        return

    lahman_dir = pick_lahman_dir(args.lahman_dir)
    fpath = os.path.join(lahman_dir, "Fielding.csv")
    hdr = fielding_columns_ok(fpath)
    print("Fielding.csv columns:", hdr)
    expected = {"playerID", "yearID", "POS", "PO", "A", "E"}
    missing = expected - set(hdr)
    if missing:
        print("WARNING: missing columns:", missing)
    if "InnOuts" not in hdr:
        print("NOTE: InnOuts missing — using G fallback only.")

    agg = load_aggregate_fielding(lahman_dir)
    f_pos = build_position_level_frame(agg)
    if f_pos.empty:
        print("No usable fielding rows after filters.")
        sys.exit(1)
    f_pos = add_league_zscores(f_pos)
    collapsed = collapse_player_season(f_pos)
    validate_report(f_pos, collapsed, lahman_dir)

    sb = None
    if args.push:
        sb = get_supabase_client()
        if not check_defense_columns(sb):
            print("\n  Columns rng/hnd/arm missing. Options:\n"
                  "  • Run: python ingest_fielding_traits.py --migrate --push  (needs Postgres URI in .env)\n"
                  "  • Or paste migrations/add_fielding_traits.sql in the Supabase SQL editor, then re-run --push.")
            sys.exit(1)
        target_seasons = seasons_filter or fetch_supabase_seasons(sb)
        print(f"  Pushing rng/hnd/arm for seasons: {target_seasons}")
        coll_sub = collapsed[collapsed["yearID"].isin(target_seasons)].copy()
        map_ps = {
            (r["playerID"], int(r["yearID"])): (int(r["rng"]), int(r["hnd"]), int(r["arm"]))
            for _, r in coll_sub.iterrows()
            if pd.notna(r["rng"])
        }
        total_updated = 0
        grand_db = 0
        grand_matched = 0
        for sy in target_seasons:
            db_rows = fetch_players_season(sb, sy)
            grand_db += len(db_rows)
            patches = []
            for row in db_rows:
                pid = row.get("player_id")
                if not pid:
                    continue
                key = (pid, sy)
                if key not in map_ps:
                    continue
                rng, hnd, arm = map_ps[key]
                row = dict(row)
                row["rng"] = rng
                row["hnd"] = hnd
                row["arm"] = arm
                if "id" in row:
                    del row["id"]
                patches.append(row)
            grand_matched += len(patches)
            for i in range(0, len(patches), UPSERT_CHUNK):
                chunk = patches[i : i + UPSERT_CHUNK]
                sb.table("players").upsert(chunk, on_conflict="player_id,season_year").execute()
                total_updated += len(chunk)
            print(f"    Season {sy}: DB rows={len(db_rows)}, upserted (fielding match)={len(patches)}, skipped={len(db_rows) - len(patches)}")
        grand_skipped = grand_db - grand_matched
        print(f"\n  === Totals across {len(target_seasons)} seasons ===")
        print(f"    DB player-season rows:     {grand_db}")
        print(f"    Updated (fielding match):  {grand_matched}")
        print(f"    Skipped (no Lahman fielding match): {grand_skipped}")
        print(f"    Upsert payload rows sent:            {total_updated}")
        n_low_sub = int(coll_sub["low_rel"].sum())
        print(
            f"  Low-reliability (innings proxy) share among player-seasons we computed for these seasons: "
            f"{n_low_sub}/{len(coll_sub)} ({100.0 * n_low_sub / len(coll_sub):.1f}%)"
            if len(coll_sub)
            else "  (no collapsed rows for filter)"
        )
        report_supabase_validation(sb, target_seasons)
    elif args.dry_run:
        print("\n  --dry_run: no Supabase writes.")
    else:
        print("\n  No --push: computed traits only. Use --push to write (after migration).")

    print("\nDone.")


if __name__ == "__main__":
    main()
