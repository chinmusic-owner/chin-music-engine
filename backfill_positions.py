"""
backfill_positions.py — one-time backfill for pitcher_role + primary_position

Reads every historical player-season already in the Supabase `players` table,
computes pitcher_role (SP/RP) and primary_position (C/1B/IF/OF/UTIL) from
local Lahman CSVs, and patches the rows in place.

Schema migration (two options — pick one):

  OPTION A — Supabase Dashboard (simplest):
    Open https://supabase.com/dashboard → your project → SQL Editor, paste and run:
        ALTER TABLE players
          ADD COLUMN IF NOT EXISTS pitcher_role     text,
          ADD COLUMN IF NOT EXISTS primary_position text;
    Then run this script normally.

  OPTION B — Automatic via direct Postgres connection:
    Add SUPABASE_DB_URL to your .env file:
        SUPABASE_DB_URL=postgresql://postgres.[project_ref]:[db_password]@aws-0-[region].pooler.supabase.com:6543/postgres
    (Find it in Supabase Dashboard → Project Settings → Database → Connection string)
    Then run this script — it will apply the migration automatically.

Usage:
    python backfill_positions.py [--lahman_dir PATH] [--dry_run]

    --lahman_dir  Path to the Lahman CSV directory (default: auto-detect)
    --dry_run     Print what would be patched without writing to Supabase
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# Reuse the derivation logic that was added to ingest_historical.py
from ingest_historical import (
    compute_pitcher_roles,
    compute_primary_positions,
    pick_lahman_dir,
)

LAHMAN_PITCHING = "Pitching.csv"
LAHMAN_FIELDING = "Fielding.csv"

UPSERT_CHUNK = 200   # rows per Supabase upsert call

MIGRATION_SQL = """
ALTER TABLE players
  ADD COLUMN IF NOT EXISTS pitcher_role     text,
  ADD COLUMN IF NOT EXISTS primary_position text;
""".strip()


# ─── Clients ──────────────────────────────────────────────────────────────────

def get_supabase_client():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY in your .env file.")
    return create_client(url, key)


def apply_migration_via_postgres(db_url: str) -> None:
    """Run the ALTER TABLE migration directly against Postgres using psycopg2."""
    try:
        import psycopg2
    except ImportError:
        raise RuntimeError("psycopg2-binary is required for automatic migration: pip install psycopg2-binary")

    print(f"Connecting to Postgres to apply migration ...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(MIGRATION_SQL)
    conn.close()
    print("  Migration applied successfully.")


def check_columns_exist(sb) -> bool:
    """
    Returns True if both pitcher_role and primary_position already exist.
    Uses a harmless SELECT that will fail with PGRST204 if either column is missing.
    """
    try:
        sb.table("players").select("pitcher_role, primary_position").limit(1).execute()
        return True
    except Exception as exc:
        if "PGRST204" in str(exc) or "column" in str(exc).lower():
            return False
        raise


# ─── Fetch helpers ────────────────────────────────────────────────────────────

def fetch_all_historical(sb) -> list[dict]:
    """
    Returns all rows from `players` that have a Lahman player_id (historical).
    Seeded/demo rows (player_id = null) are excluded automatically.
    """
    rows: list[dict] = []
    page_size = 1000
    offset    = 0

    while True:
        resp = (
            sb.table("players")
            .select("id, player_id, season_year, stuff")
            .not_.is_("player_id", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill pitcher_role + primary_position")
    parser.add_argument("--lahman_dir", default=None)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print patch counts without writing to Supabase")
    args = parser.parse_args()

    lahman_dir    = pick_lahman_dir(args.lahman_dir)
    pitching_path = os.path.join(lahman_dir, LAHMAN_PITCHING)
    fielding_path = os.path.join(lahman_dir, LAHMAN_FIELDING)

    for path in [pitching_path, fielding_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Lahman file: {path}")

    # ── Build lookup tables ───────────────────────────────────────────────────
    print("Loading Lahman Pitching.csv ...")
    pitching = pd.read_csv(pitching_path)
    pitcher_roles = compute_pitcher_roles(pitching)
    print(f"  → {len(pitcher_roles):,} pitcher-season role assignments computed")

    print("Loading Lahman Fielding.csv ...")
    fielding = pd.read_csv(fielding_path)
    primary_positions = compute_primary_positions(fielding)
    print(f"  → {len(primary_positions):,} hitter-season primary positions computed")

    # ── Connect + apply migration if needed ──────────────────────────────────
    load_dotenv()
    sb = get_supabase_client()

    if not check_columns_exist(sb):
        db_url = os.getenv("SUPABASE_DB_URL")
        if db_url:
            apply_migration_via_postgres(db_url)
        else:
            print("\n" + "═" * 68)
            print("  SCHEMA MIGRATION REQUIRED")
            print("  The players table is missing pitcher_role / primary_position.")
            print()
            print("  OPTION A — run this SQL in the Supabase Dashboard → SQL Editor:")
            print()
            print(f"    {MIGRATION_SQL.replace(chr(10), chr(10) + '    ')}")
            print()
            print("  OPTION B — add SUPABASE_DB_URL to your .env and re-run:")
            print("    SUPABASE_DB_URL=postgresql://postgres.[ref]:[pwd]@...")
            print("═" * 68)
            sys.exit(1)

    # ── Fetch existing DB rows ────────────────────────────────────────────────
    print("\nFetching historical rows from Supabase ...")
    rows = fetch_all_historical(sb)
    print(f"  → {len(rows)} historical player-seasons found")

    # ── Build patch list ──────────────────────────────────────────────────────
    patches:        list[dict] = []
    no_role_match:  int = 0
    no_pos_match:   int = 0

    for row in rows:
        pid     = row["player_id"]
        year    = int(row["season_year"])
        stuff   = row.get("stuff")
        key     = (pid, year)

        is_pitcher = stuff is not None and stuff > 0

        if is_pitcher:
            role = pitcher_roles.get(key)
            if role:
                patches.append({
                    "player_id":        pid,
                    "season_year":      year,
                    "pitcher_role":     role,
                    "primary_position": None,
                })
            else:
                no_role_match += 1
        else:
            pos = primary_positions.get(key)
            if pos:
                patches.append({
                    "player_id":        pid,
                    "season_year":      year,
                    "pitcher_role":     None,
                    "primary_position": pos,
                })
            else:
                no_pos_match += 1

    pitchers_patched = sum(1 for p in patches if p["pitcher_role"])
    hitters_patched  = sum(1 for p in patches if p["primary_position"])
    print(f"\nPatch summary:")
    print(f"  Pitchers with role   : {pitchers_patched}")
    print(f"  Hitters with position: {hitters_patched}")
    print(f"  No role match        : {no_role_match}  (pitcher in DB but not in Pitching.csv)")
    print(f"  No position match    : {no_pos_match}  (hitter in DB but not in Fielding.csv)")

    if args.dry_run:
        # Show a sample from each group
        sp_ex = next((p for p in patches if p.get("pitcher_role") == "SP"), None)
        rp_ex = next((p for p in patches if p.get("pitcher_role") == "RP"), None)
        c_ex  = next((p for p in patches if p.get("primary_position") == "C"), None)
        of_ex = next((p for p in patches if p.get("primary_position") == "OF"), None)
        if_ex = next((p for p in patches if p.get("primary_position") == "IF"), None)
        print("\nSample patches (dry run — nothing written):")
        for label, ex in [("SP", sp_ex), ("RP", rp_ex), ("C", c_ex), ("OF", of_ex), ("IF", if_ex)]:
            if ex:
                print(f"  [{label}] player_id={ex['player_id']}  year={ex['season_year']}  "
                      f"pitcher_role={ex['pitcher_role']!r}  primary_position={ex['primary_position']!r}")
        print("\nRe-run without --dry_run to write changes.")
        return

    # ── Upsert in chunks ─────────────────────────────────────────────────────
    print(f"\nUpserting {len(patches)} rows in chunks of {UPSERT_CHUNK} ...")
    for start in range(0, len(patches), UPSERT_CHUNK):
        chunk = patches[start : start + UPSERT_CHUNK]
        sb.table("players").upsert(
            chunk,
            on_conflict="player_id,season_year",
        ).execute()
        end = min(start + UPSERT_CHUNK, len(patches))
        print(f"  [{end:>5}/{len(patches)}] upserted")

    print(f"\nBackfill complete — {len(patches)} rows patched.")
    print("  Rerun salary_report_clean.py to see updated position groups.")


if __name__ == "__main__":
    main()
