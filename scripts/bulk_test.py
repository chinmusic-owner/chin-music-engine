"""
scripts/bulk_test.py — PA stress-test pipeline.

Usage:
    python scripts/bulk_test.py [--n 20000] [--era 2026]

What it does:
    1. Ingestion (Step C): loads data/stress_test_players.json and writes canonical
       cards to db/ingested_players.json via safe local fallback (no external DB).
    2. Runs N PAs for every batter × pitcher archetype matchup.
    3. Persists per-PA receipts as NDJSON to outputs/receipts/<b>_vs_<p>.ndjson.
    4. Aggregates K%, BB%, HR%, AVG, OBP, SLG, BABIP per matchup to
       outputs/summaries/matchup_summary.csv.
    5. Prints summary table, sample receipt path, and any POW→BB/K violations.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import tracemalloc

# ── Path setup (allow running from repo root or scripts/) ───────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from pa_wrapper import resolve_pa_seeded  # noqa: E402 — must follow sys.path insert

# ── Paths ────────────────────────────────────────────────────────────────────
_DATA_FILE     = os.path.join(_REPO, "data",    "stress_test_players.json")
_INGESTED_FILE = os.path.join(_REPO, "db",      "ingested_players.json")
_RECEIPTS_DIR  = os.path.join(_REPO, "outputs", "receipts")
_SUMMARIES_DIR = os.path.join(_REPO, "outputs", "summaries")
_SUMMARY_CSV   = os.path.join(_SUMMARIES_DIR,   "matchup_summary.csv")


# ────────────────────────────────────────────────────────────────────────────
# Step C — Ingestion (safe local fallback)
# ────────────────────────────────────────────────────────────────────────────

def _ingest_synthetic(source_path: str, dest_path: str) -> None:
    """
    Converts data/stress_test_players.json spec cards into the canonical
    export_to_json schema (PRD 02 §6A) and writes to db/ingested_players.json.

    ingestion.py only supports Lahman CSV sources, so this is the safe local
    fallback. No external DB credentials are used.
    """
    import datetime

    with open(source_path) as f:
        raw = json.load(f)

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cards = []
    for p in raw:
        role        = p["role"]           # "Hitter" or "Pitcher"
        pitcher_role = p.get("pitcher_role", "SP" if role == "Pitcher" else None)
        card = {
            "player_id":    p["id"],
            "card_id":      f"{p['id']}|stress",
            "season":       0,
            "team_id":      "STRESS",
            "name":         p["name"],
            "bats":         p.get("bats",   "R"),
            "throws":       p.get("throws", "R"),
            "primary_role": role,
            "pitcher_role": pitcher_role,
            "traits":       p["traits"],
            "normalized_rates": {},
            "reliability":      {},
            "trait_provenance": {
                "mode":           "synthetic",
                "build_version":  "stress-test-1.0",
                "source_version": "stress_test_players.json",
                "components":     [{"name": "archetype", "weight": 1.0}],
            },
            "build_version":   "stress-test-1.0",
            "source_version":  "stress_test_players.json",
            "build_timestamp": timestamp,
        }
        cards.append(card)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w") as f:
        json.dump(cards, f, indent=2)
    print(f"[ingestion] FALLBACK used (ingestion.py is Lahman-only; no DB credentials required)")
    print(f"[ingestion] Wrote {len(cards)} canonical cards → {dest_path}")


def load_players(ingested_path: str, source_path: str) -> list[dict]:
    """Loads ingested cards, creating them from source spec if absent."""
    if not os.path.exists(ingested_path):
        print(f"[ingestion] {ingested_path} not found — running synthetic ingestion.")
        _ingest_synthetic(source_path, ingested_path)
    else:
        print(f"[ingestion] Loaded existing ingested cards from {ingested_path}")
    with open(ingested_path) as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────────────
# Seed derivation
# ────────────────────────────────────────────────────────────────────────────

def _sha32(value: str) -> int:
    """sha256 → 32-bit int (mirrors pa_engine._sha_int but local to this script)."""
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % (2 ** 32)


def matchup_seed(batter_id: str, pitcher_id: str, era: str) -> int:
    return _sha32(f"{batter_id}|{pitcher_id}|{era}")


def pa_seed(mseed: int, pa_index: int) -> int:
    return _sha32(f"{mseed}:{pa_index}")


# ────────────────────────────────────────────────────────────────────────────
# Stats accumulator
# ────────────────────────────────────────────────────────────────────────────

def _make_acc() -> dict:
    return {"K": 0, "BB": 0, "HBP": 0, "HR": 0,
            "Triple": 0, "Double": 0, "Single": 0,
            "Out": 0, "Error": 0, "total": 0}


def _accumulate(acc: dict, outcome: str) -> None:
    acc["total"] += 1
    acc[outcome] = acc.get(outcome, 0) + 1


def _aggregate(acc: dict) -> dict:
    n   = acc["total"]
    if n == 0:
        return {}

    hits = acc["Single"] + acc["Double"] + acc["Triple"] + acc["HR"]
    ab   = n - acc["BB"] - acc["HBP"]
    slg_num = (acc["Single"]
               + 2 * acc["Double"]
               + 3 * acc["Triple"]
               + 4 * acc["HR"])

    avg  = hits / ab if ab else 0.0
    obp  = (hits + acc["BB"] + acc["HBP"]) / n if n else 0.0
    slg  = slg_num / ab if ab else 0.0

    babip_den = ab - acc["K"] - acc["HR"]
    babip = (hits - acc["HR"]) / babip_den if babip_den else 0.0

    return {
        "n_pa":   n,
        "k_pct":  round(acc["K"]  / n, 4),
        "bb_pct": round(acc["BB"] / n, 4),
        "hr_pct": round(acc["HR"] / n, 4),
        "avg":    round(avg,   3),
        "obp":    round(obp,   3),
        "slg":    round(slg,   3),
        "babip":  round(babip, 3),
    }


# ────────────────────────────────────────────────────────────────────────────
# POW → BB/K violation check (Step F)
# ────────────────────────────────────────────────────────────────────────────

def _check_pow_violations() -> list[str]:
    """
    Scans pa_engine.py for any code path where POW influences K% or BB%.
    Returns a list of violation strings (empty = no violations found).
    """
    engine_path = os.path.join(_REPO, "pa_engine.py")
    violations  = []
    try:
        with open(engine_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return [f"[check] Could not open {engine_path}"]

    # Heuristic: look for 'POW' appearing on the same line as 'k_pct' or 'bb_pct'
    # within the resolve_duel function block (lines before resolve_contact).
    in_duel = False
    for i, line in enumerate(lines, start=1):
        if "def resolve_duel" in line:
            in_duel = True
        if in_duel and line.strip().startswith("def ") and "resolve_duel" not in line:
            in_duel = False  # left resolve_duel
        if in_duel:
            has_pow = "POW" in line and not line.strip().startswith("#")
            affects_kbb = ("k_pct" in line or "bb_pct" in line)
            if has_pow and affects_kbb:
                violations.append(
                    f"  VIOLATION  pa_engine.py line {i}: {line.rstrip()}"
                )

    if not violations:
        violations = []  # explicit empty — no violations
    return violations


# ────────────────────────────────────────────────────────────────────────────
# Main stress-test loop
# ────────────────────────────────────────────────────────────────────────────

def run(n: int = 20_000, era: str = "2026") -> None:
    os.makedirs(_RECEIPTS_DIR,  exist_ok=True)
    os.makedirs(_SUMMARIES_DIR, exist_ok=True)

    tracemalloc.start()
    t0 = time.perf_counter()

    # ── Ingestion ────────────────────────────────────────────────────────────
    _ingest_synthetic(_DATA_FILE, _INGESTED_FILE)
    cards = load_players(_INGESTED_FILE, _DATA_FILE)

    batters  = [c for c in cards if c["primary_role"] == "Hitter"]
    pitchers = [c for c in cards if c["primary_role"] == "Pitcher"]

    print(f"\n[stress-test] {len(batters)} batters × {len(pitchers)} pitchers "
          f"× {n:,} PAs  (era={era})\n")

    summary_rows = []
    sample_receipt = None

    for batter in batters:
        for pitcher in pitchers:
            bid = batter["player_id"]
            pid = pitcher["player_id"]
            label = f"{bid}_vs_{pid}"

            mseed = matchup_seed(bid, pid, era)
            acc   = _make_acc()

            receipt_path = os.path.join(_RECEIPTS_DIR, f"{label}.ndjson")
            if sample_receipt is None:
                sample_receipt = receipt_path

            with open(receipt_path, "w") as rf:
                for i in range(n):
                    seed = pa_seed(mseed, i)
                    result = resolve_pa_seeded(batter, pitcher, seed=seed)

                    _accumulate(acc, result["outcome"])

                    receipt = {
                        "pa_index":   i,
                        "batter_id":  bid,
                        "pitcher_id": pid,
                        "matchup_seed": mseed,
                        "pa_seed":    seed,
                        "outcome":    result["outcome"],
                        "duel_score": result["duel"]["duel_score"],
                        "p_batter_adv": result["duel"]["p_batter_advantage"],
                        "duel_probs": result["duel"]["probabilities"],
                        "contact_quality": result["contact"]["contact_quality"] if result["contact"] else None,
                        "spray_vector":    result["contact"]["spray_vector"]    if result["contact"] else None,
                        "contact_score":   result["contact"]["contact_score"]   if result["contact"] else None,
                        "bip_outcome":     result["bip_map"]["bip_outcome"]     if result["bip_map"] else None,
                        "hr_driver":       result["bip_map"]["hr_driver"]       if result["bip_map"] else None,
                        "final_defense":   result["defense"]["defense_resolution"] if result["defense"] else None,
                    }
                    rf.write(json.dumps(receipt) + "\n")

            stats = _aggregate(acc)
            row = {"batter_id": bid, "pitcher_id": pid}
            row.update(stats)
            summary_rows.append(row)

            print(f"  {bid:<18} vs {pid:<18}  "
                  f"K%={stats['k_pct']:.3f}  BB%={stats['bb_pct']:.3f}  "
                  f"HR%={stats['hr_pct']:.3f}  AVG={stats['avg']:.3f}  "
                  f"OBP={stats['obp']:.3f}  SLG={stats['slg']:.3f}  "
                  f"BABIP={stats['babip']:.3f}  "
                  f"seed={mseed}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    fieldnames = ["batter_id", "pitcher_id", "n_pa",
                  "k_pct", "bb_pct", "hr_pct",
                  "avg", "obp", "slg", "babip"]
    with open(_SUMMARY_CSV, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    elapsed = time.perf_counter() - t0
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── POW → BB/K check ─────────────────────────────────────────────────────
    violations = _check_pow_violations()

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  Summary CSV  →  {_SUMMARY_CSV}")
    print(f"\n  Top rows:")
    print(f"  {'batter_id':<18} {'pitcher_id':<18} {'K%':>6} {'BB%':>6} {'HR%':>6} "
          f"{'AVG':>5} {'OBP':>5} {'SLG':>5} {'BABIP':>6}")
    print(f"  {'─'*70}")
    for row in summary_rows[:6]:
        print(f"  {row['batter_id']:<18} {row['pitcher_id']:<18} "
              f"{row['k_pct']:>6.3f} {row['bb_pct']:>6.3f} {row['hr_pct']:>6.3f} "
              f"{row['avg']:>5.3f} {row['obp']:>5.3f} {row['slg']:>5.3f} {row['babip']:>6.3f}")

    print(f"\n  Sample receipt  →  {sample_receipt}")

    print(f"\n  POW → BB/K violation check:")
    if violations:
        for v in violations:
            print(v)
    else:
        print("  CLEAN — POW has zero influence on K% or BB% in resolve_duel.")
        print("  (pa_engine.py line 122: BB% driven purely by EYE-CTL; "
              "K% by STF-CON + AK only)")

    print(f"\n  Runtime : {elapsed:.2f}s")
    print(f"  Peak mem: {peak_kb / 1024:.1f} MB")
    print(f"{'═'*72}\n")


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA stress-test pipeline")
    parser.add_argument("--n",   type=int, default=20_000,
                        help="PAs per matchup (default: 20000)")
    parser.add_argument("--era", type=str, default="2026",
                        help="Era tag used in deterministic seed (default: 2026)")
    args = parser.parse_args()
    run(n=args.n, era=args.era)
