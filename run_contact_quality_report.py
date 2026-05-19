"""
run_contact_quality_report.py — HR / contact-quality diagnostic.

For each matchup, runs N PAs via resolve_pa_seeded and collects:
  - outcome + contact_quality (Weak / Medium / Hard) for every BIP
  - HR split by contact quality
  - HR per Hard BIP
  - Hard contact rate (Hard BIPs / total BIPs)

Matchups:
  1. Avg Regular vs Avg Regular  (50k PAs)
  2. Gehrig 1927  vs Avg Pitcher  (50k PAs)
  3. Pedro 2000   vs Gehrig 1927  (50k PAs)

Usage:
    python run_contact_quality_report.py [--n 50000]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pa_wrapper import resolve_pa_seeded
from core_stress_test import (
    _avg_hitter,
    _avg_pitcher,
    build_card_hitter,
    build_card_pitcher,
)
from ingestion import LahmanAdapter

DEFAULT_FIELDER = {"RNG": 50, "HND": 50, "ARM": 50}


# ── Seeding ───────────────────────────────────────────────────────────────────

def _sha32(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 32)


# ── Data collection ───────────────────────────────────────────────────────────

@dataclass
class ContactStats:
    pa:        int = 0
    k:         int = 0
    bb:        int = 0
    hbp:       int = 0
    bip:       int = 0
    hard_bips: int = 0
    med_bips:  int = 0
    weak_bips: int = 0
    hr_total:  int = 0
    hr_hard:   int = 0
    hr_medium: int = 0
    hr_weak:   int = 0
    hits_on_bip: int = 0  # H on BIP (excl. HR) — numerator for BABIP

    def record(self, result: dict) -> None:
        self.pa += 1
        outcome = result["outcome"]
        contact = result["contact"]

        if contact is None:
            # Stage-1 outcome: K, BB, or HBP
            if outcome == "K":
                self.k += 1
            elif outcome == "BB":
                self.bb += 1
            elif outcome == "HBP":
                self.hbp += 1
            return

        self.bip += 1
        cq = contact.get("contact_quality", "")

        if cq == "Hard":
            self.hard_bips += 1
        elif cq == "Medium":
            self.med_bips += 1
        else:
            self.weak_bips += 1

        if outcome == "HR":
            self.hr_total += 1
            if cq == "Hard":
                self.hr_hard += 1
            elif cq == "Medium":
                self.hr_medium += 1
            else:
                self.hr_weak += 1
        elif outcome in ("Single", "Double", "Triple"):
            self.hits_on_bip += 1

    def report(self, label: str) -> str:
        hr_pa        = self.hr_total  / self.pa      if self.pa        else 0.0
        hard_rate    = self.hard_bips / self.bip     if self.bip       else 0.0
        hr_per_hbip  = self.hr_hard   / self.hard_bips if self.hard_bips else 0.0
        k_pct        = self.k  / self.pa             if self.pa        else 0.0
        bb_pct       = self.bb / self.pa             if self.pa        else 0.0
        # BABIP = H-on-BIP / (BIP - HR); errors excluded per standard def
        babip_denom  = self.bip - self.hr_total
        babip        = self.hits_on_bip / babip_denom if babip_denom > 0 else 0.0

        hr_from_hard = self.hr_hard   / self.hr_total * 100 if self.hr_total else 0.0
        hr_from_med  = self.hr_medium / self.hr_total * 100 if self.hr_total else 0.0
        hr_from_weak = self.hr_weak   / self.hr_total * 100 if self.hr_total else 0.0

        lines = [
            f"  ┌─ {label}",
            f"  │  PA={self.pa:,}  BIP={self.bip:,}  HR={self.hr_total:,}",
            f"  │",
            f"  │  Hard contact rate   : {hard_rate:.3f}  ({self.hard_bips:,} Hard / {self.bip:,} BIP)",
            f"  │  HR per Hard BIP     : {hr_per_hbip:.4f}  ({self.hr_hard:,} HR / {self.hard_bips:,} Hard BIPs)",
            f"  │  HR/PA               : {hr_pa:.4f}",
            f"  │",
            f"  │  HR by contact quality:",
            f"  │    Hard   → {self.hr_hard:>5,}  ({hr_from_hard:5.1f}% of HRs)",
            f"  │    Medium → {self.hr_medium:>5,}  ({hr_from_med:5.1f}% of HRs)",
            f"  │    Weak   → {self.hr_weak:>5,}  ({hr_from_weak:5.1f}% of HRs)",
            f"  │",
            f"  │  Stage-1 / BABIP (locked baselines):",
            f"  │    K%    = {k_pct:.3f}   BB%   = {bb_pct:.3f}   BABIP = {babip:.3f}",
            f"  └{'─'*60}",
        ]
        return "\n".join(lines)


# ── Simulation ────────────────────────────────────────────────────────────────

def run_matchup(
    batter:  dict,
    pitcher: dict,
    n_pa:    int,
    tag:     str,
) -> ContactStats:
    base    = _sha32(f"{batter['player_id']}|{pitcher['player_id']}")
    context = {"fielder": DEFAULT_FIELDER}
    stats   = ContactStats()

    for i in range(n_pa):
        seed   = _sha32(f"{base}:{i}")
        result = resolve_pa_seeded(batter, pitcher, context=context, seed=seed)
        stats.record(result)

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HR / contact-quality diagnostic")
    parser.add_argument("--n", type=int, default=50_000, help="PAs per matchup (default: 50,000)")
    args = parser.parse_args()
    n_pa = args.n

    print(f"\n{'═'*64}")
    print(f"  Contact Quality × HR Diagnostic  —  {n_pa:,} PAs each")
    print(f"{'═'*64}\n")

    adapter = LahmanAdapter()

    matchups: list[tuple[str, dict, dict]] = [
        (
            "Avg Regular vs Avg Pitcher",
            _avg_hitter("avg_regular_bat"),
            _avg_pitcher("avg_regular_pit"),
        ),
        (
            "Gehrig 1927 vs Avg Pitcher",
            build_card_hitter("gehrilo01", 1927, adapter),
            _avg_pitcher("avg_regular_pit"),
        ),
        (
            "Gehrig 1927 vs Pedro 2000",
            build_card_hitter("gehrilo01", 1927, adapter),
            build_card_pitcher("martipe02", 2000, adapter),
        ),
    ]

    all_stats: list[tuple[str, ContactStats]] = []
    for label, batter, pitcher in matchups:
        print(f"  Simulating: {label} …")
        stats = run_matchup(batter, pitcher, n_pa, label)
        all_stats.append((label, stats))
        print(stats.report(label))
        print()

    # ── Summary comparison table ──────────────────────────────────────────────
    print(f"{'═'*80}")
    print(f"  {'Matchup':<35}  {'Hard%':>6}  {'HR/HBIP':>7}  {'HR/PA':>6}  {'Med%HR':>6}  {'K%':>5}  {'BB%':>5}  {'BABIP':>5}")
    print(f"  {'─'*35}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}")

    for label, stats in all_stats:
        hard_rate   = stats.hard_bips  / stats.bip      if stats.bip       else 0.0
        hr_hbip     = stats.hr_hard    / stats.hard_bips if stats.hard_bips else 0.0
        hr_pa       = stats.hr_total   / stats.pa        if stats.pa        else 0.0
        med_pct     = stats.hr_medium  / stats.hr_total * 100 if stats.hr_total else 0.0
        k_pct       = stats.k  / stats.pa               if stats.pa        else 0.0
        bb_pct      = stats.bb / stats.pa               if stats.pa        else 0.0
        babip_d     = stats.bip - stats.hr_total
        babip       = stats.hits_on_bip / babip_d if babip_d > 0 else 0.0
        print(
            f"  {label:<35}  {hard_rate:>6.3f}  {hr_hbip:>7.4f}  {hr_pa:>6.4f}"
            f"  {med_pct:>5.1f}%  {k_pct:>5.3f}  {bb_pct:>5.3f}  {babip:>5.3f}"
        )

    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()
