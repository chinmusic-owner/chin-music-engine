# Chin Music Engine — Simulation Technical Reference

**Audience:** Engineers building on or extending the simulation stack.

**Engine status:** PA Engine FROZEN as of calibration session May 2026. Game Engine, Roster Manager, and simulation tooling are active. Do not modify `pa_engine.py` or `sim_constants.json` calibration constants without running a new mid-tier validation.

**Last updated:** May 19, 2026 — 1954 AL historical replay validation complete.

---

## Table of Contents

1. [PA Engine — Inputs & Outputs](#1-pa-engine)
2. [Game Engine](#2-game-engine)
3. [Roster Manager](#3-roster-manager)
4. [Simulation Tooling](#4-simulation-tooling)
5. [Historical Validation — 1954 AL Replay](#5-historical-validation)
6. [Calibration Log](#6-calibration-log)

---

## 1. PA Engine

### Entry point

```python
from pa_wrapper import resolve_pa_seeded
result = resolve_pa_seeded(batter, pitcher, context=None, seed=0)
```

All simulation runs go through `pa_wrapper.resolve_pa_seeded`. Do not call `pa_engine` stages directly from outside modules.

---

### `batter` — dict

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `"traits"` | `dict` | **Yes** | Flat dict of batting trait keys (see below) |
| `"bats"` | `str` | Recommended | Handedness: `"R"`, `"L"`, or `"B"`. Defaults to `"R"` if absent. |

**Batter trait keys** (all integers, scale 1–99):

| Key | Trait | Engine role |
|-----|-------|-------------|
| `"CON"` | Contact | Stage 1 (K% suppression via `CON − STF` edge) |
| `"EYE"` | Plate discipline | Stage 1 (BB% via `EYE − CTL` edge) |
| `"AK"` | Anti-K / bat-to-ball | Stage 1 (primary K% adjustment; high AK suppresses Ks, low AK inflates them) |
| `"POW"` | Raw power | Stage 2.5 (HR gate — `POW^2.5` term; raw POW, not CON-gated) |
| `"GAP"` | Gap power | Stage 2 (contact score contribution) + Stage 2.5 (Double probability shift) |

> `BNT` (Bunt) exists on `PlayerCard` but is **not read by `pa_engine`**. The Game Engine is responsible for modeling bunt decisions.

---

### `pitcher` — dict

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `"traits"` | `dict` | **Yes** | Flat dict of pitching trait keys (see below) |
| `"throws"` | `str` | Recommended | Handedness: `"R"` or `"L"`. Defaults to `"R"` if absent. |

**Pitcher trait keys** (all integers, scale 1–99):

| Key | Trait | Engine role |
|-----|-------|-------------|
| `"STF"` | Stuff / velocity | Stage 1 (K% via raw STF dominance term) + Stage 2 (contact score suppressor) |
| `"CTL"` | Control | Stage 1 (BB% via `CTL − EYE` edge; HBP base when CTL is low) |
| `"CMD"` | Command precision | Stage 2 (variance on contact quality — low CMD widens distribution) + Stage 2.5 (HR mistake-factor) |
| `"STA"` | Stamina | **Not read by `pa_engine`** — reserved for the Game Engine's fatigue model |

> `STA` is deliberately excluded from all `pa_engine` calculations. The Game Engine owns fatigue and decline curves; it adjusts the pitcher card passed to `resolve_pa_seeded` based on pitch count.

---

### `context` — dict (optional)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"fielder"` | `dict` | `{"RNG": 50, "HND": 50, "ARM": 50}` | Fielder traits for Stage 3 |
| `"constants"` | `dict` | Loaded from `sim_constants.json` | Full constants override — for testing only |

**Fielder trait keys:**

| Key | Trait | Engine role |
|-----|-------|-------------|
| `"RNG"` | Range | Stage 3: probability of reaching ball in play (affects Out/Single conversion) |
| `"HND"` | Hands | Stage 3: error probability on any reached ball |
| `"ARM"` | Arm strength | Stage 3 + `resolve_baserunning()`: holding runners on extra-base attempts |

```python
from pa_wrapper import fielder_from_row
fielder = fielder_from_row(supabase_player_row)  # falls back to 50 with warning if null
```

---

### `seed` — int

Any non-negative integer. The same `(batter, pitcher, context, seed)` tuple always produces the exact same outcome.

```python
import hashlib

def derive_seed(game_id: str, pa_index: int) -> int:
    raw = hashlib.sha256(f"{game_id}:{pa_index}".encode()).hexdigest()
    return int(raw, 16) % (2 ** 32)
```

---

### Outputs

`resolve_pa_seeded` returns a single `dict`:

| Key | Type | Description |
|-----|------|-------------|
| `"outcome"` | `str` | **The only field the Game Engine needs for game state.** See values below. |
| `"seed"` | `int` | Echo of input seed |
| `"duel"` | `dict` | Stage 1 internals |
| `"contact"` | `dict \| None` | Stage 2 internals. `None` if Stage 1 resolved to K/BB/HBP. |
| `"bip_map"` | `dict \| None` | Stage 2.5 internals. `None` if Stage 1 resolved to K/BB/HBP. |
| `"defense"` | `dict \| None` | Stage 3 internals. `None` if Stage 1 resolved to K/BB/HBP. |

**All possible `"outcome"` values:**

| Value | Category | Base-runner effect |
|-------|----------|--------------------|
| `"K"` | Out | Strikeout |
| `"BB"` | On base | Force advances apply |
| `"HBP"` | On base | Force advances apply |
| `"HR"` | Score | All runners + batter score; bases clear |
| `"Single"` | On base | Game Engine advances runners via baserunning rules |
| `"Double"` | On base | Game Engine advances runners via baserunning rules |
| `"Triple"` | On base | Game Engine advances runners via baserunning rules |
| `"Out"` | Out | Generic out; Game Engine applies DP/force logic |
| `"Error"` | On base | Batter reaches first; does not count as a hit |

> `pa_engine` has no concept of sacrifice fly, double play, fielder's choice, or passed ball. All are Game Engine responsibilities.

---

### PA Engine constants (pa_engine.py)

**K% formula:**
```
k_pct = k_base
      + K_STF_ABSOLUTE × stf_dominance      (pitcher raw stuff, matchup-independent)
      + K_STF_CON_COEFF × stf_con_edge      (relative STF−CON matchup, can be negative)
      − ak_suppression                       (high AK: batter bat-to-ball skill)
      + ak_boost                             (low AK: poor bat-to-ball)
```

| Constant | Value | Role |
|----------|-------|------|
| `league_avg_k_pct` *(sim_constants.json)* | `0.165` | Baseline K rate; primary dial for era K% |
| `K_STF_ABSOLUTE` | `0.25` | Raw stuff contribution to K%; matchup-independent |
| `K_STF_CON_COEFF` | `0.09` | Relative STF−CON edge multiplier; reduced from 0.20 to fix era equity |

**Contact quality:**

| Constant | Value | Role |
|----------|-------|------|
| `_HARD_Q_MID` | `30.0` | Q_noisy where P(Hard) = 35%; raised from 28.0 to reduce Hard% by ~8pp |
| `_HARD_Q_SCALE` | `3.0` | Steepness of logistic; do not go below 2.5 |
| `_HARD_FLOOR / _HARD_CEIL` | `0.10 / 0.60` | Hard contact probability bounds |
| `_WEAK_Q_MID` | `35.0` | Q_noisy where P(Weak) = 21.5% |

**Non-HR hit distribution by contact tier** (`_NON_HR_BASE`):

| Tier | Out | Single | Double | Triple | Error |
|------|-----|--------|--------|--------|-------|
| Weak | 0.830 | 0.140 | 0.020 | 0.005 | 0.005 |
| Medium | 0.740 | 0.205 | 0.040 | 0.008 | 0.007 |
| Hard | 0.680 | 0.190 | 0.130 | 0.015 | 0.015 |

> The 2B→1B shifts are BABIP-neutral: total hit weight (1B+2B+3B) is preserved within each tier. Only SLG changes.

**HR gate:**

| Constant | Value | Role |
|----------|-------|------|
| `_HR_POW_SCALE` | `0.12` | Main coefficient: `POW^2.5` → HR probability on Hard contact |
| `_HARD_HR_FLOOR_BONUS` | `0.040` | Additive constant on Hard-contact HR only; preserves elite power identity |

---

### Statelessness guarantee

- No global mutable state; `sim_constants.json` read once at import.
- No inning/score/base-state context — resolves one matchup and returns.
- No side effects.
- Full determinism: identical inputs → bit-for-bit identical output on any machine, Python ≥ 3.10.
- Thread-safe: no shared mutable state.

---

## 2. Game Engine

**File:** `game_engine.py`

### Data structures

```python
@dataclass
class PAEvent:
    inning:          int
    half:            str      # "top" (away bats) or "bottom" (home bats)
    pa_number:       int      # 1-indexed global PA counter
    batter_id:       str
    batter_name:     str
    pitcher_id:      str
    pitcher_name:    str
    outcome:         str      # resolved outcome string from pa_wrapper
    outs_before:     int      # outs at start of PA (0, 1, or 2)
    bases_before:    list     # [1B_occ, 2B_occ, 3B_occ] bool snapshot
    runs_scored:     int      # runs that crossed on this PA
    seed:            int      # PA seed (for full deterministic replay)
    raw_result:      dict     # full dict from resolve_pa_seeded (stage metadata)
    baserunning_note: str     # narrative for any extra-base attempt

@dataclass
class BoxScore:
    away_team_id:   str
    home_team_id:   str
    final_score:    dict     # {"away": int, "home": int}
    linescore:      dict     # {"away": [int, …], "home": [int, …]} — one per inning
    pa_events:      list     # list[PAEvent]
    innings_played: int
    walk_off:       bool
```

---

### Base advancement rules

| Outcome | 3B runner | 2B runner | 1B runner | Batter |
|---------|-----------|-----------|-----------|--------|
| Single | Scores | Scores | → 2B | → 1B |
| Double | Scores | Scores | → 3B | → 2B |
| Triple | Scores | Scores | Scores | → 3B |
| HR | Scores | Scores | Scores | Scores |
| BB / HBP | Scores if bases loaded | → 3B if forced | → 2B if forced | → 1B |
| Out / K | — | — | — | Out; outs++ |

Walk/HBP forces are applied in sequence: 1B→2B only if batter walks and 1B was occupied; 2B→3B only if both 1B and 2B were occupied; 3B scores only if bases were loaded.

---

### Groundball logic

After `resolve_pa_seeded` returns `"Out"`, the Game Engine classifies the ball in play using `contact_quality` from `result["contact"]`:

**Infield hit probability** (before resolving out/double play):

| Contact tier | Base IH% | SPD modifier |
|--------------|----------|--------------|
| Hard | 1.0% | ±0.00225 per point from SPD 50 |
| Medium | 3.5% | ±0.00225 per point from SPD 50 |
| Weak | 9.0% | ±0.00225 per point from SPD 50 |

If an infield hit occurs: batter safe at 1B, runners advance conservatively, no double play possible.

**Double play probability** (runner on 1B, <2 outs, no infield hit):

| Contact tier | Base DP% | SPD modifier |
|--------------|----------|--------------|
| Hard | 70% | ±0.003 per point from SPD 50 |
| Medium | 47.5% | ±0.003 per point from SPD 50 |
| Weak | 12.5% | ±0.003 per point from SPD 50 |

`_SPD_DP_MOD = 0.003` — slow runners (SPD 30) add +0.06 to DP%; fast runners (SPD 70) subtract −0.06.

---

### Baserunning — `resolve_baserunning()`

Called on Singles and Doubles to decide if a runner attempts an extra base. Inputs used:

| Trait | Source | Role |
|-------|--------|------|
| `SPD` | Runner card | Base probability of attempting extra base |
| `BIQ` | Runner card | Decision quality: low BIQ → reckless, high BIQ → conservative situationally |
| `ARM` | Fielder (outfield) | Probability of throwing the runner out |

Outcomes:
- `HOLD` — runner holds at the conservative base
- `ADVANCE_EXTRA` — runner takes extra base safely
- `OUT_ON_BASES` — runner thrown out; outs++; inning ends immediately if outs reach 3; no run scores

---

### Pitcher state machine — `PitcherState`

```python
@dataclass
class PitcherState:
    current:          dict   # active pitcher card
    pa_faced:         int    # PAs faced by current pitcher this game
    runs_while_tired: int    # runs allowed after stamina limit
    bullpen:          list   # queue of remaining pitchers
```

**Stamina limits:**

| Role | Formula | Example (STA=75) |
|------|---------|------------------|
| SP | `round(STA × 0.40)` | 30 PA ≈ 7 innings |
| RP | `round(STA × 0.20)` | 15 PA ≈ 3 innings |

**Fatigue penalty:** once `pa_faced >= max_pa`, each additional batter reduces STF and CTL by 2 points, floored at 40.

**Hook triggers** (either condition pulls the pitcher):
- 6+ batters faced over the stamina limit (hard auto-hook)
- 2+ runs allowed while over-limit (soft manager hook)

**Bullpen cycling:** `pull_next()` pops `bullpen[0]` and resets `pa_faced`. If the bullpen is empty, the tired starter remains.

---

### Narrative layer — `narrative_dictionary.py`

`NARRATIVE_TEMPLATES` maps event keys to lists of 5+ variant strings. `print_game_log()` calls `random.choice()` to select a template and fills in `{name}`.

Key event types: `STRIKEOUT_LOOKING`, `STRIKEOUT_SWINGING`, `WALK_PATIENT`, `SINGLE_CLEAN`, `DOUBLE_GAP`, `TRIPLE_LEGGED`, `HOME_RUN`, `GROUNDOUT_DP`, `FLYOUT_WARNING`, `INFIELD_HIT`, `WEAK_ROLLER_NO_DP`, `PITCHER_TIRED`, `PITCHER_HOOK`, `RECKLESS_ADVANCE`, `THROWN_OUT`, `SMART_HOLD`, `ADVANCE_EXTRA`.

---

### `simulate_game()` signature

```python
from game_engine import simulate_game

box: BoxScore = simulate_game(
    away_team,    # dict: {"team_id", "lineup", "pitcher", "bullpen", "arm"}
    home_team,    # same format
    game_id=None, # str — deterministic seed derivation (defaults to random UUID4)
    verbose=False # bool — prints play-by-play via print_game_log()
)
```

The `game_id` string is hashed to derive per-PA seeds. Passing the same `game_id` replays the game identically.

---

## 3. Roster Manager

**File:** `roster_manager.py`

### `load_team(season_year, team_name, use_dh=True) → TeamRoster`

Queries Supabase `players` table for all rows matching `year` and `team`. Builds batter and pitcher cards with z-score traits from the `global_calibration` pipeline.

**Lineup construction:**
1. Separate batters (no `pitcher_role`) from pitchers.
2. Sort batters by `pa` descending; top 9 → starting lineup.
3. Sort starting lineup by OBP proxy descending.
4. If `use_dh=False`, 9th slot is replaced by the starting pitcher's card.

**Staff construction:**
1. Sort pitchers by `ip` descending.
2. `pitcher` = highest-IP pitcher (ace).
3. `bullpen` = remaining pitchers in IP order.
4. `pitcher_role` (`"SP"` / `"RP"`) is assigned during ingestion based on GS/G ratio.

### `TeamRoster.for_game_engine(rotation_index=0) → dict`

Returns a dict ready for `simulate_game()`. Cycles through the SP rotation via `rotation_index % len(sp_rotation)`. RP staff always stays in the bullpen queue regardless of rotation index.

```python
roster = load_team(1954, "CLE")

# Game 1 — Early Wynn starts
away = roster.for_game_engine(rotation_index=0)

# Game 2 — next SP in rotation
away = roster.for_game_engine(rotation_index=1)
```

### `load_team_from_pilot(json_path, team_id=None, use_dh=True) → TeamRoster`

Loads from a local JSON pilot file instead of Supabase. Same output shape as `load_team()`. Used for offline development and testing.

---

### Player card schema (Supabase `players` table)

**Batters:**

| Column | Maps to trait | Notes |
|--------|---------------|-------|
| `contact` | `CON` | Z-score calibrated against global pool |
| `power` | `POW` | Z-score calibrated |
| `eye` | `EYE` | Z-score calibrated |
| `speed` | `SPD` | Z-score calibrated |
| `avoid_k` | `AK` | Z-score calibrated |
| `gap` | `GAP` | Z-score calibrated |
| `biq` | `BIQ` | Z-score calibrated |
| `pa` | Lineup sort key | Real plate appearances from Lahman |
| `bats` | Handedness | `"R"`, `"L"`, `"B"` |

**Pitchers:**

| Column | Maps to trait | Notes |
|--------|---------------|-------|
| `stuff` | `STF` | Z-score calibrated |
| `control` | `CTL` | Z-score calibrated |
| `movement` | `CMD` | Z-score calibrated |
| `stamina` | `STA` | Z-score calibrated |
| `ip` | Staff sort key | Real innings pitched from Lahman |
| `throws` | Handedness | `"R"`, `"L"` |
| `pitcher_role` | Rotation slot | `"SP"` if GS/G ≥ 0.5, else `"RP"` |

---

## 4. Simulation Tooling

### `ingest_historical.py`

Ingests a single team-season from Lahman CSVs into Supabase.

```bash
python3 ingest_historical.py --season 1954 --team CLE --push
```

Arguments:
- `--season YEAR` — Lahman season year
- `--team TEAM_ID` — Lahman team ID (e.g., `CLE`, `NYA`, `BOS`)
- `--push` — write to Supabase (omit to dry-run)

After ingestion, run `global_calibration.py` to recompute z-scores across the full pool, then re-ingest all affected seasons to refresh player cards.

### `global_calibration.py`

Computes cross-era z-score reference statistics from all seasons in the Supabase pool. Must be re-run whenever new seasons are ingested.

### `simulate_series.py`

Monte Carlo simulation: NYA vs BOS (or any two teams), configurable game count, cycles SP rotation.

### `round_robin_sim.py`

Round-robin across 4+ team archetypes. Tracks RS/G, W%, AVG/OBP/SLG per team. Used for cross-era balance testing.

### `calibration_report.py`

Prints trait distributions and outcome distributions (K%, BB%, AVG, BABIP, contact %) for each team in the pool. Includes league-normalized z-scores. Read-only — no engine changes.

### `run_mid_tier_validation.py`

Round-robin simulation using 6–8 near-.500 teams across eras. Primary validation tool for offensive environment changes. Sanity checks:

| Metric | Target range |
|--------|-------------|
| League AVG | .250–.285 |
| League OBP | .310–.360 |
| League SLG | .370–.450 |
| BABIP | .285–.315 |
| K% | 14–19% |
| BB% | 7–12% |
| RS/G | 3.8–5.5 |
| No team W% | > .610 |

### `run_1954_al_replay.py`

Full 1954 American League season replay — see Section 5.

---

## 5. Historical Validation

### 1954 AL Historical Replay

**File:** `run_1954_al_replay.py`

**Scope:** All 8 real 1954 AL teams. 154-game schedule (22 games × 7 opponents). Multiple seeds. No tuning applied — uses calibrated engine as-is.

**Teams ingested:**

| Team | Ace | Rotation depth |
|------|-----|---------------|
| CLE | Early Wynn | 5 |
| NYA | Whitey Ford | 10 |
| CHA | Virgil Trucks | 4 |
| BOS | Frank Sullivan | 6 |
| DET | Steve Gromek | 6 |
| WS1 | Bob Porterfield | 5 |
| BAL | Bob Turley | 6 |
| PHA | Arnie Portocarrero | 7 |

All ingested via `ingest_historical.py --season 1954 --team <ID> --push`.

**4-seed replay results (4 × 154 games = 2,464 total games):**

| Team | Sim W | Real W | Δ PCT | RS/G | RA/G |
|------|-------|--------|-------|------|------|
| NYA | 106 | 103 | +0.016 | 4.15 | 2.46 |
| CHA | 96 | 94 | +0.015 | 3.44 | 2.39 |
| CLE | 89 | 111 | **−0.143** ⚠ | 3.47 | 2.89 |
| BOS | 79 | 69 | +0.063 | 3.62 | 3.46 |
| BAL | 74 | 54 | **+0.128** ⚠ | 2.82 | 3.08 |
| DET | 66 | 68 | −0.010 | 2.82 | 3.49 |
| WS1 | 65 | 66 | −0.005 | 2.95 | 3.45 |
| PHA | 41 | 51 | −0.065 | 2.14 | 4.19 |

**League environment:**

| Metric | Sim | Historical 1954 | Flag |
|--------|-----|----------------|------|
| AVG | .237 | .261 | ⚠ suppressed |
| OBP | .314 | .337 | ✓ |
| SLG | .349 | .387 | ⚠ suppressed |
| BABIP | .274 | .292 | ⚠ below floor |
| ERA | 3.40 | 3.72 | ✓ |
| K% | 17.9% | 9.8% | ⚠ (expected — cross-era) |
| BB% | 9.0% | 9.5% | ✓ |
| HR/G/team | 1.65 | 0.77 | ⚠ elevated |

**Structural authenticity:**

- ✓ NYA/CHA correctly positioned (within ±3 wins of historical)
- ✓ PHA dead last — consistent with historical 51-103
- ✓ Ted Williams leads AVG (.309), HR (49.8/seed), OPS (1.010)
- ✓ Bobby Avila (CLE) in top-5 batting average
- ✓ CLE pitchers (Mike Garcia 23 W, Early Wynn 19 W) in Wins top 3
- ✓ ERA structurally plausible (3.40 vs 3.72 historical)
- ✓ Yogi Berra, Mantle, Miñoso, Nellie Fox surface correctly in leaderboards

**Known cross-era seams (by design, not calibration defects):**

1. **K% elevation (+8pp):** The cross-era z-score calibration normalizes AK/STF against all of baseball history (1871–present), not within-era. 1954 hitters receive average bat-to-ball ratings relative to modern standards. K% runs ~18% vs historical ~10% — expected behavior. This cascades into AVG/SLG suppression (~10pp each).

2. **CLE underperformance (−22 wins):** Cleveland's 1954 dominance was partly driven by historically elite team-wide offense (Doby 32 HR/126 RBI, Rosen, Avila). The cross-era calibration assigns these hitters average-to-above-average CON/POW z-scores — their era-specific greatness is partially flattened. Their pitching calibrates correctly (Garcia/Wynn are elite), but the offense doesn't replicate the historical run environment.

3. **BAL overperformance (+20 wins):** Bob Turley's high STF z-score (a legitimate K-rate outlier for his era) generates ~380 K/seed in the cross-era environment, pushing him into historically-unrealistic dominance. His traits are correctly calibrated relative to history; the issue is that cross-era opponents have much lower AK than 1954 reality.

4. **HR rate doubled (1.65 vs 0.77 historical):** The engine's HR gate is calibrated for a blended cross-era pool. 1954 was a relatively low-HR era. HR/G will always run elevated vs real 1954 figures in this configuration.

---

## 6. Calibration Log

All changes are listed in chronological order. The PA Engine is frozen at the state described here.

| # | Date | Constant | File | Before → After | Effect |
|---|------|----------|------|----------------|--------|
| 1 | Initial | `_HARD_Q_MID` | `pa_engine.py` | `44.0 → 28.0` | Hard contact rate for avg hitter: 10% → 26%; routes HRs through Hard contact path |
| 2 | Initial | `_HARD_HR_FLOOR_BONUS` | `pa_engine.py` | `(new) 0.055` | HR/Hard BIP for avg hitter: .060 → .116 |
| 3 | May 2026 | `league_avg_k_pct` | `sim_constants.json` | `0.13 → 0.165` | Baseline K rate raised; K% 11.5% → ~16%; primary fix for K% inflation |
| 4 | May 2026 | `_HARD_Q_MID` | `pa_engine.py` | `28.0 → 30.0` | Hard contact frequency reduced ~8pp for mid-tier hitters; elite hitters (Q≈40+) near ceiling still |
| 5 | May 2026 | `_NON_HR_BASE["Hard"]["Double"]` | `pa_engine.py` | `0.180 → 0.130` | Hard 2B rate: 54% of hits → 38% of hits; BABIP-neutral SLG reduction |
| 6 | May 2026 | `_NON_HR_BASE["Hard"]["Single"]` | `pa_engine.py` | `0.140 → 0.190` | Compensating 1B increase to preserve BABIP on Hard contact |
| 7 | May 2026 | `_NON_HR_BASE["Medium"]["Double"]` | `pa_engine.py` | `0.060 → 0.040` | Medium 2B rate: 24% of hits → 13% of hits; BABIP-neutral SLG reduction |
| 8 | May 2026 | `_NON_HR_BASE["Medium"]["Single"]` | `pa_engine.py` | `0.185 → 0.205` | Compensating 1B increase to preserve BABIP on Medium contact |
| 9 | May 2026 | `_HARD_HR_FLOOR_BONUS` | `pa_engine.py` | `0.055 → 0.040` | HR/G reduced; elite power hitters (POW=90+) in 15–20% HR/Hard BIP; avg power ~7–9% |
| 10 | May 2026 | `_HR_POW_SCALE` | `pa_engine.py` | `0.13 → 0.12` | Main HR coefficient trimmed proportionally with Hard% reduction |
| 11 | May 2026 | `K_STF_CON_COEFF` | `pa_engine.py` | `0.20 → 0.09` | Era equity fix: reduces artificial K% suppression for high-CON old-era hitters; W% spread compressed from 0.198 → 0.125 across mid-tier pool |

**Frozen calibration targets (mid-tier round-robin validation, 1000+ games):**

| Metric | Value |
|--------|-------|
| League AVG | .260 |
| League OBP | .320 |
| League SLG | .400 |
| K% | 16.2% |
| BB% | 9.1% |
| BABIP | .289 |
| RS/G | 3.55–3.87 |
| W% spread (8-team pool) | 0.125 |

---

*Chin Music Engine — internal technical reference. Not for distribution.*
