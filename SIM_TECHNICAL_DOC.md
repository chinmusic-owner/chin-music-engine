# Chin Music Engine — PA Engine Technical Reference

**Audience:** Engineers building the Inning Engine, Roster Loader, or any module that calls `resolve_pa_seeded`.

**Engine status:** FROZEN as of calibration session May 2026. Do not modify `pa_engine.py` or `pa_wrapper.py` calibration constants without a new calibration session.

---

## 1. INPUTS

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
| `"CON"` | Contact | Stage 1 (K% suppression via `CON - STF` edge) |
| `"EYE"` | Plate discipline | Stage 1 (BB% via `EYE - CTL` edge) |
| `"AK"` | Anti-K / bat-to-ball | Stage 1 (K% adjustment; high AK suppresses Ks, low AK inflates them) |
| `"POW"` | Raw power | Stage 2.5 (HR gate — `POW^2.5` term; raw POW, not CON-gated) |
| `"GAP"` | Gap power | Stage 2 (contact score contribution) + Stage 2.5 (Double probability shift) |

> `BNT` (Bunt) exists on `PlayerCard` but is **not read by `pa_engine`**. The Inning Engine is responsible for modeling bunt decisions.

---

### `pitcher` — dict

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `"traits"` | `dict` | **Yes** | Flat dict of pitching trait keys (see below) |
| `"throws"` | `str` | Recommended | Handedness: `"R"` or `"L"`. Defaults to `"R"` if absent. |

**Pitcher trait keys** (all integers, scale 1–99):

| Key | Trait | Engine role |
|-----|-------|-------------|
| `"STF"` | Stuff / velocity | Stage 1 (K% via `STF - CON` edge and raw STF dominance term) + Stage 2 (contact score suppressor) |
| `"CTL"` | Control / command | Stage 1 (BB% via `CTL - EYE` edge; HBP base when CTL is low) |
| `"CMD"` | Command precision | Stage 2 (variance on contact quality — low CMD widens distribution) + Stage 2.5 (BABIP suppressor; HR mistake-factor) |
| `"STA"` | Stamina | **Not read by `pa_engine`** — reserved for the Inning Engine's fatigue model |

> `STA` is deliberately excluded from all `pa_engine` calculations. The Inning Engine owns fatigue and decline curves; it adjusts or replaces the pitcher card passed to `resolve_pa_seeded` based on pitch count/inning.

---

### `context` — dict (optional)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `"fielder"` | `dict` | `{"RNG": 50, "HND": 50, "ARM": 50}` | Fielder traits for Stage 3 (see Fielder section below) |
| `"constants"` | `dict` | Loaded from `sim_constants.json` | Full constants override — for testing only; never override in production |

**Fielder trait keys:**

| Key | Trait | Engine role |
|-----|-------|-------------|
| `"RNG"` | Range | Stage 3: probability of reaching ball in play (affects Out/Single conversion) |
| `"HND"` | Hands | Stage 3: error probability on any reached ball |
| `"ARM"` | Arm strength | Stage 3: probability of holding a runner (Triple→Double, Double→Single) |

To build a fielder dict from a Supabase player row:
```python
from pa_wrapper import fielder_from_row
fielder = fielder_from_row(supabase_player_row)  # falls back to 50 with warning if null
```

---

### `seed` — int

Any non-negative integer. The same `(batter, pitcher, context, seed)` tuple **always** produces the exact same outcome. The Inning Engine is responsible for generating unique seeds per PA.

Recommended seeding pattern (mirrors the existing valuation engine):
```python
import hashlib

def derive_seed(game_id: str, pa_index: int) -> int:
    raw = hashlib.sha256(f"{game_id}:{pa_index}".encode()).hexdigest()
    return int(raw, 16) % (2 ** 32)
```

---

## 2. OUTPUTS

`resolve_pa_seeded` returns a single `dict` with the following top-level keys:

### Top-level result dict

| Key | Type | Always present | Description |
|-----|------|----------------|-------------|
| `"outcome"` | `str` | **Yes** | **The only field the Inning Engine needs for game state.** See outcome values below. |
| `"seed"` | `int` | Yes | The seed that was used (echo of input) |
| `"duel"` | `dict` | Yes | Stage 1 internals (see Stage Metadata below) |
| `"contact"` | `dict \| None` | Yes | Stage 2 internals. `None` if Stage 1 resolved to K/BB/HBP. |
| `"bip_map"` | `dict \| None` | Yes | Stage 2.5 internals. `None` if Stage 1 resolved to K/BB/HBP. |
| `"defense"` | `dict \| None` | Yes | Stage 3 internals. `None` if Stage 1 resolved to K/BB/HBP. |

### `"outcome"` — the terminal output

All possible values:

| Value | Category | Base-runner effect |
|-------|----------|--------------------|
| `"K"` | Out | Strikeout; no baserunner advance unless passed ball (Inning Engine concern) |
| `"BB"` | On base | Batter to first; force advances apply |
| `"HBP"` | On base | Batter to first; force advances apply |
| `"HR"` | On base | Batter + all runners score; bases clear |
| `"Single"` | On base | Batter to first; Inning Engine advances other runners by its own baserunning rules |
| `"Double"` | On base | Batter to second; Inning Engine advances other runners |
| `"Triple"` | On base | Batter to third; Inning Engine advances other runners |
| `"Out"` | Out | Generic out (groundout/flyout/lineout); no baserunner change beyond force plays |
| `"Error"` | On base | Fielding error; batter reaches first — does **not** count as a hit for batting average purposes |

> **`pa_engine` has no concept of sacrifice fly, sacrifice bunt, fielder's choice, double play, or passed ball.** All of these are Inning Engine responsibilities, applied after `resolve_pa_seeded` returns.

---

### Stage metadata (available for diagnostics, logging, or future sim features)

**`result["duel"]`** (Stage 1):

| Key | Type | Description |
|-----|------|-------------|
| `"outcome"` | `str` | Internal outcome before BIP resolution: `"K"`, `"BB"`, `"HBP"`, or `"BIP"` |
| `"duel_score"` | `float` | Raw duel score `D_raw` (batter edge over pitcher, unbounded) |
| `"p_batter_advantage"` | `float` | Logistic-transformed batter advantage probability [0,1] |
| `"probabilities"` | `dict` | `{"K": float, "BB": float, "HBP": float, "BIP": float}` — the four Stage 1 probabilities |

**`result["contact"]`** (Stage 2, `None` if not BIP):

| Key | Type | Description |
|-----|------|-------------|
| `"contact_score"` | `float` | Normalized quality score 0–100; feeds the HR gate in Stage 2.5 |
| `"contact_quality"` | `str` | `"Weak"`, `"Medium"`, or `"Hard"` — the sampled tier |
| `"spray_vector"` | `str` | `"Pull"`, `"Center"`, or `"Oppo"` |
| `"effective_pow"` | `float` | Raw POW after CON-gate (`POW × logistic((CON−45)/15)`) |
| `"cmd_noise"` | `float` | Gaussian noise term from pitcher CMD (wider = wilder contact distribution) |

**`result["bip_map"]`** (Stage 2.5, `None` if not BIP):

| Key | Type | Description |
|-----|------|-------------|
| `"bip_outcome"` | `str` | Pre-defense outcome: `"HR"`, `"Single"`, `"Double"`, `"Triple"`, `"Error"`, or `"Out"` |
| `"bip_probabilities"` | `dict` | Full multinomial table including HR gate probability |
| `"hr_driver"` | `float` | `POW^2.5` contribution to HR gate (for diagnostics) |

**`result["defense"]`** (Stage 3, `None` if not BIP):

| Key | Type | Description |
|-----|------|-------------|
| `"final_outcome"` | `str` | Post-defense outcome — this is what `result["outcome"]` at the top level reflects |
| `"defense_resolution"` | `dict` | `{"RNG_check": str, "HND_check": str, "ARM_check": str, "result": str}` — each check is `"reached"`, `"not_reached"`, `"held"`, `"not_held"`, `"error"`, `"clean"`, or `"skipped"` |

---

## 3. PURE FUNCTIONS — STATELESSNESS GUARANTEE

`resolve_pa_seeded` and all `pa_engine` stage functions are **completely stateless**. Specifically:

- **No global mutable state.** `sim_constants.json` is read once at import time into a module-level constant (`_CONSTANTS`) and never modified.
- **No inning or game context.** The engine has no knowledge of the current inning, score, outs, base-runner configuration, pitch count, or fatigue. It resolves one matchup and returns.
- **No side effects.** Nothing is written to disk, database, or any external system.
- **Full determinism.** Given identical `(batter, pitcher, context, seed)` inputs, the output is bit-for-bit identical on every call, on every machine, across Python versions ≥ 3.10.
- **Thread-safe.** Because there is no shared mutable state, multiple Inning Engine workers can call `resolve_pa_seeded` in parallel without locks.

The only randomness lives inside the `random.Random(seed)` instance created fresh on each call and discarded on return.

---

## 4. INTEGRATION GUIDELINES

### For the Inning Engine

**The Inning Engine owns everything the PA engine does not:**

| Responsibility | Owner |
|----------------|-------|
| Outs counter (0–2) | Inning Engine |
| Base-runner state (1B/2B/3B occupied) | Inning Engine |
| Advancing runners on hits | Inning Engine (use `outcome` string to decide movement) |
| Force plays and double plays | Inning Engine |
| Sacrifice fly / sacrifice bunt decisions | Inning Engine |
| Stolen base attempts | Inning Engine |
| Passed balls / wild pitches | Inning Engine |
| Pitcher fatigue / STA decay | Inning Engine (adjust or swap pitcher card; `pa_engine` never reads `STA`) |
| Pinch hitter / pinch runner decisions | Inning Engine |
| Walk-off detection | Inning Engine |

**Correct call pattern:**

```python
# Inning Engine loop (pseudocode)
for each PA in inning:
    pitcher_card = current_pitcher_card(fatigue_model)  # Inning Engine adjusts traits
    fielder      = fielder_from_row(active_fielder_row)  # or default 50/50/50
    seed         = derive_seed(game_id, pa_global_index)

    result = resolve_pa_seeded(batter_card, pitcher_card, context={"fielder": fielder}, seed=seed)

    outcome = result["outcome"]
    update_inning_state(outcome, bases, outs)  # Inning Engine's own logic
```

**Do not:**
- Pass inning number, score, or base state into `resolve_pa_seeded` — it ignores them.
- Cache or reuse the `result` dict across innings — always re-derive seeds per PA.
- Modify any `pa_engine.py` constant without a calibration session.

---

### For the Roster Loader

Player cards must conform to the shape `resolve_pa_seeded` expects:

```python
# Minimum valid batter card
batter = {
    "player_id": "ruthba01",
    "name":      "Babe Ruth",
    "bats":      "L",
    "throws":    "R",
    "traits": {
        "CON": 80,
        "EYE": 75,
        "AK":  70,
        "POW": 99,
        "GAP": 85,
    },
}

# Minimum valid pitcher card
pitcher = {
    "player_id": "martipe02",
    "name":      "Pedro Martinez",
    "bats":      "R",
    "throws":    "R",
    "traits": {
        "STF": 99,
        "CTL": 90,
        "CMD": 93,
        "STA": 80,   # stored on card; Inning Engine uses this, pa_engine ignores it
    },
}
```

Extra keys (e.g., `"season"`, `"primary_role"`, `"pitcher_role"`) are silently ignored by `resolve_pa_seeded`.

---

## 5. CALIBRATION LOG (frozen state)

| Step | Constant | File | Change | Effect |
|------|----------|------|--------|--------|
| 1 | `_HARD_Q_MID` | `pa_engine.py` | `44.0 → 28.0` | Hard contact rate for avg hitter: 10% → 26%; routes HRs through Hard contact path |
| 2 | `_HARD_HR_FLOOR_BONUS` | `pa_engine.py` | `(new) 0.055` | HR/Hard BIP for avg hitter: .060 → .116; avg HR/PA: .016 → .028 |

**Frozen calibration targets (Avg Regular vs Avg Pitcher, 50,000 PAs):**

| Metric | Value |
|--------|-------|
| K% | .130 |
| BB% | .086 |
| HR/PA | .028 |
| Hard contact rate | 26.2% |
| % of HRs from Hard contact | 84% |
| BABIP | .315 |
