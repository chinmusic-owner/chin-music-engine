### PRD 1: Core Simulation Engine ("The Brain")
#### Plate Appearance Engine v1.1

**Version:** 1.1 
**Status:** Draft  
**Owner:** TBD  
**Depends On:** None (foundational)  
**Feeds Into:** PRD 2 (Player Ingestion), PRD 5 (Game Loop), PRD 6 (Explanation Layer)

#### 1. Purpose

Define and build the Plate Appearance (PA) Engine — the probabilistic core that resolves every batter-pitcher matchup in Chin Music. This is "The Brain." Every other system in the platform depends on this engine being correct, explainable, and deterministic.

The PA Engine does **not** care where player data comes from. It receives a normalized trait vector and resolves an outcome. That separation is intentional.

#### 2. Scope

**In Scope (v1):**
- 3-Stage PA resolution: Duel → Contact → Defense
- Trait input model (CON, GAP, POW, EYE, AK, STF, CTL, CMD, STA, RNG, HND, ARM)
- Outcome probability mapping (K, BB, HBP, BIP → Single, Double, Triple, HR, Out, Error)
- Platoon/handedness modifiers
- Variance injection (per-stage, not post-hoc)
- Deterministic seeding (sim_seed + game_seed)
- PA Audit Log ("Receipt") — full explainability metadata per PA
- Synthetic test player generator (for validation without real data)
- Simulation constants config file

**Out of Scope (v1):**
- Real historical player data ingestion (PRD 2)
- Fatigue/trait decay (PRD 5)
- Park factors (PRD 5)
- Era calibration (PRD 5)
- Narrative/explanation layer (PRD 6)
- Lineup sequencing, bullpen logic (PRD 5)

#### 3. The Data Question (Answered)

The PA Engine **does** require a data layer — but not historical stats. It requires:

**3A. Trait Input Schema (Required)**
The engine needs a defined, stable input contract. Every player passed to the engine must conform to this schema:

```json
{
  "player_id": "string",
  "player_type": "batter | pitcher | fielder",
  "handedness": "L | R | S",
  "traits": {
    "CON": 0-100,
    "GAP": 0-100,
    "POW": 0-100,
    "EYE": 0-100,
    "AK":  0-100,
    "BNT": 0-100,
    "STF": 0-100,
    "CTL": 0-100,
    "CMD": 0-100,
    "STA": 0-100,
    "RNG": 0-100,
    "HND": 0-100,
    "ARM": 0-100
  }
}
```

**3B. Simulation Constants File (Required)**
A config file that defines the "physics" of the simulation world. This is not player data — it is the gravity of the engine:

```json
{
  "replacement_level_skillscore": 50,
  "league_avg_k_pct": 0.215,
  "league_avg_bb_pct": 0.085,
  "league_avg_hr_per_fb": 0.12,
  "league_avg_babip": 0.300,
  "league_avg_runs_per_game": 4.5,
  "duel_weights": {
    "w1_con_stf": 0.55,
    "w2_eye_ctl": 0.45
  },
  "contact_quality_weights": {
    "a_pow": 0.40,
    "b_con": 0.35,
    "d_gap": 0.25
  },
  "ace_tax_threshold": 150,
  "ace_tax_alpha": 0.3,
  "ace_tax_q": 1.8,
  "era_calibration_tolerance": 0.05
}
```

**3C. Synthetic Test Player Set (Required for Validation)**
Before real data exists, you need mock players to stress-test the engine. Deep Agent is ideal for generating this. Target archetypes:

| Player ID | Type | Profile |
|---|---|---|
| `test_apex_hitter` | Batter | CON 95, POW 95, EYE 90 |
| `test_avg_hitter` | Batter | CON 70, POW 65, EYE 65 |
| `test_replacement_hitter` | Batter | CON 50, POW 45, EYE 50 |
| `test_ace_pitcher` | Pitcher | STF 95, CTL 88, CMD 85 |
| `test_avg_pitcher` | Pitcher | STF 72, CTL 70, CMD 68 |
| `test_replacement_pitcher` | Pitcher | STF 55, CTL 55, CMD 52 |
| `test_elite_defense` | Fielder | RNG 90, HND 88, ARM 85 |
| `test_avg_defense` | Fielder | RNG 65, HND 65, ARM 65 |

**3D. Trait Provenance + Component Metadata (Optional; required for CM Live later)**
The PA Engine remains agnostic to how traits are built. However, to support future Live modes and manager-facing transparency, the engine must accept (and log) optional metadata describing where the trait values came from.
Requirement:
• The existing required trait schema remains unchanged.
• Add an optional trait_metadata object to the runtime input contract.
• If trait_metadata is present, the PA Audit Log (“Receipt”) must include it verbatim.
Proposed optional input extension (do not break existing callers):
{
  "player_id": "string",
  "player_type": "batter | pitcher | fielder",
  "handedness": "L | R | S",
  "traits": { "...": "0-100 trait map" },
  "trait_metadata": {
    "mode": "historical | live",
    "as_of_date": "YYYY-MM-DD (optional)",
    "components": [
      {
        "name": "yesterday",
        "weight": "0.0-1.0",
        "notes": "optional free text"
      },
      {
        "name": "last_7_14_days",
        "weight": "0.0-1.0"
      },
      {
        "name": "season_2026",
        "weight": "0.0-1.0"
      },
      {
        "name": "prior_2025",
        "weight": "0.0-1.0"
      }
    ],
    "reliability": {
      "CON": "0.0-1.0",
      "POW": "0.0-1.0",
      "EYE": "0.0-1.0",
      "AK":  "0.0-1.0",
      "STF": "0.0-1.0",
      "CTL": "0.0-1.0",
      "CMD": "0.0-1.0",
      "STA": "0.0-1.0"
    }
  }
}

Notes:
• trait_metadata is not used to resolve outcomes in v1 Historical.
• Its purpose is future-proofing + auditability + explanation.

#### 4. 3-Stage PA Resolution

##### Stage 1 — Duel (K / BB / HBP / BIP)

**Inputs:** Pitcher (STF, CTL) vs Batter (CON, EYE, AK) + handedness modifier

**Duel Score:**

```text
D = w1 * (CON - STF) + w2 * (EYE - CTL)
P(batter_advantage) = 1 / (1 + e^(-D))
```

**Outcome Mapping:**
- `K%` — driven by `STF - CON`, reduced by `AK`
- `BB%` — driven by `EYE - CTL`; POW has **zero** influence here
- `HBP%` — low-rate stochastic, slightly influenced by low CTL
- `BIP% = 1 - K% - BB% - HBP%`

**Constraints:**
- All probabilities sum to 1.0
- Hard floors/ceilings: no 0% or 100% outcomes
- Diminishing returns at trait extremes (logistic scaling)

##### Stage 2 — Contact Quality (triggered on BIP)

**Inputs:** Batter (POW, CON, GAP) vs Pitcher (STF, CMD)

**Contact Score:**

```text
Q = a*POW + b*CON + d*GAP - c*STF + CMD_variance_term
```

**Outputs:** Weak / Medium / Hard contact + spray distribution

**Rules:**
- POW is gated by CON — high POW with low CON = inconsistent hard contact
- CMD tightens distribution (reduces mistakes); low CMD = higher variance
- POW cannot reduce K% or increase BB% (those are Duel-only)

##### Stage 3 — Defense (Ball vs Fielders)

**Inputs:** Batted-ball vector (quality + spray) + defender traits (RNG, HND, ARM)

**Sequence:**
1. RNG check — does the defender reach it?
2. HND check — is the play converted cleanly?
3. ARM check — baserunner advancement / out probability

**Outcomes:** Out / Single / Double / Triple / Error

**4A. External Context Inputs (Optional; reserved for Live and advanced modes)**
Future modes (Live, advanced manager options) require the PA Engine to log the context assumptions under which a PA was resolved (e.g., whether a player was hot/cold, platoon context, or other non-trait state that influenced trait construction upstream).
Requirement:
• Accept an optional pa_context object at runtime.
• Log pa_context into the PA Receipt if provided.
• Do not require it for Historical v1.
Proposed optional pa_context:
{
  "game_date": "YYYY-MM-DD (optional)",
  "batter_usage": { "started": "bool (optional)", "lineup_slot": "1-9 (optional)" },
  "pitcher_usage": { "role": "SP|RP (optional)" }
}

#### 5. Variance Model

Variance is injected **inside** each stage, not as a single post-hoc modifier. This produces:
- Natural BABIP variance
- Bloop hits and unlucky lineouts
- Sequencing effects that feel like real baseball

Do **not** add a single "luck multiplier" at the end. That breaks explainability.

#### 6. Platoon / Handedness

Apply a modifier to the Duel Score based on pitcher/batter handedness matchup:

```text
platoon_modifier = config["platoon_advantage_delta"]  # e.g., +3 to +5 duel points
```

- Same-hand matchup: pitcher advantage
- Opposite-hand matchup: batter advantage
- Switch hitters: always use favorable side

#### 7. Deterministic Seeding (Non-Negotiable)

Every simulation run must be reproducible:

```python
sim_seed = generate_sim_seed()          # persisted per sim window
game_seed = hash(sim_seed, game_id)     # derived per game
pa_seed   = hash(game_seed, pa_index)   # derived per PA
```

- Re-running with the same `sim_seed` must produce **identical** outcomes
- `sim_seed` is stored in the replay header and all audit logs
- Required for: debugging, dispute resolution, QA, user trust

#### 8. PA Audit Log ("Receipt") — (Updated for Live transparency passthrough)
Every resolved PA must generate an immutable log entry. The receipt has two jobs:
1. Debuggability: engineering can replay and validate any outcome deterministically.
2. Explainability: the Narrative/Explanation layer can later render why a player’s CM outcomes differ from MLB outcomes, using the same underlying inputs (without inventing new logic).
Important: the PA Engine does not compute Live weights. It only logs any provenance/weight metadata passed in by the upstream Player State / Card Builder.
```json
{
  "sim_id": "...",
  "game_id": "...",
  "pa_index": 42,
  "sim_seed": "...",
  "game_seed": "...",

  "engine_metadata": {
    "engine_version": "1.1.0",
    "constants_version": "2026.01",
    "trait_schema_version": "1.0",
    "deterministic_seed": "pa_seed_value"
  },

  "batter_id": "...",
  "pitcher_id": "...",

  "inning": 7,
  "outs": 1,
  "runners_state": "1B_3B",

  "pa_context": {
    "game_date": "YYYY-MM-DD",
    "batter_usage": { "started": true, "lineup_slot": 3 },
    "pitcher_usage": { "role": "SP" }
  },

  "pre_pa_traits": {
    "batter": { "...": "final 0-100 traits used by engine" },
    "pitcher": { "...": "final 0-100 traits used by engine" }
  },

  "pre_pa_trait_metadata": {
    "batter": {
      "mode": "historical | live",
      "as_of_date": "YYYY-MM-DD",
      "components": [
        { "name": "yesterday", "weight": 0.05 },
        { "name": "last_7_14_days", "weight": 0.15 },
        { "name": "season_2026", "weight": 0.60 },
        { "name": "prior_2025", "weight": 0.20 }
      ],
      "reliability": {
        "CON": 0.82,
        "GAP": 0.70,
        "POW": 0.78,
        "EYE": 0.88,
        "AK": 0.90
      }
    },
    "pitcher": {
      "mode": "historical | live",
      "as_of_date": "YYYY-MM-DD",
      "components": [
        { "name": "yesterday", "weight": 0.04 },
        { "name": "last_7_14_days", "weight": 0.16 },
        { "name": "season_2026", "weight": 0.62 },
        { "name": "prior_2025", "weight": 0.18 }
      ],
      "reliability": {
        "STF": 0.80,
        "CTL": 0.84,
        "CMD": 0.72,
        "STA": 0.90
      }
    }
  },

  "duel_score": -0.23,
  "duel_probabilities": { "K": 0.24, "BB": 0.09, "HBP": 0.01, "BIP": 0.66 },
  "initial_outcome": "BIP",

  "contact_quality": "Hard",
  "spray_vector": "Pull",

  "defense_resolution": {
    "RNG_check": "reached",
    "HND_check": "clean",
    "result": "Out"
  },

  "final_outcome": "Out",

  "explanation_tags": {
    "duel": "K: STF(92) > CON(78); AK(70) reduced K by 4%",
    "contact": "Hard contact: POW(90) + CON(85)",
    "defense": "Out: RNG(88) + HND(91)",
    "live_inputs": "Trait blend (batter): season_2026 0.60, prior_2025 0.20, last_7_14_days 0.15, yesterday 0.05"
  }
}
```

Receipt rules:
• pre_pa_traits must always reflect the final effective trait values used to resolve the PA.
• pre_pa_trait_metadata is optional passthrough:
  • If absent (historical v1), the receipt remains valid.
  • If present (live), it must be persisted unchanged for transparency.
• pa_context is optional passthrough (useful for explaining usage/role effects in Live).
• engine_metadata is required for every PA and must reflect the exact versions used at runtime.


#### 9. Validation Criteria

Before this PRD is considered complete, the engine must pass:

| Test | Requirement |
|---|---|
| Probability sum | K% + BB% + HBP% + BIP% = 1.000 for every matchup |
| Apex vs Replacement | Apex hitter vs replacement pitcher produces realistic K/BB/HR rates |
| Replacement vs Apex | Replacement hitter vs apex pitcher produces realistic suppression |
| Determinism | Same seed = identical PA sequence across 10,000 runs |
| BABIP range | League-sim BABIP across 10k PAs falls within .280–.320 |
| K% range | League-sim K% falls within .190–.240 |
| BB% range | League-sim BB% falls within .075–.100 |
| Runs/G range | Full game sim produces 3.8–5.2 R/G across 1,000 games |
| Explainability | Every PA receipt contains all required fields |
| No black boxes | No outcome is produced without a traceable explanation tag |

#### 10. Acceptance Criteria (Definition of Done)

- [ ] Trait input schema defined and documented
- [ ] Simulation constants config file created and version-controlled
- [ ] Synthetic test player set generated (8 archetypes minimum)
- [ ] Stage 1 (Duel) implemented and unit tested
- [ ] Stage 2 (Contact Quality) implemented and unit tested
- [ ] Stage 3 (Defense) implemented and unit tested
- [ ] Platoon/handedness modifier implemented
- [ ] Variance injection per-stage (not post-hoc)
- [ ] Deterministic seeding implemented and verified
- [ ] PA Audit Receipt generated for every PA
- [ ] All validation criteria above pass
- [ ] Engine runs 10,000 simulated PAs in < 5 seconds (performance baseline)
- [ ] No dependency on real historical data (PRD 2 is separate)

#### 11. Open Questions

1. **Platoon delta magnitude** — what is the right `+/- duel point` adjustment for handedness? Needs calibration against historical platoon splits.
2. **CMD variance term** — exact formula for how CMD affects contact distribution spread needs to be defined before Stage 2 implementation.
3. **Spray vector model** — pull/center/oppo distribution: does this need a full probability matrix per contact quality tier, or a simpler lookup table for v1?
4. **HBP rate** — what is the right baseline HBP% and how much does low CTL move it?
5. **Fielder assignment** — for Stage 3, how does the engine know which fielder to route the ball to? Does v1 use a simplified positional lookup or a full spray-angle model?

#### 99. The "Live-Proofing" Safety Contract
1. Trait Purity: The PA Engine (PRD 01) must never calculate a player's skill. It only consumes final 0-100 traits.
2. Metadata Passthrough: Any extra "Live" data (like 'Battery Reliability' or 'Clutch Weights') must be treated as "Passive Metadata." The engine should save it to the Receipt but never use it to change the Home Run vs. Strikeout math in this version.
3. Separation of Concerns: The "Brain" (PRD 01) resolves the play. The "Ingestion" (PRD 02) creates the player. No logic from one shall ever leak into the other.