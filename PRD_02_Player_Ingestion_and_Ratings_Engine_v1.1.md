### PRD 2: Player Ingestion + Ratings Engine
#### Historical Player Card Builder v1.1 (Raw Stats → Normalized Rates → 0–100 Traits)

**Version:** 1.1
**Status:** Draft  
**Owner:** TBD  
**Depends On:** PRD 1 (Plate Appearance Engine v1) *only for integration testing* (the card builder can be built in parallel)  
**Feeds Into:** PRD 1 (runtime trait inputs), PRD 3 (SkillScore/SGP salary model), Pool Construction (Franchise Apex + Clash of Titans), Player Value Guide

#### 1. Purpose

Build the ingestion and ratings pipeline that turns historical baseball data into **Chin Music Player Cards**.

A **Player Card** is the precomputed artifact used at runtime by the simulation engine.

Per the Game Bible’s canonical execution order, this PRD covers:

- **Step 1 — Raw Data Input**
- **Step 2 — Era Normalization**
- **Step 3 — Trait Conversion (the Player Card)**

This system is responsible for the platform’s credibility. If the traits are wrong or inconsistent across eras, the sim will feel fake no matter how good the PA engine is.

#### 2. Product Outcome (what this enables)

- **Brushback Preview Trial / Franchise Apex:** generate complete, “as-played” team-season rosters with player traits.
- **Clash of Titans:** generate high-quality single-season cards across all eras that are comparable and draftable.
- **Explanation Layer:** preserve traceability from outcome → trait → normalized stat → original stat.

#### 3. Scope

**In Scope (v1):**
- Data ingestion into a **canonical internal schema** (batting, pitching, fielding, baserunning, team context)
- Validation + cleaning (missing values, duplicates, bad seasons)
- Era/league normalization to a neutral baseline
- Trait conversion into 0–100 traits:
  - Hitters: `CON`, `GAP`, `POW`, `EYE`, `AK`, `BNT`
  - Pitchers: `STF`, `CTL`, `CMD`, `STA`
  - Defense: `RNG`, `HND`, `ARM`, optional `CTCH` (catcher)
- Role + position tagging (SP/RP, primary positions, eligibility)
- Sample-size handling (regression to mean, reliability scores)
- Export of Player Cards as JSON (or a DB table) usable by PRD 1
- Build a **small pilot library** for testing (enough teams + players to run real games)

**Out of Scope (v1):**
- Full economic valuation (SkillScore → SGP → salary curve) (PRD 3)
- Park factors (runtime or card-level) (PRD 5)
- Fatigue and in-game trait decay (PRD 5)
- Deep pitch-level modeling (statcast pitch shapes) (future)
- Minor leagues, Negro Leagues, international leagues (future)

#### 4. Guiding Principles (non-negotiables)

1. **Separation of concerns**
   - The sim engine consumes traits; it does not compute traits.
2. **Cross-era comparability**
   - 1910 contact skill and 1998 contact skill must mean the same thing in the neutral environment.
3. **Traceability**
   - Every trait on a card must be explainable as a transformation of specific normalized inputs.
4. **Deterministic builds**
   - Same source data + same build version = identical card outputs.
5. **No hidden buffs**
   - Era identity should emerge from player pools and the engine, not from arbitrary era bonuses.

**4A. Live-Compatibility Principle (Non-negotiable architecture)**
The historical ratings engine must be implemented as reusable transformation functions that can operate over different time windows in the future (e.g., yesterday, last 7–14 days, season-to-date, prior year), even though v1 uses full historical seasons.
Requirement:
• The pipeline must separate:
1) normalization logic
2) trait conversion logic
3) reliability (sample-size) logic
• These functions must be callable with different “windowed inputs” later without rewriting the PA engine or redefining traits.

#### 5. Inputs

This PRD intentionally does **not** hard-require a specific public dataset.

**Required input concept:** a structured dataset containing player-season batting, pitching, and (at least basic) fielding, plus league-year context (run environment).

**5A. Raw stat “minimum viable” fields (canonical internal schema)**

You can source these from any dataset, but the card builder will convert into this internal schema.

**Batting (player-season):**
- Identifiers: `player_id`, `season`, optional `team_id`
- Playing time: `PA` (or fields to compute it), `AB`
- Outcomes: `H`, `2B`, `3B`, `HR`, `BB`, `SO`, `HBP`
- Useful extras (if available): `SF`, `SH` (for bunting), `IBB`
- Baserunning (if available): `SB`, `CS`

**Pitching (player-season):**
- Identifiers: `player_id`, `season`, optional `team_id`
- Playing time: `BF` (batters faced) or fields to compute it
- Outcomes: `H_allowed`, `HR_allowed`, `BB_allowed`, `SO`, `HBP_allowed`
- Runs: `ER` (or R), `IP_outs` or `IP`
- Usage: `GS`, `G` (for starter/reliever classification)

**Fielding (player-season, by position if possible):**
- Identifiers: `player_id`, `season`, `position`
- Playing time proxy: `innings` (or games at position)
- Events: `PO`, `A`, `E` (basic)
- Catcher extras (if available): passed balls, CS%, etc.

**League-year context:**
- `season`, `league_id`
- League averages for key rates (K%, BB%, HR/PA, BABIP proxy, runs/game)

**5B. Data source adapters (required)**

Card builder must support adapters that map raw sources into the canonical schema.

- Adapter 1: `csv_adapter_v1` (folder of CSVs)
- Adapter 2: `manual_override_adapter_v1` (small patch file for fixes)

*Note:* exact file names/columns will be defined once you pick a dataset.

#### 6. Outputs (artifacts)

**6A. Player Card (authoritative runtime record)**

Each player-season card must export:

- Stable identifiers
  - `player_id`
  - `season`
  - `card_id` (e.g., `player_id|season`)
  - optional: `team_id` (for Franchise Apex)
- Classification
  - `bats` / `throws` (if available)
  - `primary_role`: `Hitter | Pitcher | TwoWay`
  - `pitcher_role`: `SP | RP | None`
  - `positions`: list with eligibility + confidence
- Traits (0–100)
  - `CON`, `GAP`, `POW`, `EYE`, `AK`, `BNT`
  - `STF`, `CTL`, `CMD`, `STA`
  - `RNG`, `HND`, `ARM`, optional `CTCH`
- Normalized stat layer (for audit)
  - `normalized_rates`: the post-normalization values used to compute traits
- Provenance + explainability
  - `source_version` (dataset/build hash)
  - `build_version` (ratings constants version)
  - `reliability`: per-trait confidence scores based on sample size
  - `notes/tags`: e.g., “small-sample regressed”, “missing defense metrics: fallback used”

**6A(i). Player Card must support “component-ready” provenance (Historical v1 emits a single component)**
Historical v1 Player Cards are built from a single season. However, to avoid rebuilding for Live, the exported card format must support a generalized concept of trait provenance/components, even if historical builds only populate one component.
Add to Player Card output definition:
• trait_provenance (or equivalent) as an optional object that records:
  • build mode (historical)
  • build version
  • source version
  • component list (historical: one component = the season)
Proposed addition (historical v1 example):
{
  "player_id": "...",
  "season": 1997,
  "traits": { "...": 0-100 },
  "reliability": { "...": "0.0-1.0" },
  "trait_provenance": {
    "mode": "historical",
    "components": [
      {
        "name": "season",
        "season": 1997,
        "weight": 1.0
      }
    ]
  }
}
This is deliberately compatible with Live later (where components become yesterday / last 14 / season / prior).

**6B. League/Season Context Tables**

Persist league-year averages and normalization baselines used by the build.

**6C. Build Report**

A build summary that includes:
- number of player-seasons processed
- number skipped + reasons
- missing data counts
- trait distributions (mean/std) by era
- sanity checks results

**6D. The "Provenance" Standard (Live-Proofing)**
To ensure we can blend "Live" stats later without breaking "Historical" cards, every Player Card must follow this "Recipe" format:
• build_timestamp: when this card was generated (new required field)
• build_recipe_version: alias of build_version (do not store two separate versions)
• trait_components: alias of trait_provenance.components (historical v1 = one component: the season)
- Note: This allows Phase 2 to simply add "Last_10_Days" or "Season_To_Date" as additional sources without changing the file structure.

#### 7. Pipeline Stages

This PRD should be implemented as a deterministic pipeline with named stages that match the Game Bible:

##### Stage 1 — `ingestRawData()`

- Load raw batting/pitching/fielding tables via adapter
- Validate data types, uniqueness, and season ranges
- Construct canonical schema tables
- Compute derived fields (PA, BF, IPouts, rate denominators) when possible
- Output: `raw_records` + `league_context`

**Validation rules (minimum):**
- No negative counting stats
- Seasons must be integers and within supported historical range (config)
- Duplicates: same `player_id|season` must resolve deterministically
- If critical denominators missing (cannot compute PA/BF/IP), flag record as unusable

##### Stage 2 — `normalizeEra()`

Convert raw player-season rates into **context-neutral rates**.

Core idea: a player’s outcomes should be translated relative to their league-year environment and mapped into a shared neutral baseline.

**Normalization requirements:**
- Use league-year context to compute relative performance for:
  - `K%` (batters and pitchers)
  - `BB%`
  - `HR/PA` or `HR/BIP` depending on available inputs
  - `BABIP proxy` (requires BIP estimation)
  - `XBH rates` for gap power (2B+3B)
- Do not hardcode era “buffs.” Use context tables.

**Neutral baseline options (choose one for v1):**
- Option A (recommended): normalize to the **global pooled baseline** of all supported seasons in the master universe.
- Option B: normalize to a chosen reference era baseline.

**Sample size handling (required):**

Small samples must be regressed toward league average.

- Compute a reliability weight per rate based on denominator size (PA for batters, BF/IP for pitchers)
- Apply empirical Bayes style shrinkage:

$$ rate_{adj} = w \cdot rate_{player} + (1-w) \cdot rate_{lg} $$

Where `w` is higher for large denominators.

Output: `normalized_rates` for each player-season.

##### Stage 3 — `convertToTraits()`

Translate normalized rates into the 0–100 trait vector used by PRD 1.

**Trait mapping principles:**
- Traits are **monotonic** with respect to their underlying driver (better performance → higher trait)
- Traits have diminishing returns near extremes
- Trait distributions should be stable across eras after normalization
- Mappings should be configurable via `rating_constants_v1.json`

**v1 Trait driver mapping (minimum definition):**

Hitters:
- `EYE` primarily driven by `BB%` (normalized)
- `AK` driven by (inverse) `K%` (normalized)
- `POW` driven by `HR/PA` and/or `ISO` proxy (normalized)
- `GAP` driven by `2B+3B` rates per PA/BIP (normalized)
- `CON` driven by in-play hit ability (BABIP proxy) and/or contact rate; must not be redundant with `AK`
- `BNT` driven by bunting events if available; otherwise use conservative default + allow manual overrides

Pitchers:
- `STF` driven by pitcher K% and HR suppression profile (normalized)
- `CTL` driven by low BB% (normalized)
- `CMD` driven by HR suppression and hit suppression conditional on BIP (if available); also affects variance term in PA engine
- `STA` driven by usage patterns (IP per start, GS%, total IP), normalized by era usage norms

Defense:
- `RNG`, `HND`, `ARM` derived from the best available defensive data.
  - If only basic fielding is available, compute conservative proxy metrics per position.
  - If no defense data exists for a player-season, use position-based default with low reliability.

**Handedness:**
- `bats` and `throws` should be included if available.
- If unavailable, default to `R/R` and tag record with `missing_handedness=true`.

**7A. Ratings Engine must output “Card Recipe” for Explanation Layer parity**
To support later Live transparency (and to strengthen Historical explainability now), the historical builder must persist a “card recipe” record per player-season that can be rendered into manager-facing explanations.
Requirement:
For any player card, engineering must be able to reconstruct:
• raw inputs (rates and denominators)
• league-year context
• normalized rates
• shrinkage/reliability weights applied
• trait mapping parameters used
• final traits produced
This can be stored as:
• a JSON blob alongside the card export, and/or
• a deterministic build report keyed by card_id.
(Exact storage mechanism is implementation-specific; the PRD requirement is traceability.)

#### 8. Role + Position Classification

**8A. Pitcher role (SP vs RP)**

The sim needs to know starter vs reliever for rotation/bullpen logic.

v1 classification rules:
- If `GS` available:
  - `GS / G >= starter_threshold` → `SP`
  - else `RP`
- If `GS` not available, use IP per appearance proxy.

All thresholds are configurable and logged in the build report.

**8B. Primary position + eligibility**

For each player-season:
- Determine primary position by highest playing time (innings or games) at position
- Store eligibility list (e.g., `['1B','OF']`) with confidence weights

Franchise Apex requires you to populate an actual lineup, so position eligibility cannot be hand-wavy.

#### 9. Pilot Library (to unblock testing fast)

Because you said you can’t give meaningful feedback until you can run a test engine, this PRD includes a required pilot output.

**Pilot requirement (v1):** produce Player Cards for a small but real subset that can run end-to-end games.

Minimum viable pilot:
- 4 team-seasons (enough for a short series + matchup variety)
- Each team-season must include:
  - at least 9 hitters with positions
  - at least 5 starters + bullpen arms

Suggested examples (from the Game Bible examples; final set can change):
- 1927 Yankees
- 1986 Mets
- 1998 Yankees
- 2001 Mariners

*Important:* these are *examples*. The actual set depends on your chosen dataset and what’s easiest to ingest first.

#### 10. Integration Contract with PRD 1 (PA Engine) — (Updated: optional metadata passthrough)
PRD 1 (PA Engine) consumes a runtime batter/pitcher input record containing a normalized 0–100 trait vector. PRD 2 is responsible for producing those traits (and supporting auditability).

**10A. Required fields (must be present for Historical v1 integration)**
PRD 2 must provide PRD 1 with the following required fields per participant:
• player_id
• player_type (batter | pitcher | fielder)
• handedness (L | R | S) (or best-available fallback + tag in provenance)
• traits (0–100), using PRD 1’s required keys:
  • Batters: CON, GAP, POW, EYE, AK, BNT
  • Pitchers: STF, CTL, CMD, STA
  • Fielders: RNG, HND, ARM (+ optional CTCH if implemented)
PRD 1 must be able to run deterministically using only these required fields.

**10B. Optional metadata passthrough (not used in outcome math; required for future Live transparency)**
To avoid rebuilding the engine for CM Live and to support manager-facing explanations later, PRD 2 may also output optional metadata that PRD 1 must accept and persist into the PA Receipt unchanged.
Optional passthrough fields from PRD 2 → PRD 1:
• trait_metadata (optional runtime field):
  • mode: historical | live
  • as_of_date (optional)
  • components: list of trait-source components with weights (historical will typically be a single season component; Live will later use yesterday, last_7_14_days, season_2026, prior_2025)
  • reliability: per-trait confidence scores (0.0–1.0) derived from sample size / shrinkage logic
Important contract rule:
PRD 1 must not use trait_metadata to change probabilities in v1. It is passthrough-only and exists for audit + explanation layers.

**10C. Mapping from PRD 2 Player Card → PRD 1 runtime input**
When a PRD 2 Player Card is loaded into a PRD 1 simulation:
• PRD 2 traits → PRD 1 traits
• PRD 2 bats/throws/handedness → PRD 1 handedness
• PRD 2 reliability + trait_provenance (or equivalent) → PRD 1 optional trait_metadata
Historical v1 example (single-component provenance):
{
  "player_id": "player_123",
  "player_type": "batter",
  "handedness": "S",
  "traits": { "CON": 72, "GAP": 58, "POW": 81, "EYE": 64, "AK": 69, "BNT": 35 },
  "trait_metadata": {
    "mode": "historical",
    "components": [
      { "name": "season", "season": 1997, "weight": 1.0 }
    ],
    "reliability": { "CON": 0.92, "GAP": 0.90, "POW": 0.93, "EYE": 0.91, "AK": 0.91, "BNT": 0.40 }
  }
}
This preserves a clean separation:
• PRD 2 builds cards (and provenance)
• PRD 1 resolves PAs using traits
• PRD 1 logs provenance so Live can later explain “why CM differs from MLB” using the same receipt mechanism

#### 11. QA + Validation

**11A. Data sanity tests**
- No card has traits outside 0–100
- No card has missing denominators used in normalization
- No card has `PA=0` for a hitter card or `BF/IP=0` for a pitcher card

**11B. Distribution sanity tests (post-normalization)**
Across the full processed universe (or the pilot dataset initially):
- Trait means/stdevs stay within expected bands
- Extreme outliers are flagged (e.g., `POW>99` but HR rate not elite)

**11C. Cross-era comparability tests**
Pick representative stars from different eras (once data is available) and ensure:
- their trait profiles align with baseball intuition
- league-average players from different eras map near the midline

**11D. Round-trip traceability**
For any player card, engineering must be able to print a “card recipe”:

- raw inputs (original rates)
- league-year context
- normalized rates
- trait mapping parameters
- final traits

#### 12. Acceptance Criteria (Definition of Done)

- [ ] Canonical internal schema defined (batting/pitching/fielding/context)
- [ ] At least one data adapter implemented (CSV folder)
- [ ] Deterministic build with versioned constants (`build_version`)
- [ ] Era normalization implemented + documented
- [ ] Regression-to-mean implemented for small samples
- [ ] Trait conversion implemented for all required traits
- [ ] Role and position classification implemented
- [ ] Player Card export produced (JSON or DB)
- [ ] Build report generated with sanity checks
- [ ] Pilot library generated (min 4 team-seasons) usable to simulate games end-to-end
☐ Player Card export includes optional trait_provenance/components structure (historical uses a single “season” component).
☐ Build output includes per-trait reliability/confidence scores (already in PRD 2; ensure it is consistently populated).
☐ Card recipe / trace record can be generated for any player-season card.

#### 13. Open Questions (decisions we can postpone, but must answer before scaling)

1. **Data source selection**: what dataset are we using for the first ingest (CSV export from where)?
2. **Defense quality**: what defensive fields/metrics will we reliably have? If minimal, what proxy do we accept for v1?
3. **Baserunning**: do we compute `SPD`-like traits now (future), or keep SB/CS only for narrative and later expansion?
4. **Handedness completeness**: do we require bats/throws in v1, or allow default with manual overrides?
5. **Two-way players**: how do we represent them in roster logic (separate cards vs combined)?
6. **Position eligibility rules**: games threshold vs innings threshold; what’s “enough” to qualify?

#### 99. The "Live-Proofing" Safety Contract
1. Trait Purity: The PA Engine (PRD 01) must never calculate a player's skill. It only consumes final 0-100 traits.
2. Metadata Passthrough: Any extra "Live" data (like 'Battery Reliability' or 'Clutch Weights') must be treated as "Passive Metadata." The engine should save it to the Receipt but never use it to change the Home Run vs. Strikeout math in this version.
3. Separation of Concerns: The "Brain" (PRD 01) resolves the play. The "Ingestion" (PRD 02) creates the player. No logic from one shall ever leak into the other.
