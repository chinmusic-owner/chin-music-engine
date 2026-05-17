### PRD 03: The Value Engine
#### SimWAR → Marginal Wins → SGP-Derived Salaries

**Version:** 1.0  
**Status:** Draft  
**Owner:** TBD  
**Depends On:** PRD 01 (PA Engine), PRD 02 (Player Cards)  
**Feeds Into:** PRD 04 (Roster Management), PRD 05 (League/Game Loop)

#### 1. Purpose

Build the **Value Engine** that converts cross-era Player Cards into a functional game economy. This system translates normalized performance (traits) into **Marginal Wins** and **Salary Values** derived from a hard salary cap.

This is the bridge from **Simulation to Game**. Every roster decision in Chin Music is an exercise in resource allocation under constraint.

#### 2. Product Outcome

- **Economic Pressure:** Make the **$260M active roster cap** a meaningful strategic hurdle.
- **Fair Valuation:** Ensure a 1927 legend and a 2000 ace are priced accurately relative to their marginal contribution to winning.
- **Explainability:** Provide a "Value Receipt" so managers can see exactly how much they are paying for a player's bat vs. their defense or positional scarcity.

#### 3. Scope

**In Scope (v1):**
- Simulation-based valuation (deterministic Monte Carlo runs using PRD 01).
- **Replacement Level logic** by position (C, IF, 1B, OF, UTIL) and role (SP, RP).
- **League Rule Sensitivity:** Adjusting scarcity and replacement based on `use_dh` toggle.
- **Salary Calibration:** Mapping wins to dollars using a $260M cap across 12 teams.
- **Minimum Salary Enforcement:** Fixed at **$1.50M** per active roster slot.
- **Defense Weighting:** Calibrating defense to ~10% of total economic value.

**Out of Scope (v1):**
- Dynamic in-season price inflation/deflation.
- Park factors or weather-based valuation.
- Multi-season contract inflation (v1 assumes "as-played" or "standardized" one-year values).

#### 4. Baseline League Config (The Constants)

| Parameter | Value |
|---|---|
| **Teams** | 12 |
| **Active Roster Size** | 26 |
| **Active Roster Cap** | $260,000,000 |
| **Total League Budget** | $3,120,000,000 |
| **Minimum Salary** | $1,500,000 |
| **Defense Scalar** | Target ~10% of total pool value |
| **Runs Per Win (RPW)** | 10.0 (Configurable baseline) |
| **DH Rule** | Boolean (Impacts roster composition + replacement depth) |

#### 5. Valuation Method (Sim-Driven)

PRD 03 does not use external WAR. It uses PRD 01 to estimate **Marginal Runs**.

##### 5A. Evaluation Setup
For every valuation run, initialize a **Neutral Environment**:
- **Baseline Pitcher:** A "League Average" card (traits ~70/100).
- **Baseline Hitter:** A "League Average" card (traits ~70/100).
- **Baseline Defense:** "League Average" defensive traits per position.
- **Determinism:** Use a fixed `valuation_seed` so player value only changes if their traits or the constants change.

##### 5B. Hitter RAR (Runs Above Replacement)
1. Simulate 10,000 PAs for `player_card` vs `baseline_pitcher`.
2. Convert outcomes to runs using **Linear Weights**.
3. Identify the **Scarcity Group**.
4. Determine the **Replacement Level Hitter** for that group.
5. Compute: `(Runs_Per_PA_Player - Runs_Per_PA_Rep) * PA_Season = Hitter_RAR`.

##### 5C. Pitcher RAR (Runs Above Replacement)
1. Simulate 10,000 PAs for `baseline_hitter` vs `pitcher_card`.
2. Compare vs the **Replacement Level Pitcher** for their role.
3. Compute: `(Runs_Per_PA_Rep - Runs_Per_PA_Player) * BF_Season = Pitcher_RAR`.

##### 5D. Defense Pricing (~10% Weight)
Defense value is computed from traits and scaled so total defense ≈ 10% of total value.

#### 6. Wins Above Replacement (WAR)

$$ WAR = rac{RAR_{bat} + RAR_{pit} + RAR_{def}}{RPW} $$

#### 7. The Salary Model

$$ DPW = rac{Total\ Budget - (Roster\ Slots \cdot Min\ Salary)}{Total\ Positive\ Wins} $$

$$ Salary = 1{,}500{,}000 + (WAR \cdot DPW) $$

#### 8. Output

Value record per player with salary, WAR, and breakdown.

#### 9. Acceptance Criteria

- Minimum salary enforced
- Cap calibration correct
- Deterministic outputs
- Defense ~10% weighting

#### 10. Open Questions

- Reserve salary handling
- Multi-position logic
- Bench value calibration
