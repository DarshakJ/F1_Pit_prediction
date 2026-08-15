# Kaggle Playground Series s6e5: F1 Pit Stop Prediction
## Comprehensive Feature Documentation & Winning Strategy Breakdown

---

## 1. Executive Summary & Winning Strategy
In **Kaggle Playground Series Season 6 Episode 5**, the goal is to predict whether a Formula 1 driver will make a pit stop on the upcoming lap (`PitNextLap` binary classification: `0` or `1`).

### Key Findings & Insights
* **Class Imbalance**: The dataset is moderately imbalanced (~80.1% `0` vs ~19.9% `1`). Optimizing the classification probability threshold post-training provides significant accuracy gains.
* **Temporal Sequence Dynamics**: Observations follow sequential race sessions (`Race`, `Year`, `Driver`, `LapNumber`). Pitting is strongly heralded by sudden pace drop-offs relative to stint benchmarks.
* **Crucial Feature Engineering**:
  * **Pace Drop-off / Lags**: Calculating lag lap times (`LapTime_lag1`) and immediate pace deltas (`LapTime - LapTime_lag1`) captures the "in-lap" preparation.
  * **Tire Degradation Intensity**: Normalizing `Cumulative_Degradation` over `TyreLife` captures non-linear tire performance cliffs.
  * **Race Progress / Strategy Windows**: Estimating total race laps (`Estimated_Remaining_Laps`) anchors strategy windows (e.g., compulsory multi-compound stop rules).
  * **Compound Baseline Pace**: Relative lap times compared to compound averages per race (`LapTime_vs_Compound_Mean`) account for track temperature and compound degradation variances.

---

## 2. Complete Feature Reference Guide

### A. Primary Identifiers & Categorical Metadata

#### 1. `id`
* **Data Type:** Integer (`int64`)
* **Unique Values:** 439,140
* **Range:** `0` to `439,139`
* **Description:** Unique row index for each observation.
* **Modeling Usage:** Passive identifier used solely for mapping predictions to submission files. Excluded from model features.

#### 2. `Driver`
* **Data Type:** Categorical String (`object`)
* **Unique Values:** 887 driver codes
* **Sample Categories:** `MAS`, `RAI`, `BAR`, `BUT`, `FIS`, `D109`, `D086`
* **Description:** F1 driver identifier.
* **Modeling Usage:** Driver style affects tire preservation. High cardinality (>800 unique values) requires **Frequency Encoding**, **Target Encoding**, or GBDT categorical handling.

#### 3. `Compound`
* **Data Type:** Categorical String (`object`)
* **Unique Values:** 5 compounds
* **Categories & Counts:**
  * `MEDIUM`: 211,141 laps
  * `HARD`: 170,518 laps
  * `SOFT`: 38,744 laps
  * `INTERMEDIATE`: 17,382 laps
  * `WET`: 1,355 laps
* **Description:** The tire compound used during the lap.
* **Modeling Usage:** Soft compounds provide speed but wear fast; Hard compounds last longer. Key categorical feature for strategy modeling.

#### 4. `Race`
* **Data Type:** Categorical String (`object`)
* **Unique Values:** 26 Grand Prix tracks
* **Top Tracks:** `Dutch Grand Prix`, `Mexico City Grand Prix`, `Pre-Season Testing`, `Hungarian Grand Prix`, `Monaco Grand Prix`
* **Description:** Grand Prix session venue.
* **Modeling Usage:** Different tracks exhibit distinct wear profiles, surface abrasiveness, and pit lane delta losses.

#### 5. `Year`
* **Data Type:** Integer (`int64`)
* **Unique Values:** 4 years (`2022` to `2025`)
* **Mean:** ~2023.52
* **Description:** Season year of the race.
* **Modeling Usage:** Paired with `Race` (`Race + Year`) to group unique race sessions.

---

### B. Race Stint & Telemetry Features

#### 6. `PitStop`
* **Data Type:** Integer (`int64`)
* **Unique Values:** 2 (`0` or `1`)
* **Mean:** 0.136 (~13.6% of laps feature a pit stop)
* **Description:** Indicates whether the driver made a pit stop on the **current** lap.
* **Modeling Usage:** Drivers rarely pit in consecutive laps. A value of `1` on lap $N$ strongly indicates `PitNextLap == 0` on lap $N+1$.

#### 7. `LapNumber`
* **Data Type:** Integer (`int64`)
* **Unique Values:** 78 laps
* **Range:** `1` to `78` (Mean: 23.11, Std: 16.96)
* **Description:** Driver's current lap number in the race.
* **Modeling Usage:** Contextualizes race duration and pit window timings.

#### 8. `Stint`
* **Data Type:** Integer (`int64`)
* **Unique Values:** 8 stints
* **Range:** `1` to `8` (Mean: 1.79, Std: 0.95)
* **Description:** Stint count for the driver (increments by +1 after each pit stop).
* **Modeling Usage:** Most races feature 1-stop or 2-stop strategies (Stints 1 to 3).

#### 9. `TyreLife`
* **Data Type:** Float (`float64`)
* **Unique Values:** 78 values
* **Range:** `1.0` to `77.0` (Mean: 14.16, Std: 9.80)
* **Description:** Total completed laps on the current set of tires.
* **Modeling Usage:** Primary driver of pit stops; higher tire age increases likelihood of pitting.

---

### C. Position & Race Context Features

#### 10. `Position`
* **Data Type:** Integer (`int64`)
* **Unique Values:** 20 positions
* **Range:** `1` (Leader) to `20` (Back of grid)
* **Description:** Driver's current position on track.
* **Modeling Usage:** Leaders often pit to cover undercut threats; trailing drivers pit early to find clean air.

#### 11. `Position_Change`
* **Data Type:** Float (`float64`)
* **Unique Values:** 37 distinct values
* **Range:** `-18.0` to `+18.0` (Mean: +0.10)
* **Description:** Net change in position relative to starting baseline.
* **Modeling Usage:** Rapidly dropping back indicates severe pace/degradation issues.

#### 12. `RaceProgress`
* **Data Type:** Float (`float64`)
* **Unique Values:** 1,898 values
* **Range:** `0.0128` (~1% complete) to `1.0000` (100% complete)
* **Description:** Proportion of total race distance completed.
* **Modeling Usage:** Defines strategy feasibility; drivers rarely pit at $0.98+$ progress unless experiencing damage.

---

### D. Lap Time & Degradation Metrics

#### 13. `LapTime (s)`
* **Data Type:** Float (`float64`)
* **Unique Values:** 37,719 values
* **Range:** `67.69` seconds to `2,507.61` seconds (Mean: 90.95s)
* **Description:** Time taken to complete the current lap.
* **Modeling Usage:** Outliers occur during safety cars or red flags; sudden spikes reflect in-lap prep.

#### 14. `LapTime_Delta`
* **Data Type:** Float (`float64`)
* **Unique Values:** 57,532 values
* **Range:** `-2,403.90` to `+2,423.93` seconds
* **Description:** Pace difference relative to the session average baseline.
* **Modeling Usage:** Quantifies relative pace drop-off.

#### 15. `Cumulative_Degradation`
* **Data Type:** Float (`float64`)
* **Unique Values:** 142,701 values
* **Range:** `-274.56` to `+2,412.03` seconds (Mean: -25.72s)
* **Description:** Accumulated degradation built up across the current tire stint.
* **Modeling Usage:** Core metric for tracking tire degradation cliffs.

---

### E. Target Variable

#### 16. `PitNextLap` *(Train Set Only)*
* **Data Type:** Float / Binary (`0.0` or `1.0`)
* **Distribution:**
  * `0.0` (No Pit): **80.1%** (351,759 rows)
  * `1.0` (Pit Next Lap): **19.9%** (87,381 rows)
* **Description:** Binary target to predict whether the driver enters the pit lane on `LapNumber + 1`.

---

## 3. Summary Feature Matrix Table

| Feature Name | Data Type | Missing Values | Range / Unique Values | Key Modeling Role |
| :--- | :--- | :--- | :--- | :--- |
| **`id`** | Integer | None | `0` to `439,139` | Sample indexing (dropped from training) |
| **`Driver`** | Categorical | None | 887 unique drivers | Driving style & tire wear characteristics |
| **`Compound`** | Categorical | None | 5 compounds | Base tire degradation rate |
| **`Race`** | Categorical | None | 26 Grand Prix tracks | Track layout & surface wear profile |
| **`Year`** | Integer | None | `2022` to `2025` | Season context & technical regulations |
| **`PitStop`** | Binary | None | `0` or `1` | Pitting on current lap makes next-lap pit unlikely |
| **`LapNumber`** | Integer | None | `1` to `78` | Lap position within the race session |
| **`Stint`** | Integer | None | `1` to `8` | Current tire stint count |
| **`TyreLife`** | Float | None | `1.0` to `77.0` | Age of current tire set |
| **`Position`** | Integer | None | `1` to `20` | Current track rank (undercut/overcut tactical context) |
| **`Position_Change`** | Float | None | `-18.0` to `+18.0` | Relative field movement |
| **`RaceProgress`** | Float | None | `0.0128` to `1.0000` | Normalized progress toward race finish |
| **`LapTime (s)`** | Float | None | `67.69` to `2507.61` | Raw driver lap time in seconds |
| **`LapTime_Delta`** | Float | None | `-2403.89` to `+2423.93` | Pace delta relative to session pace |
| **`Cumulative_Degradation`**| Float | None | `-274.56` to `+2412.03` | Total degradation accumulated on current tires |
| **`PitNextLap` (Target)** | Binary | N/A (Train) | `0.0` or `1.0` | Target prediction: Will driver pit on next lap? |
