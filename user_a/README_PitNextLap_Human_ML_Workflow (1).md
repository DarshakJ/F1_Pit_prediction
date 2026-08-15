# PitNextLap — Human-Implemented Machine Learning Workflow

## 1. Executive Summary

This project develops a transparent, human-implemented machine learning solution for predicting whether a Formula 1 pit stop will occur on the **next lap**.

The objective is to predict:

> **`PitNextLap` = 1 if a pit stop occurs on the next lap, otherwise 0.**

The original benchmark was created using AutoGluon. The purpose of this notebook is **not** to reproduce AutoGluon's automation, but to understand the modeling choices that produced the strong benchmark and translate those choices into an explicit, auditable machine-learning workflow.

The final workflow deliberately uses conventional Python/scikit-learn-style modeling with explicit preprocessing, validation, model comparison, hyperparameter tuning, interpretation and final submission generation.

### Final result from the executed notebook

| Metric | Result |
|---|---:|
| Training rows | **540,445** |
| Modeling features | **14** |
| Positive target rate | **20.94%** |
| Selected model | **XGBoost** |
| Validation ROC-AUC | **0.95678** |
| Validation PR-AUC | **0.85167** |
| Validation Accuracy | **0.90978** |
| Validation Precision | **0.79406** |
| Validation Recall | **0.76845** |
| Validation F1 | **0.78104** |

The result is close to the original AutoGluon benchmark, which reported approximately **0.95824 ROC-AUC for XGBoost** and approximately **0.95999 for its weighted ensemble**.

The important point is that the human implementation obtains a validation ROC-AUC of approximately **95.68%** without using AutoGluon or an automated ensemble.

---

# 2. Problem Understanding

The task is a binary classification problem.

### Target

`PitNextLap`

- `0` → no pit stop on the next lap
- `1` → pit stop on the next lap

### Why ROC-AUC?

The original benchmark used **ROC-AUC** as its evaluation metric. We therefore retained ROC-AUC as the primary metric.

ROC-AUC is appropriate because the model produces a probability/ranking rather than only a hard 0/1 decision.

We also calculate **PR-AUC / Average Precision** because the positive class is smaller than the negative class.

---

# 3. Dataset Overview

The executed notebook loaded the same files referenced by the original experiment:

- `train.csv`
- `train_merged.csv`
- `test.csv`
- `sample_submission.csv`

The main modeling table is:

### `train_merged.csv`

- **540,445 rows**
- **15 columns**
- 14 predictor variables
- 1 target variable: `PitNextLap`

The other files were also present:

| Dataset | Rows | Columns |
|---|---:|---:|
| `train_merged.csv` | 540,445 | 15 |
| `train.csv` | 439,140 | 16 |
| `test.csv` | 188,165 | 15 |
| `sample_submission.csv` | 188,165 | 2 |

The notebook intentionally uses `train_merged.csv` for modeling because that is the table used by the successful AutoGluon experiment.

---

# 4. Feature Inventory

The modeling dataset contains 14 predictors.

### Categorical variables

- `Driver`
- `Compound`
- `Race`

### Numeric variables

- `Year`
- `PitStop`
- `LapNumber`
- `Stint`
- `TyreLife`
- `Position`
- `LapTime (s)`
- `LapTime_Delta`
- `Cumulative_Degradation`
- `RaceProgress`
- `Position_Change`

### Target

- `PitNextLap`

---

# 5. Data Quality Findings

## 5.1 Dataset shape

The main training table contains:

**540,445 observations × 15 columns**

This is a sufficiently large dataset for tree-based machine learning.

---

## 5.2 Duplicate rows

The executed EDA found:

**0 duplicate rows**

This is useful because duplicate observations can artificially inflate model performance or cause train/validation contamination.

No duplicate rows were removed.

---

## 5.3 Duplicate columns

The dataset contained:

**0 duplicate column names**

---

## 5.4 Missing values

The EDA found:

**0 missing values across the 14 predictor variables.**

This simplifies the dataset considerably.

Nevertheless, the notebook keeps imputation inside the preprocessing pipelines. This makes the workflow robust if the input data later contains missing values.

The key principle is:

> Preprocessing should be learned from training data rather than from the entire dataset before validation.

---

# 6. Target Distribution

The target distribution was:

| `PitNextLap` | Count | Proportion |
|---|---:|---:|
| 0 | 427,273 | 79.06% |
| 1 | 113,172 | 20.94% |

Therefore:

**Positive-class rate = 20.94%**

This is an imbalanced binary classification problem, but it is **not extremely imbalanced**.

Approximately one out of every five observations corresponds to a next-lap pit stop.

### Why this matters

Accuracy alone would not be sufficient for model comparison.

For example, a trivial model predicting class 0 for every observation would already obtain roughly 79% accuracy but would have no ability to identify pit stops.

Therefore:

- ROC-AUC is the primary metric.
- PR-AUC is used as a secondary metric.
- Precision, recall and F1 are also reported.

---

# 7. Numerical EDA

The numerical variables showed several important patterns.

## 7.1 Lap number

- Mean: **24.48**
- Median: **21**
- Minimum: **1**
- Maximum: **78**

The target relationship was strong:

| Target | Mean LapNumber |
|---|---:|
| 0 | 22.24 |
| 1 | 32.92 |

This indicates that pit-stop decisions are strongly associated with race progression.

A pit stop is naturally more likely later in a stint/race than during the opening laps.

---

## 7.2 Tyre life

- Mean: **14.23 laps**
- Median: **12 laps**
- Maximum: **78 laps**

Target-level means:

| Target | Mean TyreLife |
|---|---:|
| 0 | 12.84 |
| 1 | 19.47 |

This is one of the clearest signals in the dataset.

The model can learn that as a tyre remains in use for longer, the probability of a pit stop on the following lap tends to increase.

---

## 7.3 Cumulative degradation

Mean:

**-26.44**

Target-level means:

| Target | Mean Cumulative Degradation |
|---|---:|
| 0 | -21.75 |
| 1 | -44.13 |

This feature shows substantial separation between the two target classes.

It is therefore an important candidate predictor for pit-stop behavior.

---

## 7.4 Race progress

Mean:

**0.355**

Target-level means:

| Target | Mean RaceProgress |
|---|---:|
| 0 | 0.333 |
| 1 | 0.442 |

This supports the same general observation as `LapNumber`: the probability of a pit stop changes as the race progresses.

---

## 7.5 Position change

The mean target-level difference is smaller than for tyre/race variables, but the positive class has a somewhat larger average position change.

This suggests that race dynamics can contribute additional predictive information.

---

# 8. Important Correlation Findings

Spearman correlation was used because it captures monotonic relationships and is less sensitive to extreme values than Pearson correlation.

The strongest relationships included:

| Feature 1 | Feature 2 | Spearman correlation |
|---|---|---:|
| `LapNumber` | `RaceProgress` | **0.9803** |
| `LapNumber` | `Stint` | **0.7417** |
| `Stint` | `RaceProgress` | **0.7374** |
| `LapNumber` | `TyreLife` | **0.6553** |
| `TyreLife` | `RaceProgress` | **0.6424** |
| `LapTime (s)` | `LapTime_Delta` | **0.4166** |

### Interpretation

The strong relationship between `LapNumber` and `RaceProgress` is expected.

Both describe where the driver is within the race.

The important modeling decision was **not to blindly remove correlated variables**.

Tree-based models can use correlated variables effectively, and removing one based only on correlation could discard useful predictive information.

Instead, the notebook retains them and evaluates their contribution through model validation and permutation importance.

---

# 9. Categorical Variables

The dataset contains:

- 887 unique `Driver` values
- 28 unique `Race` values
- 5 unique tyre compounds

These variables were retained.

### Why?

Driver, race and tyre compound can influence pit-stop strategy.

For example:

- different drivers may have different historical strategies,
- different races have different circuit characteristics,
- different compounds have different degradation and expected stint lengths.

The model therefore receives these variables rather than discarding them.

---

# 10. Leakage Considerations

Leakage was treated as a major modeling concern.

A model can achieve an apparently outstanding AUC if it is accidentally given information that would only be known after the prediction point.

The notebook therefore:

1. Separates training and validation data.
2. Places preprocessing inside sklearn pipelines.
3. Avoids fitting transformations on the complete dataset before validation.
4. Reviews identifier/high-cardinality variables.
5. Treats domain availability of features as a final human review requirement.

The EDA did not find any near-unique identifier-style predictor based on the simple uniqueness heuristic.

However, **domain-level leakage cannot be proven from column names alone**. The panel presentation should explicitly state that prediction-time availability must be verified against the F1 data-generation process.

---

# 11. Validation Strategy

A stratified 80/20 train-validation split was used.

### Why stratification?

The target is imbalanced.

Stratification keeps the class proportions approximately consistent between training and validation.

### Why a separate validation set?

The validation set provides a clean final comparison after model selection.

In addition, 3-fold stratified cross-validation was used during baseline comparison and hyperparameter tuning.

This gives two perspectives:

- CV → stability across multiple folds
- Held-out validation → final model comparison

---

# 12. Preprocessing Strategy

Two preprocessing pipelines were used.

## Logistic Regression

- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- One-hot encoding for categorical variables
- Standard scaling for numerical variables

This is appropriate because Logistic Regression is a linear model and benefits from standardized numerical variables.

## Tree models

- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- Ordinal encoding for categorical variables

Tree models do not require standardization.

Ordinal encoding also avoids creating a potentially very large one-hot matrix for high-cardinality categorical features such as `Driver`.

---

# 13. Algorithms Compared

The notebook intentionally compares multiple classical algorithms rather than relying on one model.

### Models

1. Logistic Regression
2. Random Forest
3. Extra Trees
4. HistGradientBoosting
5. LightGBM
6. LightGBM Extra Trees
7. XGBoost
8. CatBoost

The boosted-tree families were given special attention because the original AutoGluon experiment showed that they were the strongest model families.

---

# 14. Baseline Model Results

The 3-fold CV results were:

| Model | CV ROC-AUC | CV PR-AUC |
|---|---:|---:|
| Random Forest | **0.95275** | 0.84049 |
| Extra Trees | 0.94939 | 0.83345 |
| XGBoost | 0.94670 | 0.81787 |
| LightGBM | 0.94506 | 0.81246 |
| HistGradientBoosting | 0.94368 | 0.80769 |
| CatBoost | 0.94130 | 0.80217 |
| LightGBM Extra Trees | 0.93336 | 0.78123 |
| Logistic Regression | 0.84339 | 0.56929 |

### Interpretation

The tree ensemble methods substantially outperform Logistic Regression.

This tells us that the relationship between race state, tyre condition, lap progression and pit-stop probability is highly nonlinear.

Random Forest had the strongest baseline CV AUC, while the boosted models became particularly competitive after tuning.

---

# 15. Baseline Held-Out Validation Results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Random Forest | **0.95440** | 0.84681 |
| Extra Trees | 0.95087 | 0.83831 |
| XGBoost | 0.94583 | 0.81577 |
| LightGBM | 0.94450 | 0.81107 |
| HistGradientBoosting | 0.94265 | 0.80575 |
| CatBoost | 0.94047 | 0.79997 |
| LightGBM Extra Trees | 0.93223 | 0.77917 |
| Logistic Regression | 0.84274 | 0.57324 |

### Key insight

Random Forest is an excellent baseline.

However, baseline performance alone does not tell us which boosted model can benefit most from tuning.

That motivated a focused tuning stage.

---

# 16. Hyperparameter Tuning

Tuning was intentionally focused on:

- LightGBM
- LightGBM Extra Trees
- XGBoost
- CatBoost

This follows the evidence from the original AutoGluon experiment.

### Parameters explored

For boosting models, the search considered combinations of:

- number of estimators / iterations
- learning rate
- tree depth / number of leaves
- minimum child size
- row subsampling
- column subsampling
- L1 regularization
- L2 regularization
- CatBoost-specific randomness and leaf regularization

A randomized search with:

- 20 parameter configurations
- 3 CV folds

was used for each candidate.

This resulted in 60 fits per tuned model.

---

# 17. Tuning Results

| Model | Best CV ROC-AUC |
|---|---:|
| LightGBM | **0.95631** |
| XGBoost | 0.95601 |
| CatBoost | 0.94967 |
| LightGBM Extra Trees | 0.94963 |

### Interpretation

Tuning substantially improved the boosted models.

The most important result is that both LightGBM and XGBoost reached approximately **0.956 CV ROC-AUC**.

This confirms that the original AutoML result was not simply the consequence of an arbitrary model choice.

---

# 18. Final Held-Out Comparison

After tuning, the models were evaluated on the untouched validation set.

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0.95678** | **0.85167** | 0.90978 | 0.79406 | 0.76845 | 0.78104 |
| LightGBM | 0.95676 | 0.85082 | 0.90974 | 0.79340 | 0.76929 | 0.78116 |
| LightGBM Extra Trees | 0.94959 | 0.83033 | 0.90066 | 0.77683 | 0.73743 | 0.75662 |
| CatBoost | 0.94913 | 0.82662 | 0.89905 | 0.76917 | 0.73995 | 0.75428 |

### Final selection

**XGBoost was selected.**

The difference between XGBoost and LightGBM is extremely small:

**0.956784 vs 0.956760**

Therefore this should not be presented as evidence that XGBoost is dramatically superior.

Instead:

> XGBoost was selected because it achieved the highest held-out ROC-AUC in the executed comparison, while LightGBM produced essentially equivalent performance.

This is a more defensible statement for a panel.

---

# 19. Final Classification Performance

For XGBoost:

### ROC-AUC

**0.95678**

This means the model has strong ability to rank observations according to their likelihood of a next-lap pit stop.

### PR-AUC

**0.85167**

This is also strong and confirms that the model is not achieving its ROC-AUC purely through poor positive-class precision.

### Accuracy

**90.98%**

### Precision

**79.41%**

### Recall

**76.84%**

### F1

**78.10%**

---

# 20. Confusion Matrix Interpretation

The validation confusion matrix was:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 80,944 | 4,511 |
| Actual 1 | 5,241 | 17,393 |

Therefore:

- True negatives = 80,944
- False positives = 4,511
- False negatives = 5,241
- True positives = 17,393

The model successfully identifies a large majority of the positive cases while keeping false positives reasonably controlled.

---

# 21. Feature Importance

Permutation importance was used because it is model-agnostic and easier to explain than relying only on tree split importance.

The strongest features were:

| Rank | Feature | Permutation importance |
|---:|---|---:|
| 1 | **TyreLife** | **0.15795** |
| 2 | **Stint** | **0.09439** |
| 3 | **Year** | **0.05499** |
| 4 | **RaceProgress** | **0.03496** |
| 5 | **LapTime_Delta** | **0.02118** |
| 6 | `Race` | 0.01637 |
| 7 | `LapNumber` | 0.01256 |
| 8 | `Cumulative_Degradation` | 0.00973 |
| 9 | `LapTime (s)` | 0.00898 |
| 10 | `Compound` | 0.00705 |

### Main interpretation

The model is driven primarily by **race/stint/tyre state**.

This is highly intuitive for the problem.

`TyreLife` being the strongest feature indicates that the length of time the current tyre has been used is highly informative about whether the driver is likely to pit next.

`Stint` is also highly important because pit strategies are naturally organized around tyre stints.

`RaceProgress` and `LapNumber` capture where the driver is in the race.

---

# 22. Score-Distribution Sanity Check

The validation predictions were divided into score deciles.

The highest predicted-probability bucket had:

**90.72% actual positive rate**

The next bucket had:

**68.81% actual positive rate**

The third-highest bucket had:

**33.42% actual positive rate**

The lowest bucket had only:

**0.046% actual positive rate**

This is a strong qualitative confirmation that the model ranks observations effectively.

In other words:

> High model scores correspond strongly to actual next-lap pit stops.

That is exactly the behavior we want from a high-ROC-AUC model.

---

# 23. Why the Human Implementation Is Valuable

The AutoML benchmark gives a strong performance reference, but it hides many decisions.

The human implementation makes those decisions explicit:

- what data is used,
- why it is used,
- how missing values are handled,
- how categorical variables are encoded,
- why ROC-AUC is selected,
- how the train/validation split works,
- which algorithms are compared,
- why those algorithms were chosen,
- which hyperparameters are tuned,
- how the winner is selected,
- which features drive predictions,
- and how the final Kaggle submission is generated.

This makes the work easier to defend in a panel discussion.

---

# 24. Relationship to the Original AutoGluon Benchmark

The original AutoGluon experiment used:

- target: `PitNextLap`
- evaluation metric: `roc_auc`
- training table: `df_train_merged`
- GBM / LightGBM
- XGBoost
- CatBoost
- LightGBM Extra Trees
- no stack levels
- best-quality AutoML configuration

Its strongest results were approximately:

| AutoGluon model | ROC-AUC |
|---|---:|
| Weighted Ensemble | **0.95999** |
| XGBoost | **0.95824** |
| CatBoost | **0.95814** |
| LightGBM Extra Trees | **0.95290** |
| LightGBM | **0.95103** |

The human implementation achieved:

**XGBoost validation ROC-AUC = 0.95678**

The gap to the original standalone XGBoost result is approximately 0.00146 AUC.

This is a very small difference and demonstrates that the explicit classical workflow is capable of reproducing most of the predictive performance without relying on AutoGluon's automation.

---

# 25. Why We Did Not Force the Model Above 95%

The objective was to obtain strong predictive performance while keeping the solution honest and reproducible.

Repeatedly tuning against the same validation set can eventually overfit the validation set.

Therefore the workflow follows this principle:

> A validation score should be used to evaluate a modeling decision, not become the target that the analyst repeatedly optimizes against.

If further improvement is required, the next logical steps are:

1. Feature engineering
2. Better temporal/group-aware validation
3. Domain-specific F1 features
4. Leakage review
5. Probability calibration if required
6. Carefully controlled additional tuning

Increasing model complexity alone is not necessarily the right solution.

---

# 26. Final Kaggle Submission

The notebook now contains a final submission section at the very bottom.

It:

1. Uses the frozen selected model.
2. Generates `PitNextLap` probabilities for `test.csv`.
3. Preserves the `id` column from `sample_submission.csv`.
4. Validates row counts.
5. Writes the final file as:

```text
/kaggle/working/submission.csv
```

The expected submission format is:

| id | PitNextLap |
|---:|---:|
| 439140 | probability |
| 439141 | probability |
| ... | ... |

The model should submit **probabilities**, not hard 0/1 predictions, because the competition metric is ROC-AUC.

---

# 27. Recommended Panel Presentation Structure

A concise panel presentation can follow this sequence:

### Slide 1 — Business / Technical Objective
Predict whether an F1 pit stop will happen on the next lap.

### Slide 2 — Dataset
540K observations, 14 predictors, 20.94% positive class.

### Slide 3 — EDA Findings
Highlight:

- tyre life
- stint
- lap number
- race progress
- cumulative degradation
- strong relationship between lap progression and pit strategy

### Slide 4 — Modeling Strategy
Explain:

- Logistic Regression baseline
- tree ensembles
- GBM / LightGBM
- XGBoost
- CatBoost
- focused tuning

### Slide 5 — Model Comparison
Show the final ROC-AUC comparison.

### Slide 6 — Final Model
XGBoost:

**ROC-AUC = 0.95678**

### Slide 7 — Explainability
Show permutation importance.

Key message:

> TyreLife and Stint are the strongest predictive variables.

### Slide 8 — Prediction Quality
Show the score-decile table / ROC curve.

### Slide 9 — Kaggle Submission
Explain that probabilities are generated for every test observation and written using the competition's required schema.

### Slide 10 — Conclusion
The human implementation reproduces approximately 95.7% ROC-AUC while remaining transparent, reproducible and easy to explain.

---

# 28. Key Takeaways

1. **The problem has strong predictive signal.**
2. **The target has a 20.94% positive rate.**
3. **There are no missing values in the current modeling data.**
4. **There are no duplicate observations.**
5. **TyreLife is the strongest feature.**
6. **Stint is the second strongest feature.**
7. **Race progression is highly informative.**
8. **LapNumber and RaceProgress are highly correlated (0.9803).**
9. **Tree-based models substantially outperform Logistic Regression.**
10. **Random Forest is an excellent baseline.**
11. **Focused tuning makes XGBoost and LightGBM the strongest models.**
12. **XGBoost achieves 0.95678 validation ROC-AUC.**
13. **LightGBM is essentially tied at 0.95676.**
14. **The model strongly separates high-risk and low-risk pit-stop observations.**
15. **The workflow avoids AutoML and makes the complete modeling process auditable.**
16. **The final notebook now includes Kaggle-ready probability submission generation.**

---

# 29. Final Conclusion

The analysis demonstrates that next-lap pit-stop prediction can be modeled very effectively using conventional supervised machine learning.

The strongest predictive information comes from the driver's current race and tyre state, especially:

- `TyreLife`
- `Stint`
- `RaceProgress`
- `LapNumber`
- `LapTime_Delta`
- `Cumulative_Degradation`

The final XGBoost model achieves:

> **95.678% validation ROC-AUC**

with:

> **85.167% PR-AUC**

and approximately:

> **90.98% accuracy**

The result is close to the original AutoGluon benchmark while being substantially easier to explain and audit.

The important achievement is therefore not simply the AUC number. It is the combination of:

**data understanding → leakage awareness → transparent EDA → multiple model comparison → focused tuning → validation → explainability → final Kaggle submission.**
