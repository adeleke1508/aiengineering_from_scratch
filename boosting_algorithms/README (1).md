# Will They Buy an EV? Boosting Algorithms from Scratch

Binary classification notebook that predicts whether a person will buy an electric vehicle (`Will_Buy_EV`). The main goal is to **implement AdaBoost from scratch** (NumPy and pandas, with decision stumps as weak learners) and benchmark it against scikit-learn's `AdaBoostClassifier` using ROC AUC.

## Dataset

| | |
|---|---|
| `train.csv` | 668,665 rows, 15 columns, includes the target `Will_Buy_EV` (`Yes`/`No`) |
| `test.csv` | Same features, no target |
| Missing values | None |

| Type | Columns |
|---|---|
| Numeric | `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level` |
| Nominal | `Gender`, `City_Type`, `Current_Car_Type` |
| Ordinal | `Range_Anxiety_Level` (Low < Medium < High) |
| Binary | `Home_Charging_Possible`, `Subsidy_Available` |
| Identifier | `id` (dropped, no signal) |

## Approach

1. **Preprocessing**
   - Stratified 80/20 train/validation split (`random_state=42`)
   - Ordinal map for `Range_Anxiety_Level`, `No`/`Yes` to 0/1 for binary columns, one-hot for nominal columns
   - Validation columns reindexed to match train
   - `StandardScaler` fitted on train only, then applied to validation and test
2. **Model:** `AdaBoostScratch` with `DecisionStump` weak learners (default 50 estimators). The stump finds its best split with a sorted cumulative-sum search, so one fit runs in O(n log n · d) instead of O(n² · d).
3. **Evaluation:** ROC AUC on the validation set, computed from continuous scores (`predict_proba` / `decision_function`).
4. **Baseline:** scikit-learn `AdaBoostClassifier()` with default settings.

## Results

| Model | Validation ROC AUC |
|---|---|
| scikit-learn `AdaBoostClassifier` | 0.9358 |
| `AdaBoostScratch` | *add your number after re-running* |

## Things to know

- **Labels:** the scratch model needs labels in `{-1, +1}`. Training it on `{0, 1}` labels makes it learn nothing and gives an AUC of exactly 0.5. The notebook converts with `np.where(y == 1, 1, -1)`. scikit-learn and the metric use the original `{0, 1}` labels.
- **Scores for AUC:** `AdaBoostScratch.predict` returns hard `-1/+1` labels. ROC AUC needs `decision_function` or `predict_proba`, so keep both methods in `boosting.py`.
- **Scaling:** trees and stumps are invariant to feature scaling, so it does not change the results here. It is kept so the same pipeline works for scale-sensitive models.

## Project structure

```
.
├── Will_buy_EV.ipynb      # analysis, preprocessing, training, evaluation
├── boosting.py            # AdaBoostScratch (and gradient boosting classes)
├── decisions_stump.py     # DecisionStump weak learner
└── README.md
```

## How to run

```bash
pip install numpy pandas scikit-learn jupyter
jupyter notebook Will_buy_EV.ipynb
```

Set `DATA_DIR` in the data-loading cell to the folder that holds `train.csv` and `test.csv`. Run the cells top to bottom. If you edit `boosting.py` or `decisions_stump.py`, restart the kernel so the notebook picks up the changes.

## Possible extensions

- Tune `n_estimators` and plot validation AUC against it
- Compare with the from-scratch gradient boosting classifier
- Rank features by total stump `alpha`
- Use quantile thresholds in the stump to speed up training further
