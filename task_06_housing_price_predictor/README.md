# Task 6 — End-to-End Predictive Model (California Housing Prices)

A complete regression pipeline predicting the **median house value** of a California district using
**Decision Trees and Random Forests with feature scaling**.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
cd task_06_housing_price_predictor
jupyter notebook housing_price_predictor.ipynb     # run all cells (~1 min)
```

The notebook is already executed, so you can read the outputs on GitHub without running anything.

## Pipeline

1. **Explore** — distributions, location/price map, correlations (`outputs/eda.png`). Found that the target is capped at $500,001.
2. **Split** — 80/20 train/test, fixed random seed.
3. **Preprocess** inside one scikit-learn `Pipeline`: median imputation of the 207 missing `total_bedrooms` → **StandardScaler** on numeric features → one-hot encoding of `ocean_proximity`.
4. **Train & compare** — Linear Regression (baseline), Decision Tree, depth-limited Decision Tree, Random Forest.
5. **Tune lightly** with 5-fold cross-validation on the training set only.
6. **Evaluate** on the untouched test set and plot actual-vs-predicted, residuals, feature importances.
7. **Save** the pipeline to `model/housing_model.joblib` (used by Tasks 4 and 5).

## Key results (test set)

| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| Linear Regression (baseline) | 70,059 | 50,670 | 0.625 |
| Decision Tree | 69,176 | 43,604 | 0.635 |
| Decision Tree (max_depth=10) | 61,280 | 40,556 | 0.713 |
| **Random Forest (final)** | **48,942** | **31,628** | **0.817** |

- The final Random Forest is typically about **$31.6k** off on a house and explains **82%** of the variance.
- A single unrestricted tree overfits; averaging 100 trees cuts the error by about 30% versus one tree.
- The three forest settings I cross-validated scored within noise of each other (CV RMSE 49.3k–49.5k, std ≈ 0.6k), so I kept the simplest.
- **Most important features:** `median_income` by a wide margin, then location (`INLAND`, longitude, latitude).
- **Limitation:** prices are capped at $500,001 in the data, so the model under-predicts the most expensive districts.

## Files
- `housing_price_predictor.ipynb` — the full pipeline (executed)
- `data/housing.csv` — dataset (20,640 rows; source: the public California Housing dataset, as distributed in *Hands-On Machine Learning* by A. Géron)
- `model/` — saved pipeline and metadata
- `outputs/` — charts
