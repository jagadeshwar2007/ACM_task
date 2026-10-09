# Task 6: Housing price predictor

Predicts the median house value of a California district using Decision Tree and Random Forest, with feature scaling.

## How to run

```
pip install pandas numpy scikit-learn matplotlib joblib jupyter
cd task_06_housing_price_predictor
jupyter notebook housing_price_predictor.ipynb
```

The dataset is in `data/housing.csv` (20,640 districts, California housing data). Running the notebook also saves the trained model in `model/`, which is used in task 4 and 5.

## What I did

1. Looked at the data. Only `total_bedrooms` has missing values, so I filled them with the median.
2. Split into 80% train and 20% test.
3. Pipeline: fill missing values, StandardScaler for the numeric columns, one-hot encoding for `ocean_proximity`.
4. Trained Linear Regression, Decision Tree and Random Forest and compared them on the test set.

## Results

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Linear Regression | 70,059 | 50,670 | 0.625 |
| Decision Tree | 69,176 | 43,604 | 0.635 |
| Decision Tree (max_depth=10) | 61,280 | 40,556 | 0.713 |
| Random Forest | 48,942 | 31,628 | 0.817 |

Random Forest is the best, so it is the final model. A single decision tree overfits, and the forest fixes that by averaging 100 trees. Median income is the most important feature.

The prices in the dataset are capped at 500,001, so the model can't predict higher than that.
