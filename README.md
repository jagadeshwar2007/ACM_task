# ACM Student Chapter — AI & ML Recruitment 2026

Five tasks, one folder each, each with its own README (how to run it and the key results).

| Folder | Task | Highlight |
|---|---|---|
| [`task_06_housing_price_predictor`](task_06_housing_price_predictor) | 6 — End-to-End Predictive Model | Imputation → StandardScaler → Random Forest; R² 0.817, RMSE ≈ $48.9k |
| [`task_04_ml_model_api`](task_04_ml_model_api) | 4 — ML Model Web API | FastAPI service serving the Task 6 model as JSON, with validation and tests |
| [`task_05_interactive_dashboard`](task_05_interactive_dashboard) | 5 — Interactive AI Dashboard | Streamlit app that re-predicts and redraws charts live as inputs change |
| [`task_09_spam_detector`](task_09_spam_detector) | 9 — Spam Detector | TF-IDF + LinearSVC on SMS data; accuracy 97.97%, spam F1 0.918 |
| [`task_12_mnist_digit_classifier`](task_12_mnist_digit_classifier) | 12 — Handwritten Digit Classifier | PyTorch CNN; **99.18%** MNIST test accuracy |

Tasks 4 and 5 are built on the model trained in Task 6 (`task_06_.../model/`), so they share a single copy of it.

## Quick start

> The trained housing model is included (`task_06_.../model/housing_model.joblib`, 30 MB), so Tasks 4 and 5 run straight away. To retrain it, re-run the Task 6 notebook (about 1 minute).

```bash
pip install -r requirements.txt

# Task 4 — API (docs at http://127.0.0.1:8000/docs)
cd task_04_ml_model_api && uvicorn main:app --reload

# Task 5 — dashboard
cd task_05_interactive_dashboard && streamlit run app.py
```

Notebooks (Tasks 6, 9, 12) are committed already executed, so results are readable directly on GitHub.
Python 3.10+ recommended; scikit-learn is pinned because the saved model files are version-sensitive.
