# ACM AI/ML recruitment tasks

My submission for the ACM Student Chapter AI & ML domain recruitment. I did 5 tasks, each in its own folder with a README.

- `task_04_ml_model_api` - FastAPI service that serves the housing model
- `task_05_interactive_dashboard` - Streamlit app to predict house prices
- `task_06_housing_price_predictor` - housing price prediction with Random Forest
- `task_09_spam_detector` - spam detection with TF-IDF + SVM
- `task_12_mnist_digit_classifier` - CNN for MNIST in PyTorch (99.18% test accuracy)

Task 4 and 5 use the model saved by task 6 (`task_06_housing_price_predictor/model/housing_model.joblib`).

## Setup

```
pip install -r requirements.txt
```

Python 3.10 or higher. scikit-learn is pinned to 1.9.1 because the saved model files only load with the same version.
