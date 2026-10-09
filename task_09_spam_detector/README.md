# Task 9 — Spam Detector (TF-IDF + SVM)

Classifies SMS messages as **spam** or **ham** with a **TF-IDF vectorizer + linear SVM** (scikit-learn).

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
cd task_09_spam_detector
jupyter notebook spam_detector.ipynb         # training, evaluation, sample predictions
python predict.py "Free entry!! Text WIN to claim your prize"   # use the saved model from the terminal
```

## Method

- Dataset: SMS Spam Collection (UCI), 5,572 messages (13% spam) → `data/sms_spam.tsv`.
- **Removed 403 exact duplicate texts before splitting**, so identical messages can't appear in both train and test and inflate the score.
- Stratified 80/20 split. TF-IDF with unigrams + bigrams, English stop-words removed, `sublinear_tf`, `min_df=2`.
- `LinearSVC` with `class_weight="balanced"` because spam is the minority class.

## Results (held-out test set)

| Metric | Value |
|---|---|
| Accuracy | **97.97%** |
| Spam precision | 93.65% |
| Spam recall | 90.08% |
| Spam F1 | **0.918** |

Accuracy alone is misleading here (always answering "ham" would score ~87%), so spam precision/recall/F1 are the headline numbers. The notebook also shows the confusion matrix, the actual misclassified messages, and the strongest spam/ham words the model learned (top spam indicators include "txt", "claim", "won", "mobile", "reply", "http", "150p"; top ham indicators include "ok", "lol", "hey", "later", "home").

## Sample predictions (new messages, all correct)

| Message | Prediction |
|---|---|
| WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 now to claim. | SPAM |
| Hey, are we still meeting for lunch at 1pm today? | ham |
| Congratulations! Free entry to win a brand new iPhone. Text WIN to 80082 now | SPAM |
| Can you send me the notes from yesterday's lecture? I missed the last 20 minutes | ham |
| URGENT! Your account has been suspended. Click http://bit.ly/verify to claim your free gift | SPAM |
| Mom said dinner will be ready by 8, don't be late | ham |

(8 samples in total are in `outputs/sample_predictions.csv`.)

## Limitations
Trained on short English SMS from around 2012 — it will not transfer directly to modern email phishing, other languages, or fake-news articles.

## Files
`spam_detector.ipynb`, `predict.py`, `model/spam_tfidf_svm.joblib`, `data/sms_spam.tsv`, `outputs/`
