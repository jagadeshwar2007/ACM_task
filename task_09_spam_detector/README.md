# Task 9: Spam detector (TF-IDF + SVM)

Classifies SMS messages as spam or ham using a TF-IDF vectorizer and a linear SVM.

## How to run

```
pip install pandas numpy scikit-learn matplotlib joblib jupyter
cd task_09_spam_detector
jupyter notebook spam_detector.ipynb
```

To try the saved model on your own messages:

```
python predict.py "Free entry win a prize now" "are you coming to class tomorrow"
```

## What I did

- Dataset: SMS Spam Collection (5,572 messages, about 13% spam), in `data/sms_spam.tsv`
- Removed 403 duplicate messages before splitting so the same message is not in both train and test
- TF-IDF (words and word pairs, english stop words removed) + LinearSVC with balanced class weights
- 80/20 train test split

## Results (test set)

- Accuracy: 97.97%
- Spam precision: 0.94, recall: 0.90, F1: 0.92

Accuracy alone is not enough here because most messages are ham, so I looked at spam precision and recall too.

## Sample predictions

| Message | Prediction |
|---|---|
| WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 now to claim. | spam |
| Hey, are we still meeting for lunch at 1pm today? | ham |
| Congratulations! Free entry to win a brand new iPhone. Text WIN to 80082 now | spam |
| Can you send me the notes from yesterday's lecture? I missed the last 20 minutes | ham |
| URGENT! Your account has been suspended. Click http://bit.ly/verify to claim your free gift | spam |
| Mom said dinner will be ready by 8, don't be late | ham |

All of these were predicted correctly. The notebook has 8 samples in total.

The model is trained on short english SMS, so it will not work well on emails or fake news without retraining.
