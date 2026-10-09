import sys
import joblib

model = joblib.load("model/spam_tfidf_svm.joblib")

for text in sys.argv[1:]:
    score = model.decision_function([text])[0]
    print("SPAM" if score > 0 else "ham", "-", text)
