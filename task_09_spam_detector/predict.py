"""Classify messages with the trained TF-IDF + SVM model.

Usage:
    python predict.py "Free entry!! Text WIN to claim your prize"
    python predict.py            # interactive mode, type messages, empty line to quit
"""
import sys
from pathlib import Path

import joblib

model = joblib.load(Path(__file__).parent / "model" / "spam_tfidf_svm.joblib")


def classify(text: str):
    score = float(model.decision_function([text])[0])
    return ("SPAM" if score > 0 else "ham"), score


if __name__ == "__main__":
    msgs = sys.argv[1:]
    if msgs:
        for m in msgs:
            label, score = classify(m)
            print(f"{label:4s} (score {score:+.2f})  {m}")
    else:
        while True:
            m = input("message> ").strip()
            if not m:
                break
            label, score = classify(m)
            print(f"  -> {label} (score {score:+.2f})")
