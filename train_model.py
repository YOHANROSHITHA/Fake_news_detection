"""Train fake/real news classifier with balanced logistic regression."""

import pickle
import re
import string
from typing import Tuple

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip urls/numbers/punctuation, remove stopwords."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def load_and_prepare(fake_path: str = "Fake.csv", true_path: str = "True.csv") -> pd.DataFrame:
    """Read datasets, tag labels, shuffle, and clean text."""
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true], ignore_index=True)
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    data = data.dropna(subset=["text"])
    data["clean_text"] = data["text"].apply(clean_text)
    return data


def vectorize(data: pd.DataFrame) -> Tuple[TfidfVectorizer, any, any]:
    """Fit TF-IDF (unigrams+bigrams) and return vectorizer and features."""
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(data["clean_text"])
    y = data["label"].values
    return vectorizer, X, y


def train_models(X_train, y_train):
    """Train calibrated Linear SVM (balanced) and Multinomial NB."""
    svm_base = LinearSVC(class_weight="balanced")
    svm_model = CalibratedClassifierCV(svm_base, method="sigmoid", cv=5)
    svm_model.fit(X_train, y_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    return svm_model, nb_model


def find_optimal_threshold(model, X_valid, y_valid, positive_label: int = 1) -> float:
    """Pick probability threshold that maximizes F1 for the positive label."""
    if not hasattr(model, "predict_proba"):
        return 0.5

    probs = model.predict_proba(X_valid)
    class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}
    pos_idx = class_indices.get(positive_label, 1 if len(model.classes_) > 1 else 0)
    pos_scores = probs[:, pos_idx]

    precision, recall, thresholds = precision_recall_curve(
        y_valid, pos_scores, pos_label=positive_label
    )
    f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.argmax(f1_scores))

    # precision_recall_curve returns one extra precision/recall than thresholds
    if best_idx >= len(thresholds):
        return 0.5

    return float(thresholds[best_idx])


def evaluate(model, X_test, y_test, name: str) -> float:
    """Print accuracy, report, and return macro F1."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"\n{name} Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_test, preds))
    return f1


def main() -> None:
    data = load_and_prepare()
    vectorizer, X, y = vectorize(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    svm_model, nb_model = train_models(X_train, y_train)

    f1_svm = evaluate(svm_model, X_test, y_test, "Calibrated Linear SVM")
    f1_nb = evaluate(nb_model, X_test, y_test, "Multinomial NB")

    best_model, best_name = (svm_model, "svm") if f1_svm >= f1_nb else (nb_model, "nb")

    best_threshold = find_optimal_threshold(best_model, X_test, y_test, positive_label=1)
    setattr(best_model, "best_threshold_", best_threshold)
    print(f"Selected threshold for REAL news: {best_threshold:.3f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nSaved best model: {best_name} to model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    main()