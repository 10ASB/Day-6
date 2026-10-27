"""Full Day 6 lab: KNN, Naive Bayes, SVM, tuning, plots, and optional SMS spam.

Install: python -m pip install numpy matplotlib scikit-learn
Run:     python day06_full_lab.py
SMS:     python day06_full_lab.py path/to/sms_spam.csv

For an SMS CSV, provide columns named 'label' and 'message'. Labels should
contain spam/ham (or 1/0). The vectorizer is fitted only on training messages.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))
    return model


def plot_decision_boundary(model, X, y, title, filename):
    """Plot a 2D decision boundary; model must accept two numeric features."""
    x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
    y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, zz, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="black", s=24)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved decision-boundary plot: {filename}")


def run_synthetic_model_comparison():
    # Non-linear toy data makes kernel choices easier to demonstrate.
    X, y = make_moons(n_samples=500, noise=0.24, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Fit scaling on training data only. Pipelines prevent leakage in CV.
    models = {
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "SVM (linear)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear", C=1.0)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]),
        "SVM (polynomial)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="poly", degree=3, C=1.0, gamma="scale")),
        ]),
    }
    fitted = {}
    for name, model in models.items():
        fitted[name] = evaluate(name, model, X_train, X_test, y_train, y_test)

    print("\nKNN tuning (5-fold CV)")
    knn_search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
        {"model__n_neighbors": [3, 5, 7, 9, 11, 15],
         "model__weights": ["uniform", "distance"]},
        cv=5, scoring="f1", n_jobs=-1,
    )
    knn_search.fit(X_train, y_train)
    print("Best KNN parameters:", knn_search.best_params_)
    print(f"Best CV F1: {knn_search.best_score_:.3f}")
    evaluate("Tuned KNN", knn_search.best_estimator_, X_train, X_test, y_train, y_test)

    print("\nSVM tuning (5-fold CV)")
    svm_search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
        {"model__kernel": ["linear", "rbf", "poly"],
         "model__C": [0.1, 1, 10],
         "model__gamma": ["scale", 0.1, 1]},
        cv=5, scoring="f1", n_jobs=-1,
    )
    svm_search.fit(X_train, y_train)
    print("Best SVM parameters:", svm_search.best_params_)
    print(f"Best CV F1: {svm_search.best_score_:.3f}")
    evaluate("Tuned SVM", svm_search.best_estimator_, X_train, X_test, y_train, y_test)

    # Plot each model's boundary on the same two-feature dataset.
    for name in ("KNN (k=5)", "SVM (linear)", "SVM (RBF)", "SVM (polynomial)"):
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_decision_boundary(
            fitted[name], X, y, name, f"decision_boundary_{safe_name}.png"
        )


def run_sms_spam(csv_path):
    # pandas is used only to load the optional external CSV.
    import pandas as pd

    data = pd.read_csv(csv_path)
    required = {"label", "message"}
    if not required.issubset(data.columns):
        raise ValueError("SMS CSV must have columns named 'label' and 'message'.")

    data = data.dropna(subset=["label", "message"]).copy()
    labels = data["label"].astype(str).str.strip().str.lower()
    mapping = {"ham": 0, "not spam": 0, "0": 0, "spam": 1, "1": 1}
    y = labels.map(mapping)
    valid = y.notna()
    messages = data.loc[valid, "message"].astype(str)
    y = y.loc[valid].astype(int)
    if y.nunique() < 2:
        raise ValueError("The SMS dataset must include both spam and ham labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        messages, y, test_size=0.2, random_state=42, stratify=y
    )
    # Pipeline ensures CountVectorizer is fitted on training text only.
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])
    evaluate("SMS spam — CountVectorizer + MultinomialNB",
             model, X_train, X_test, y_train, y_test)


def main():
    run_synthetic_model_comparison()
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_file():
            raise FileNotFoundError(f"SMS CSV not found: {path}")
        run_sms_spam(path)
    else:
        print("\nSMS spam task not run: no CSV supplied.")
        print("Pass a CSV with 'label' and 'message' columns to run it.")


if __name__ == "__main__":
    main()
