"""Day 6 starter: KNN and Gaussian Naive Bayes from scratch with NumPy.

This is educational code. The roadmap's full lab also asks for SVM, plots,
hyperparameter tuning, and an SMS-spam experiment; see day06_lab_guide.md.
"""
import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2))


class KNearestNeighbors:
    def __init__(self, k=5):
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        if self.X.ndim != 2 or len(self.X) != len(self.y):
            raise ValueError("X must be 2D and have the same number of rows as y.")
        if self.k > len(self.X):
            raise ValueError("k cannot exceed the number of training examples.")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = []
        for row in X:
            distances = np.sqrt(np.sum((self.X - row) ** 2, axis=1))
            nearest = np.argsort(distances)[: self.k]
            labels, counts = np.unique(self.y[nearest], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)


class GaussianNaiveBayes:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same number of rows as y.")

        self.classes = np.unique(y)
        self.priors = {}
        self.means = {}
        self.variances = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            self.priors[cls] = len(X_cls) / len(X)
            self.means[cls] = X_cls.mean(axis=0)
            self.variances[cls] = X_cls.var(axis=0) + 1e-9
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = []
        for row in X:
            log_scores = []
            for cls in self.classes:
                mean = self.means[cls]
                var = self.variances[cls]
                log_likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * var) + ((row - mean) ** 2) / var
                )
                log_scores.append(np.log(self.priors[cls]) + log_likelihood)
            predictions.append(self.classes[int(np.argmax(log_scores))])
        return np.asarray(predictions)


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Inputs must have equal, nonzero length.")
    return float(np.mean(y_true == y_pred))


def main():
    # Reproducible synthetic binary classification example.
    rng = np.random.default_rng(42)
    class0 = rng.normal(loc=(-1.5, -1.0), scale=0.8, size=(100, 2))
    class1 = rng.normal(loc=(1.5, 1.0), scale=0.8, size=(100, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * len(class0) + [1] * len(class1))

    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale using training-set statistics only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    for model in (KNearestNeighbors(k=5), GaussianNaiveBayes()):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"{model.__class__.__name__}: accuracy={accuracy(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
