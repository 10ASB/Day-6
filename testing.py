from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Create a dataset with two features so decision boundaries can be visualized.
X, y = make_classification(
    n_samples=150,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=69,
)

# Split into training and test data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train all three models on the same training split.
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel="rbf", C=1, gamma="scale"),
}
for model in models.values():
    model.fit(X_train, y_train)

print("Test-set accuracy")
for name, model in models.items():
    prediction = model.predict(X_test)
    print(f"{name}: {accuracy_score(y_test, prediction):.3f}")

# Draw a decision-boundary panel for each classifier.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    boundary = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, boundary, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
    ax.set_title(name)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
plt.tight_layout()
plt.show()
