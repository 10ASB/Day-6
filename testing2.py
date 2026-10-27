from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=150,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=69,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# KNN tuning with five-fold cross-validation.
knn_params = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
knn_grid = GridSearchCV(
    KNeighborsClassifier(), knn_params, cv=5, scoring="accuracy"
)
knn_grid.fit(X_train, y_train)
print("Best k:", knn_grid.best_params_)
print("Best KNN CV score:", round(knn_grid.best_score_, 4))
print("\nKNN results:")
for k, score in zip(
    knn_grid.cv_results_["param_n_neighbors"],
    knn_grid.cv_results_["mean_test_score"],
):
    print("k =", k, "->", round(score, 4))

# RBF SVM tuning over C and gamma.
svm_params = {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]}
svm_grid = GridSearchCV(
    SVC(kernel="rbf"), svm_params, cv=5, scoring="accuracy"
)
svm_grid.fit(X_train, y_train)
print("\nBest SVM parameters:", svm_grid.best_params_)
print("Best SVM CV score:", round(svm_grid.best_score_, 4))
print("\nSVM results:")
for params, score in zip(
    svm_grid.cv_results_["params"],
    svm_grid.cv_results_["mean_test_score"],
):
    print(params, "->", round(score, 4))
