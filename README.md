# Day 6 — Distance & Margin-Based Models: KNN, Naive Bayes & SVM

**AI/ML Internship · Week 2 · Day 6**

This repository records the actual Day 6 exercises: training and comparing KNN, Gaussian Naive Bayes and an RBF SVM; plotting decision boundaries; tuning KNN and SVM hyperparameters; and applying Multinomial Naive Bayes to SMS spam messages.

## Tasks completed in the supplied work

### Task 1 — Train models and plot decision boundaries
- Generate a two-feature classification dataset with `make_classification`.
- Split it into training and test sets.
- Train KNN (`k=5`), Gaussian Naive Bayes and an RBF-kernel SVM.
- Compare test accuracy and visualize the three decision boundaries.

### Task 2 — Hyperparameter tuning
- Use five-fold `GridSearchCV` to compare KNN values `k = 1, 3, 5, 7, 9, 11`.
- Tune the RBF SVM over `C = 0.1, 1, 10, 100` and `gamma = 0.01, 0.1, 1, 10`.
- Print the best parameters, best cross-validation score and per-setting results.

### Task 3 — SMS spam classification
- Load `spam.csv`, map `ham` to 0 and `spam` to 1.
- Split messages into training and test sets.
- Fit `CountVectorizer` on training messages only and transform the test messages.
- Train `MultinomialNB` and report accuracy and classification metrics.

### Task 4 — Evaluation and prediction experiments
- Inspect a confusion matrix.
- Test the SMS classifier on example messages.
- Re-run the KNN and SVM tuning exercises.

## Files

| File | Purpose |
|---|---|
| `testing.py` | Model training, accuracy evaluation and decision-boundary visualization |
| `testing2.py` | KNN and SVM hyperparameter tuning |
| `testing3.py` | SMS spam classification with CountVectorizer and Multinomial Naive Bayes |
| `task1.ipynb` | Main notebook: model training, metrics, decision boundaries, tuning and SMS task |
| `task2.ipynb` | Follow-up notebook: confusion matrix, sample-message predictions and tuning exercises |
| `spam.csv` | Required input dataset; currently not committed to this repository, so add your supplied CSV locally before running the SMS task |

## Setup

Python 3.10+ recommended.

```bash
python -m pip install numpy pandas matplotlib scikit-learn seaborn jupyter
```

Run the scripts from the directory containing `spam.csv`:

```bash
python testing.py
python testing2.py
python testing3.py
jupyter notebook
```

Open `task1.ipynb` and `task2.ipynb` in Jupyter or Google Colab to run the notebook cells in order.

## Notes

- `testing3.py` expects `spam.csv` in the current working directory. It reads using `encoding="latin-1"` and uses the first two columns as label and message. The dataset file itself is not currently in this GitHub repository.
- The notebook SMS sections need `spam.csv` to be available in the notebook working directory. Upload the dataset to Jupyter/Colab or place it beside the scripts before running the SMS task.
- The synthetic dataset is for learning and plotting; its scores are not evidence of real-world performance.
- For rigorous model evaluation, use a stratified split and fit preprocessing only on training data. The current scripts preserve the original exercise flow and can be improved in a later revision.
