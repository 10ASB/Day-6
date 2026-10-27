# Day 6 — Distance & Margin Based Models: KNN, Naive Bayes, SVM

**AI/ML Internship · Week 2 · Day 6**

This repository follows the Day 6 section of the 4-week AI/ML internship roadmap.

## Roadmap description

### Theory
- **K-Nearest Neighbors (KNN):** distance metrics, choosing (k), the curse of dimensionality, and why feature scaling is important.
- **Naive Bayes:** Bayes' theorem, the conditional-independence assumption, and Gaussian, Multinomial, and Bernoulli variants; common applications in text classification.
- **Support Vector Machines (SVM):** maximum-margin classification, support vectors, soft margin (C), and the kernel trick (linear, RBF, and polynomial kernels), including gamma.

### Hands-on lab
- Train KNN, Naive Bayes, and SVM on the same dataset and compare results.
- Plot decision boundaries using two features where applicable.
- Tune KNN's (k), and SVM's (C) and (gamma); document the effect of each setting.
- Apply Naive Bayes to an SMS spam dataset with CountVectorizer.

### Expected deliverable
- This repository's Python exercises and a comparison table documenting model performance and parameter effects. The roadmap names the notebook deliverable `day06_knn_nb_svm.ipynb`.

## Files
- `day06_models.py` — educational NumPy implementations of KNN and Gaussian Naive Bayes, plus an example comparison on a synthetic dataset.
- `day06_lab_guide.md` — lab tasks, tuning checklist, and SMS-spam instructions.\n- `day06_full_lab.py` — KNN/Naive Bayes/SVM comparison, KNN and SVM cross-validation tuning, decision-boundary plots, and optional SMS spam using CountVectorizer.

## Run
Install NumPy:
```bash
python -m pip install numpy
python day06_models.py
```

The NumPy starter code teaches the mechanics. The full lab script adds scikit-learn model comparisons, tuning, plots, and an optional SMS spam run. The SMS task requires a CSV with `label` and `message` columns; it is not run unless you supply that dataset.

## Completion checklist
- [ ] Train and compare KNN, Naive Bayes, and SVM on one dataset.
- [ ] Plot decision boundaries on two features.
- [ ] Tune (k), (C), and (gamma), recording their effects.
- [ ] Run Naive Bayes on SMS spam using CountVectorizer.
- [ ] Add a comparison table and written observations.
