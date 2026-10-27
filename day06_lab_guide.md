# Day 6 Lab Guide — KNN, Naive Bayes, and SVM

## Part A — Compare the model families
Use the same dataset and the same train/test split for each model. Record:
- Dataset and target
- Preprocessing applied
- Train and test scores
- Confusion matrix and useful classification metrics
- Observations about errors and limitations

## Part B — K-Nearest Neighbors
1. Try several values of (k), for example 1, 3, 5, 7, and 11.
2. Compare Euclidean and Manhattan distance if available in your chosen implementation.
3. Standardize features using training data only, then apply the same transformation to test data.
4. Explain how (k) changes the bias/variance balance and why distance becomes less informative in high dimensions.

## Part C — Naive Bayes
Study the conditional-independence assumption and distinguish:
- Gaussian Naive Bayes: continuous features
- Multinomial Naive Bayes: count/frequency features such as word counts
- Bernoulli Naive Bayes: binary features

### SMS spam extension
1. Load a labeled SMS spam dataset.
2. Split into train/test before fitting any vectorizer.
3. Fit CountVectorizer on training messages only; transform train and test messages.
4. Train Multinomial Naive Bayes and report precision, recall, F1, and confusion matrix.
5. Discuss false positives and false negatives.

## Part D — Support Vector Machines
Using an SVM implementation such as scikit-learn for this comparison:
1. Compare linear, RBF, and polynomial kernels.
2. Tune (C) and, for the RBF kernel, (gamma).
3. Plot decision boundaries on a two-feature dataset.
4. Explain the role of support vectors, margin, and the kernel trick.

## Comparison table

| Model | Preprocessing | Hyperparameters | Train score | Test score | Notes |
|---|---|---|---:|---:|---|
| KNN | | k / distance | | | |
| Naive Bayes | | variant | | | |
| SVM | | kernel / C / gamma | | | |

## Checklist
- [ ] Compare all three models on the same split.
- [ ] Plot decision boundaries using two features.
- [ ] Tune k, C, and gamma and explain the observed changes.
- [ ] Run the SMS-spam Naive Bayes experiment with CountVectorizer.
- [ ] Commit the comparison table and written conclusions.
