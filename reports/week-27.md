# Week 27 - classical algorithms cont.

Problems:
- Suspected overfitting from decision tree (since they are known to have this problem)
- Data set is highly correlated, so some sort of dimensionality reduction (probably PCA)
- Analyse how and why classical algorithms can perform well

- Train random forest model on extracted "average dark" feature with NO DM
    - Heavy overfitting, acc=0.98
    - No idea why
    - Tuning hyperparams, min_samples_leaf=200, min_samples_split=100, max_samples=0.66
      max_depth=7, max_features=0.2
    - Spent way too much time before I forgot that I was operating on data without dim reduction
- Do dimensionality reduction

![](../figs/classical/pca.png)

- Only 7 components explain most of the variance present in the set of features
- Train random forest on PCA with n=7
    - Still 0.99 accuracy
    - Highly suspicious but also interesting?
- Try something else
    - What if we get just n=1 from PCA?
    - Correlation between (pca_data, labels) = -0.733
- What about a OLS model?
    - PCA with n=7, R^2 = 0.628

To do:
- Get weights of PCA
- Plot PCA v labels