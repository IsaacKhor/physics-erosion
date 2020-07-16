# Paper

Method:
- Mathematical simulation of flow and erosion
    - Randomised initial volume sections
    - Evolving according to ---- equations
- Train different models on the simulated data with different parameters

Analysis:
- Model 1: simple models
    - Distribution of sum of left/right volume fractions
    - Clustering algorithms show no siginificant clustering
    - Clustering v time or channel progression (incomplete)
- Model 2: classical models, specifically a random forest or decision tree
    - Different extractions of feature
- Model 3: neural methods
    - CNN on independent images, GRU on time series
- All 3: quality of predictions v time / channel progression

Discussion:
- Compare the 3 approaches and many trained models
    - Metrics
    - Caveats around the metrics and under what conditions to interpret

# Outline

## Background

- Trying to analyse flow and erosion
- Do so with simulations in the lab

_picture of physical experiments_

- For ease of analysis, build a mathematical model of erosion
    - _overview of mathematics behind the model_
    - Start with randomised volume fractions over a grid
    - Parameters include flow rate, ramp rate, etc
    - Tweak where the entrance/exit points are for different behaviours

## Methods

### Manually extracting features

- Start with the simplest feature - can we determine by summing up volume fractions
  on left/right side of the system and determine if that's left or right?
    - Simplest parameters - low flow and ramp rate, only goes left/right
    - Generate set of `n=2000` simulations
    - For each simulation, take `t=0` image and divide into left and right half
    - Simply sum up the pixel intensities (and thus volume fraction) of each half
    - Results in 2-tuple of `(left-sum, right-sum)`
    - Plot the tuples on 2d graph, coloring points red/blue depending on if the
      stream went left or right
    - ![](../figs/step1-lr-distribution.png)
    - The same but for t=200 when system is done with evolving
    - ![](../figs/step-201-lr-distribution.png)
    - By visual inspection, we can see that there's no kind of clustering algorithm
      that will function well
    - Confirmed by training a naive bayes classifier, `accuracy=0.53`
- Next feature: split image into two half, and for each half, determine the furthest
  row where the channel has progressed
    - Scan each row for at least 2 pixels that fall below a `threshold=50`
    - Once again, tuple of `(left-row, right-row)`
    - ![](../figs/classical/highest_lr_last.png)
    - Everything is blue because it's overriding all the reds, manual inspection
      shows that they are there just being hidden
- Next feature: determine the "centre" of the stream
    - Calculate by $$ \frac{(\max(row) - row) * range(1,50)}{\sum row} $$
    - Get a 50-tuple specifying the "centre" of the stream at that row
    - Train a decision tree, but couldn't find a configuration that doesn't overfit
    - 50-tuple has a high degree of colinearity
    - ![](../figs/classical/pca.png)

### PCA

- Continueing from the dataset of "centre" of stream, instead of all 50 features,
  use PCA to reduce down to `n=7` features
    - Visualisation of PCA with `n=2` features
    - ![](../figs/classical/pca2_clustering.png)
    - 
