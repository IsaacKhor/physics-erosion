# Week 25: naive RNN and new simulation data

- Obtained new simulation code, optimised it and ran it with different parameters
- Due to problems with overfitting, implemented
    - Increase dropout to 0.45
    - Remove a convolution layer
    - Increase normalisation
    - Early stopping when accuracy scores dont improve much
- Trained another CNN on all the new data
- Trained a simple naive RNN with input data
    - Just feed time steps 25-75 into RNN and try to get prediction data
    - Problems with overfitting and other unknown stuff causing failures
- Thought more deeply about archinecture, try
    - Separate convolutions for each input into a GRU for time step 25-50
    - Still separate convolutions, but GRU with sliding window of 5 images

Explore next week:
- Train classical models and explore their effectiveness
- Train on specific simulation parameters
