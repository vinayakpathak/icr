An LPE experiment for k-Markov or Bernoulli (which really is k-Markov for k=0) should create a report that must include the following:
1. Details of the model architecture
2. Details of the training, such as optimizer, step size, number of warmup steps.
3. Training curve
4. The loss achieved by the final model and its comparison with the loss of the optimal Bayes predictive
5. Statistics and graphs demonstrating the quality of the posterior samples obtained from the posterior sampling approach used in the experiment. For example, the KL-divergence between the true posterior vs the empirical posterior distribution implied by the samples. The difference between the true posterior and the sampled posterior.
6. The quality of the LPE estimation. Relative error between the estimated probability and the true probability. Also, an estimate of the relative error we would expect from a naive monte carlo approach if we spent the same amount of compute on it.
7. Aything else that you think is important for properly evaluating the quality of this estimation method. 