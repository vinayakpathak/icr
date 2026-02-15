I want to continue the experiments in markov_transformer.py.

This is based on elicitation_lpe.tex, and also similar to the experiment in bernoulli_transformer.py.

Step 1:
Train a model that can do in-context Bayes inference on order-k Markov processes on a binary alphabet.

An order-k Markov process is a process that generates a distribution over binary sequences. The process specifies for each k-bit binary string, the probability that the next bit is 0. Thus it is parametrized by 2^k probability values. 

Imagine a distribution over order-k Markov processes. So this is a distribution over 2^k probability values. This distribution is going to be the prior. I'm not sure what the best distribution is. Maybe we start with a Beta(1,1) distribution on each of those 2^k values. But I am open to other more appropriate distributions.

Given this prior, we can imagine a Bayesian learner that sees a sequence y_{1:n} and predicts y_{n+1}. This learner updates its prior over the 2^k values to an appropriate posterior, and then its prediction of y_{n+1} is the Bayes predictive, i.e., the expected value of the prediction over the posterior.

Let's call this model the ground truth.

Next, train a transformer model that can replicate the ground truth as best as possible. The way to train this is the following: sample a 2^k-dimensional vector of probabilities from the prior you have chosen, then sample several sequences from the Markov model parametrized by the sampled vector, and then do a gradient step using autoregressive training on the transformer. 

At the end of the training, the loss of the transformer should be almost the same (within 2-3%) as the loss of the ground truth. If this isn't the case then train a bigger transformer.

Find a small-ish transformer that, at the end of trainig, gets close to the ground truth in terms of loss. Plot graphs comparing the transformer against the true Bayesian ground truth.

Make sure to upload the final transformer checkpoint to R2.

One question is what k to use? I am interested in k = 1 to 7 for now. I want to train one transformer for each k. So there will be 7 transformers in total.

Step 2:

Now implement the posterior sampling method, so that given a sequence s, we can draw samples from the transformer's inner posterior over its 2^k dimension vectors. The way to do it is to generate a large rollout starting from s as the initial input, and then count the frequency with which each of the transitions happen in the rollout. Thus each rollout gives a 2^k dimensional probability vector, which is the sample from the posterior. Thus generating a good number of rollouts we get a good number of samples. Let's say number of rollouts to use is 200. Length of each rollout is trickier. For now, let's go with 100*2^k.

It is important that the transformers you train do not use positional encoding btw. It is also important that the length of the input they can accept is at least as big as the length of the rollouts we want to generate. I don't want the rollout generation process to use sliding windows as input. 

Compare the posterior samples thus generated against the true posterior of an ideal Bayes predictor. These two should match very closely. Plot graphs.

Step 3:

Now it's time to do some LPE calculations. 

Generate a few random strings s_1, ... s_l of varying lengths ranging between say 10-20. For each s_i, feed it to the trained transformers. Then, use a fixed string s^* of length 100. We are going to answer the question: what is the probability that the transformer's rollout starting from s_i is exactly s. The way we will do that is via the posterior samples. Given each k-dimensional vector \tilde{p}, we can calculate in closed form the probability of the rollout being s. Let's call this number \tilde{p}(s) (abusing notation). Thus the probability of the transformer outputting s (without conditioning on \tilde{p}) will be approximated by the empirical mean of \tilde{p}(s) over the posterior samples. 

Report this number for each transformer and for each s_1, ... , s_l. Also calculate the true probability of seeing s from the true Bayes predictive. Compare the two. These two should be very close to each other. 

Step 4:

Write a latex report convincing a sceptical reader that you have done the correct thing. Include all the plots and numbers that you think add the most amount of evidence towards your correctness.

Compile the latex and generate the pdf.