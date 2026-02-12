I have some experiments to run on language models but before that I want to test something about language model behaviour.

Pick some very small language model from huggingface. Pick something that we can iterate on quickly.

Let `p(y_n | y_{1:n-1})` be the probability distribution assigned by the model to the next token given a history of tokens.

Step 1:
Start with `y_1 = some fixed token`. And generate the rollout `y_{2:N}` for some large N by concatenating the output token to the input one by one. That is, `y_2 \sim p( \cdot | y_1`, `y_3 \sim p( \cdot | y_{1:2})` and so on. I want to monitor, for each k, the histogram of tokens for `y_{1:k}`. In particular, I want to check if the frequency of each token stabilizes as k gets larger. I think we need to pick a language model that uses a small number of tokens if we want to be able to do this. So make your decision appropriately.

Step 2:
For the same language model as in step 1, I want to now monitor not the frequency of each token, but the frequency of transitions from one token to another. That is, for each pair a, b of tokens, I want to check the fraction of times that b follows a out of all the times that a occurs. I want to now check if this frequency stabilizes as k increases.

This means we are now checking `m^2` pairs where m is the number of tokens in the vocabulary. So once again, it's quite crucial to pick a model that has a very small number of tokens.

Step k:
I want to do this for as large a k as feasible. For example, fot the case of three tokens, we would check the fraction of times that ab is follwed by c for all sets (a,b,c). And I want to see if this stabilizes as we consider longer and longer rollouts.

