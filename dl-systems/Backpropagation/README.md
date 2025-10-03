# Excercises

### 1. Backpropagation with a single modified neuron

Suppose we modify a single neuron in a feedforward network so that the output from the neuron is given by $$f\left(\sum_j w_j x_j + b\right)$$, where $$f$$ is some function other than the sigmoid. How should we modify the backpropagation algorithm in this case?

- **Implementation Note:** The conceptual changes for this exercise will be demonstrated within the backprop method of `backprop_algo.py`.
  
### 2. Backpropagation with linear neurons

Suppose we replace the usual non-linear $$\sigma$$ function with $$\sigma(z) = z$$ throughout the network. Rewrite the backpropagation algorithm for this case.

- **Implementation Note:** The conceptual changes for this exercise will be demonstrated within the backprop method of `backprop_algo.py`.

# Problem

### 1. Fully matrix-based approach to backpropagation over a mini-batch

Optimize the backpropagation algorithm to process an entire mini-batch of training examples at once, rather than iterating through them individually.  

The idea is that instead of beginning with a single input vector, $$x$$, we can begin with a matrix $$X = [x_1\ x_2\ \ldots\ x_m]$$ whose columns are the vectors in the mini-batch. We forward-propagate by multiplying by the weight matrices, adding a suitable matrix for the bias terms, and applying the sigmoid function everywhere.

The advantage of this approach is that it takes full advantage of modern libraries for linear algebra. As a result it can be quite a bit faster than looping over the mini-batch. In practice, all serious libraries for backpropagation use this fully matrix-based approach or some variant.

- **Solution File:** `vectorized_backprop.py`

# References

- [Neural Networks and Deep Learning by Michael Nielsen: Chapter 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- [3Blue1Brown's "Neural Networks" Series: Videos 3 & 4](https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PLZHQObOWTQDNU6bTj_5S_L3Bw5Y7P_pG)

