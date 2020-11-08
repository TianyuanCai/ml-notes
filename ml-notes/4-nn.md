# Deep Learning

## Hyperparameters

### Batch Size

The number of training examples utilized in one iteration.

TL;DR – NN is sensitive to batch size. Tune it.

#### In convex optimization

A larger batch size allows computational speedups from the parallelism of GPUs. However, it is well known that too large of a batch size will lead to poor generalization (although currently it’s not known why this is so).

Smaller batch sizes have been empirically shown to have faster convergence to “good” solutions. This is intuitively explained by the fact that smaller batch sizes allow the model to “start learning before having to see all the data.” The downside of using a smaller batch size is that the model is not guaranteed to converge to the global optima. It will bounce around the global optima, staying outside some $\epsilon$-ball of the optima where $\epsilon$ depends on the ratio of the batch size to the dataset size.

Using a batch equal to the entire dataset guarantees convergence to the global optima of the objective function. However, this is at the cost of slower, empirical convergence to that optima.

Therefore, under no computational constraints, it is often advised that one starts at a small batch size, reaping the benefits of faster training dynamics, and steadily grows the batch size through training, also reaping the benefits of guaranteed convergence.

#### Neural Network

Neural Network is not always convex.

Because neural network systems are extremely prone to overfitting, the idea is that seeing many small batch sizes, each batch is a “noisy” representation of the entire dataset, will cause a sort of “tug-and-pull” dynamic. This “tug-and-pull” dynamic prevents the neural network from overfitting on the training set and hence performing badly on the test set.

## Autoencoders

Learn representation of input data by learning function $X=f(X)$.

The result hidden layer representation can be used for data compression, dimensionality reduction, and feature learning.

By stacking Autoencoders to learn the representation of data, and train it greedily, hopefully, we can train deep net effectively.

### Vanilla

[Overview](https://wiseodd.github.io/techblog/2016/12/03/autoencoders/)

### Variational

[Variational](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/)

A VAE is an autoencoder whose encodings distribution is **regularised** during the training in order to ensure that its latent space has good properties allowing us to generate some new data. Moreover, the term “variational” comes from the close relationship there is between the regularisation and the variational inference method in statistics.

- The input is encoded as a distribution over the latent space
- A point from the latent space is sampled from that distribution
- The sampled point is decoded and the reconstruction error can be computed
- The reconstruction error is backpropagated through the network
