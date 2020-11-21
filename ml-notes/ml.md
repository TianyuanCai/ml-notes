---
title: Machine Learning
weight: 3
draft: false
---
<!-- Just need a katex comment to generate katex -->

{< katex >} {< /katex >}

In an effort to not to search the same thing twice, I tried to jot down my learning and file them to the right folder whenever I have a question about machine learning. The result is this cheatsheet.

The content and explainability of this document will grow as I learn over time. 

What is an interesting way to learn ML?

# Machine Learning

## Learning Theory

Much of the notes in here come from my learning in Prof. Yaron Singer’s CS183 Foundations of Machine Learning course at Harvard and Prof. Shai Ben-David’s online course based on his book “Understanding Machine Learning: From Theory to Algorithms” ([PDF][1])

### PAC Learnability

General idea is to upper-bound the probability of observing an unusual training set. Only when the training set does not generalize well to the test set do we get an error greater than $epsilon$.

For a set of $m$ examples, a surprising training set appears when we did not observe any of the $epsilon$ possible surprising observations. The chance of that happening is $(1-epsilon)^m$.

PAC in Realizable Setting

$P[L\_D(h)\>epsilon]\<delta$

Proof of Agnostic PAC with Hoeffding Bound

$P[|L\_D(h)-L\_S(h)|\>epsilon]\<delta$

Interchangeable Expressions of PAC Learnability

$P[L\_D(A(S))\>epsilon]\<delta$

$E\_{mrightarrowinfty}[L\_D(A(S))]=0$

Notation $m\_H(frac18, frac18)$ means the minimum number of samples needed to learn with an error rate $frac18$ with a probability of failure of $frac18$.

Relationships

Learnable — there exists a learning algorithm for it, such as ERM.

- PAC Learnable
  - Threshold function
  - Agnostic PAC Learnable
	- Every finite hypothesis is agnostic PAC learnable?
  - Learnable by ERM
- H of all function over an infinite domain, not learnable (No free lunch theorem)

### Bayes Optimal Predictor

See bayesian statistics

### VC dimension

Vapnik–Chervonenkis

The VC dimension is a measure of the ability of a (binary) classifier to separate labels. It is mainly important as indicating when we are at risk of overfitting (by choosing _too complex_ a classification model with a high VC dimension), and for estimating the accuracy (generalization error) of a classifier from the training error.

A hypothesis class can shatter a set of n points if it contains a function that can generate all $2^n$ possible binary labelings of those points correctly (i.e. if it can classify every point in the set correctly, no matter how we label them).

The VC dimension of a hypothesized class is defined to be the largest n such that we can find at least one set of n points it shatters, but no set of n+1 points that it shatters. This is how we can prove the VC Dimension of a hypothesis class $mathcal{H}$.

If you have n data points, you have 2^n possible labelings. For each of these labelings, if you can draw a function from your function family that separates the labels, then the set of n points is said to have been shattered by your family of functions. The maximum n which you can shatter is the VC dimension, h, of your function family.

Thus, the VC dimension gives you a measure of confidence for the "separating capability" of the function family by looking at how many points you can actually separate knowing nothing about the distribution of labels.

There is a catch here though - a function with a VC dimension of h, in general, will not be able to shatter all possible sets of h points; all that is guaranteed is that there is some set of h points that can be shattered.

So if you want to keep your Risk at bay and you have a low empirical risk you do not want a high VC dimension. For two function families, both of which give you the same empirical risk, you would want to pick the one with the lower VC dimension. A high VC dimension, in this case, tells you the function may be too flexible and the empirical risk might have come at the cost of overfitting the data.

Thus, if you have another function family with a lower VC Dimension, and you have managed to obtain the same empirical risk with it, why risk overfitting?

VC Dim of Common Hypothesis Classes

Circles in $R^2$ - 3

Given unit vector $mathbf{e}$ in each dimension, then the circle centered at $sum\_{f(mathbf{e\_i})=+1}{mathbf{e}}$ can correctly classify all unit vector and the origin.

- If the origin has negative label, set the length of the circle to be less than 1
- Otherwise, equal to $sqrt{sum\_{f(mathbf{e\_i})=+1}{mathbf{e^2}}}$.

We can also achieve this proof by giving an example of three non-collinear points and discuss the three possible configurations of labels, one, two, or three positive labels.

Now, we also need to prove that the hypothesis class cannot correctly classify 4 points. There are three possible ways for the data to be laid out — When the convex hull of the four points

- form a quadrilateral – would misclassify one of the negative points when two diagonal points are positive.
- form a triangle – would misclassify the point within the convex hull when that’s the only negative point.
- form a line — would misclassify if every other point on the line is positive.

The idea of Convex Hull can be used to discuss the shape formed by hypothesis class.

Triangle – 7

Halfspace in $R^d$ – d + 1

$text{sign}(mathbf{w}^topmathbf{x}+b)Leftrightarrowmathbf{w}^topmathbf{x}+b\>0$

For instance, in $R^2$, we only need a function $wx+b\>0$ to test for labels.

Infinite VCdim means not PAC learnable

Assume b.w.o.c that such H is PAC Learnable, then for some $m$, we have $P[L\_D(A(S))\>epsilon]\<delta$. Consider $m\_H(frac{1}{8}, frac{1}{8})$, we should be able to find an $m$ to achieve $P[L\_D(A(S))\>epsilon]\<delta$.

By $text{VCdim}(H)=infty$, there exists some $wsubset X$ s.t. H shatters W and $|W|\>2m\_H(frac{1}{8}, frac{1}{8})$. There is no upperbound to the size of the set we can shatter.

Therefore, we have a shattered set that’s bigger than the size $m$ needed to be PAC learnable. If $w$ can be shattered, H induces every possible functions from $W$ to {0, 1}. But by NFL theorem, in such a condition, $m\_H(frac{1}{8}, frac{1}{8})geq |w|/2geq m\_H(frac{1}{8}, frac{1}{8})$. Contradiction.

If the VC dimension is large, we need a large sample size to learn. VC dimension measures the capacity for classifying all possible behaviors in a sample space. NFL then quantifies the difficulty with learning all possible behaviors.

### [No-Free-Lunch Theorem][2]

For a set of $m$ examples, and let H be **all** functions from X to ${0, 1}$. $m\_H(frac{1}{8}, frac{1}{8})geqfrac{|X|}{2}$. No algorithm can successfully learn H for the given error rate without seeing at least half of all data. This is almost just memorization, and it’s impossible for H with an infinite VC dimension to learn it.

- There exists a hypothesis with 0 loss.
- $P[L\_D(A(S))\>frac{1}{8}]\>frac{1}{7}$

**Intuition:** In a sample with $2^m$ possible combinations of labels, There may be a good hypothesis class with 0 loss, but on average, most of the hypothesis class will perform poorly. Thus, no hypothesis class can achieve good performance on all samples.

Proof

- Proof Idea –\* We want to show that on average, every learner is going to fail. We do that by estimating the probability that any given learner A will err on a random point.

- The probability that a random point X is not covered by S is 0.5
- The probability that a given $hin H$ predicts correctly is 0.5

Thus, if we have all the possible functions, the average performance on the data is $frac{1}{4}$.

### Fundamental Theorem of Statistical Learning

For every domain X and every class H of functions from X to ${0, 1}$, the following statements are equivalent

- H has the uniform convergence property (for every pair of $epsilon, delta$, there is a $m(epsilon, delta)$ that's guaranteed to be $epsilon$-representative)
- ERM is a successful agnostic PAC learner for H
  (Proof of 1 to 2 goes back to proving agnostic PAC using UC property)
- H is agnostic PAC learnable
  (learnable means there exists a successful learning algorithm, and we know from above that ERM is a successful learner. Thus, the proof is trivial.)
- ERM is a successful PAC learner for H
  (2 goes here – it’s harder to agnostic learn because we no longer have the realizable assumption)
- H is PAC learnable
- _VCdim(H) is finite_

All these functions are tied to _VCdim(H) is finite_.

### Optimization Methods

#### Convex Function

**Convex Set:** A convex set is a set of points such that, given any two points A, B in that set, the line AB joining them lies entirely within that set.

#### Gradient Descent

The case for Perceptron

Note that the learning rate doesn’t matter for Perceptron. Prove by induction.

#### Stochastic Gradient Descent

for i in range(iterations\_count):

param\_gradients = evaluate\_gradients(loss\_function, data, params)

params -= learning\_rate \* param\_gradients

#### Stochastic gradient descent with momentum

moment = 0

for i in range(iterations\_count):

param\_gradients = evaluate\_gradients(loss\_function, data, params)

```python
moment = gamma * moment + param_gradients

params += learning_rate * moment
```

(where the moment is building a moving average of gradients. Gamma gives friction = 0.9 or 0.99)

#### AdaGrad Optimization

```python
squared_gradients = 0
for i in range(iterations_count):
    param_gradients = evaluate_gradients(loss_function, data, params)
        squared_gradients += param_gradients ^ 2
        params -= learning_rate * param_gradients/(np.sqrt(squared_gradients) + 1e-8)

# (1e-8 is to avoid divide by zero)
```

For parameters with high gradient values, the squared term will be large, and hence dividing with a large term would make the gradient accelerate slowly in that direction. Similarly, parameters with low gradients will produce smaller squared terms and hence gradient will accelerate faster in that direction.

As gradient is squared at every step, the moving estimate will grow monotonically over time, and hence the step size our algorithm will take to converge to a minimum would get smaller and smaller.

#### RMSProp Optimization

```python
squared_gradients = 0

for i in range(iterations_count):
    param_gradients = evaluate_gradients(loss_function, data, params)
    squared_gradients = decay_rate *squared_gradients + (1 - decay_rate)* param_gradients ^ 2
    params -= learning_rate * param_gradients/(np.sqrt(squared_gradients) + 1e-8)
```

Similar to AdaGrad, here as well, we will keep the estimate of the squared gradient but instead of letting that squared estimate accumulate overtraining we rather let that estimate decay gradually. To accomplish this, we multiply the current estimate of squared gradients with the decay rate.

#### Adam

```python
first_moment = 0
second_moment = 0

for step in range(iterations_count):
    param_gradients = evaluate_gradients(loss_function, data, params)

    first_moment = beta_1 * first_moment + (1 - beta_1) * param_gradients
    second_moment = beta_2 * second_moment + (1 - beta_2) * param_gradients * param_gradients

        first_bias_correction = first_moment/(1 - beta_1  step)

        second_bias_correction = second_moment/(1 - beta_2  step)

        params -= learning_rate * first_bias_correction/(np.sqrt(second_bias_correction) + 1e-8)
```

This incorporates all the nice features of RMSProp and Gradient descent with momentum.

Specifically, this algorithm calculates an exponential moving average of gradients and the squared gradients whereas parameters $beta\_1$ and $beta\_2$ control the decay rates of these moving averages.

Notice that we’ve initialized second\_moment to zero. So, in the beginning, second\_moment would be calculated as somewhere very close to zero. Consequently, we are updating parameters by dividing with a very small number and hence making large updates to the parameter. That means initially, the algorithm would make larger steps. To rectify that we create an unbiased estimate of those first and second moment by incorporating the current step. And then we make updates to parameters based on these unbiased estimates rather than first and second moments.

#### Summary

Adding momentum provides a boost to accuracy. In practice, however, Adam is known to perform very well with large data sets and complex features.

#### Resources on Convex Optimization

Coursera on Optimization [link][3]

[cs229 Notes][4]

## Exploratory Data Analysis

Tools
I start with the non-interactive ones in R, first `summary tools` followed by `dataMaid`. I then play around with `explore`, and finally, I spend the most time with `pandas-profiling` in Python.

`pandas-profiling` is probably the most popular package for automatic EDA in the python world. You can use it from within a notebook and thanks to the fantastic ipywidgets library, you get an interactive and very rich visualization to explore your data.

## Preprocessing

### Scale

- Scale, standardize, normalize
  - For instance, many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) assume that all features are centered around zero and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
  - TL;DR
	- Use MinMaxScaler as your default
	- Use RobustScaler if you have outliers and can handle a larger range
	- Use StandardScaler if you need normalized features
	- Use Normalizer sparingly - it normalizes rows, not columns
- Normalization
  - The L1 norm is calculated as the sum of the absolute values of the vector.
  - The L2 norm is calculated as the square root of the sum of the squared vector values.
  - The max norm that is calculated as the maximum vector values.
  - Normalization is the process of scaling individual samples to have a unit norm. This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
  - This assumption is the base of the [Vector Space Model][5] often used in text classification and clustering contexts.
  - The function [normalize][6] provides a quick and easy way to perform this operation on a single array-like dataset, either using the l1 or l2 norms
- Resources
  - [Use of various Python scalers and their effects][7]
  - [Effects on different models][8]

### Remove Outliers

Isolation Forest

### Missing Data

kNN

[Part 1][9]

[Part 2][10]

## Other Common Topics

## Dimension Reduction

- By constraining or shrinking the estimated coefficients, we can often substantially reduce the variance at the cost of a negligible increase in bias. This can lead to substantial improvements in the accuracy with which we can predict the response for observations not used in model training.
- Irrelevant variables lead to unnecessary complexity in the resulting model.

### Regression in high dimension

When p\>n, regardless of whether there truly is a relationship between the features and the response, least-squares will yield a set of coefficient estimates that result in a perfect fit to the data, such that the residuals are zero. This is problematic because this perfect fit will almost certainly lead to overfitting of the data.

- regularization or shrinkage plays a key role in high-dimensional problems,
- appropriate tuning parameter selection is crucial for good predictive performance, and
- the test error tends to increase as the dimensionality of the problem (i.e. the number of features or predictors) increases unless the additional features are truly associated with the response. [Curse of high dimensions]

Adding noise features that are not truly associated with the response will lead to a deterioration in the fitted model, and consequently an increased test set error. This is because noise features increase the dimensionality of the problem, exacerbating the risk of overfitting (since noise features may be assigned nonzero coefficients due to chance associations with the response on the training set) without any potential upside in terms of improved test set error. Thus, we see that new technologies that allow for the collection of measurements for thousands or millions of features are a double-edged sword: they can lead to improved predictive models if these features are relevant to the problem at hand, but will lead to worse results if the features are not relevant. Even if they are relevant, the variance incurred in fitting their coefficients may outweigh the reduction in bias that they bring.

## Subset Selection

This approach involves identifying a subset of the p predictors that we believe to be related to the response. We then fit a model using least-squares on the reduced set of variables.

### Stepwise

- Computationally cheaper than best subset selection. Can be applied in settings where p is too large to apply the best subset selection.
- Not guaranteed to yield the best model containing a subset of the p predictors. For instance, suppose that in a given data set with p = 3 predictors, the best possible one-variable model contains X1, and the best possible two-variable model instead contains X2 and X3. Then forward stepwise selection will fail to select the best possible two-variable model, because M1 will contain X1, so M2 must also contain X1 together with one additional variable.
- p & n: Backward selection requires that the number of samples n is larger than the number of variables p (so that the full model can be fit). In contrast, forward stepwise can be used even when n ? p, and so is the only viable subset method when p is very large.
- Hybrid: Variables are added to the model sequentially, in analogy to forward selection. However, after adding each new variable, the method may also remove any variables that no longer provide an improvement in the model fit. Such an approach attempts to more closely mimic best subset selection while retaining the computational advantages of forwarding and backward stepwise selection.

### Best subset selection

- Why not R-squared? RSS of these p + 1 models decreases monotonically, and the R2 increases monotonically – use AIC, BIC, Adjusted $R^2$ so that we don’t get the largest model.
- Why CV? A low RSS or high $R^2$ indicates low training error, but we want low testing error so CV.
- Applying to Logistic regressions? Instead of ordering models by RSS in Step 2 of Algorithm 6.1, we instead use the deviance, measure deviance that plays the role of RSS for a broader class of models. The deviance is negative two times the maximized log-likelihood; the smaller the deviance, the better the fit.
- Limitations? Need $2^p$ models for $p$ predictors. Computationally expensive.

### Regularization

#### Bias-variance trade-off

The main motivation for regularization: Bias-variance trade-off.

Models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples, and vice versa.

This approach involves fitting a model involving all p predictors. However, the estimated coefficients are shrunk towards zero relatives to the least-squares estimates. This shrinkage (also known as regularization) has the effect of reducing variance, and therefore improve the fit. Depending on what type of shrinkage is performed, some coefficients may be estimated to be exactly zero. Hence, shrinkage methods can also perform variable selection.

#### Lasso, L1 Regularization

- As with ridge regression, the lasso shrinks the coefficient estimates towards zero.
- Lasso’s L1 penalty has the effect of forcing some coefficient estimates to be exactly equal to zero when the tuning parameter $lambda$ is sufficiently large. So, lasso also performs variable selection.
- Therefore, models generated from the lasso are generally much easier to interpret than those produced by ridge regression. We say that the lasso yields sparse models—that is, models that involve only a subset of the variables.
- Use cross validation to select $lambda$
- Tends to do well when only a few predictors influence the response.
- Does variable selection.

#### Ridge, L2 Regularization

The value of $beta X$ is affected by the scaling of predictors. Therefore, all variables should be scaled by their standard deviation.

- As $lambda$ increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias. As $lambda$ increases, the shrinkage of the ridge coefficient estimates leads to a substantial reduction in the variance of the predictions, at the expense of a slight increase in bias.
- In general, in situations where the relationship between the response and the predictors is close to linear, the least-squares estimates will have low bias but may have high variance. This means that a small change in the training data can cause a large change in the least-squares coefficient estimates. In particular, when the number of variables p is almost as large as the number of observations n. Hence, ridge regression works best in situations where the least-squares estimates have high variance.
  - models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples, and vice versa.
- Advantage: Computational efficiency over best subset selection.
- Disadvantage: Ridge regression will include all p predictors in the final model. The penalty factor will set them towards 0 but not exactly 0. Lasso overcomes this.
- Works well if most predictors impact the response.
- Does not do the variable selection.

##### Elastic Net

- Lasso and ridge are computationally feasible alternatives to best subset selection that replace the intractable form of the budget in (6.10) with forms that are much easier to solve. Of course, the lasso is much more closely related to best subset selection, since only the lasso performs feature selection for s sufficiently small.
- Cross-validation provides a simple way to tackle this problem. We choose a grid of $lambda$ values and compute the cross-validation error for each value of $lambda$

## Distance Metrics

- Euclidean
	  - Applications: kNN, kMeans
- Cosine
	  - Application
		- The cosine similarity is most commonly used in high-dimensional positive spaces. For example, in information retrieval and text mining, each term is notionally assigned a different dimension and a document is characterized by a vector where the value in each dimension corresponds to the number of times the term appears in the document. Cosine similarity then gives a useful measure of how similar two documents are likely to be in terms of their subject matter.
		- We can create vector data in a manner that can be used to retrieve information when queried. Once the unstructured data is transformed into vector form, we can use the cosine similarity metric to filter out the irrelevant documents from the corpus.
- Cosine vs Euclidean
	  - Cosine similarity is a metric used to determine how similar the documents are irrespective of their size.
	  - When plotted on a multi-dimensional space, where each dimension corresponds to a word in the document, the cosine similarity captures the orientation (the angle) of the documents and not the magnitude. If you want the magnitude, compute the Euclidean distance instead.
	  - The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance because of the size (like, the word ‘cricket’ appeared 50 times in one document and 10 times in another) they could still have a smaller angle between them. Smaller the angle, the higher the similarity.
	  - Intuition: direction and magnitude.
		- Direction: "preference" / "style" / "sentiment" / "latent variable" of the vector,
		- Magnitude: how strong it is towards that direction.
		- When classifying documents we'd like to categorize them by their overall sentiment, so we use the angular distance.
		- Euclidean distance is susceptible to documents being clustered by their L2-norm (magnitude, in the 2-dimensional case) instead of direction, i.e. vectors with quite different directions would be clustered because their distances from the origin are similar.

```python
corpus = [
    'brown fox jumped over the brown dog',
    'quick brown fox',
    'brown brown dog',
    'fox ate the dog']

query = ["brown"]

X = vectorizer.fit_transform(corpus)

Y = vectorizer.transform(query)

cosine_similarity(Y, X.toarray())

Results:

[0.54267123, 0.44181486, 0.84003859, 0]
```

- Edit Distance
	- String and list similarity

## Class Imbalance

- An imbalance occurs when one or more classes have very low proportions in the training data as compared to the other classes. Imbalance can be present in any data set or application, and hence, the practitioner should be aware of the implications of modeling this type of data. Examples
  - Online advertising: An advertisement is presented to a viewer which creates an impression. The click-through rate is the number of times an ad was clicked on divided by the total number of impressions and tends to be very low (Richardson et al. 2007 cite a rate less than 2.4%). Pharmaceutical research: High-throughput screening is an experimental technique where large numbers of molecules (10000s) are rapidly evaluated for biological activity. Usually, only a few molecules show high activity; therefore, the frequency of interesting compounds is low. Insurance claims: Artis et al. (2002) investigated auto insurance damage claims in Spain between the years of 1993 and 1996. Of claims undergoing auditing, the rate of fraud was estimated to be approximately 22 %.
- These results imply that the models achieve good specificity (in the Ads example, since almost no one clicks on ads) but have poor sensitivity.

### Correction Methods

- Determine alternative cutoffs
  - When there are two possible outcome categories, another method for increasing the prediction accuracy of the minority class samples is to determine alternative cutoffs for the predicted probabilities which effectively change the definition of a predicted event. The most straightforward approach is to use the ROC curve since it calculates the sensitivity and specificity across a continuum of cutoffs. Using this curve, an appropriate balance between sensitivity and specificity can be determined.
  - decreasing the cutoff for the probability of responding increases the sensitivity (at the expense of the specificity). There may be situations where the sensitivity/specificity trade-off can be accomplished without severely compromising the accuracy of the majority class (which, of course, depends on the context of the problem). First, if there is a particular target that must be met for the sensitivity or specificity, this point can be found on the ROC curve and the corresponding cutoff can be determined.
  - It is important, especially for small sample sizes, to use an independent data set to derive the cutoff.
	- If the training set predictions are used, there is likely a large optimistic bias in the class probabilities that will lead to inaccurate assessments of the sensitivity and specificity.
	- If the test set is used, it is no longer an unbiased source to judge model performance.
  - When comparing models, assessing model performance using class membership prediction might not be the best approach because it depends on our alternative cutoffs. Instead, ROC/AUC is better for tuning and model comparison.
### Adjusting case weights
  - Many of the predictive models for classification can use case weights where each data point can be given more emphasis in the model training phase. For example, previously discussed boosting approaches to classification and regression trees can create a sequence of models, each of which applies different case weights at each iteration.

### Up/Down-Sampling
  - Instead of having the model deal with the imbalance, we can attempt to balance the class frequencies. Taking this approach eliminates the fundamental imbalance issue that plagues model training. However, if the training set is sampled to be balanced, the test set should be sampled to be more consistent with the state of nature and should reflect the imbalance so that honest estimates of future performance can be computed.
  - up-sampling in which cases from the minority classes are sampled with replacement until each class has approximately the same number.
  - It should be noted that when using modified versions of the training set, resampled estimates of model performance can become biased. For example, if the data are up-sampled, resampling procedures are likely to have the same sample in the cases that are used to build the model as well as the holdout set, leading to optimistic results. Despite this, resampling methods can still be effective at tuning the models.
- SMOTE 
	[smote][11]
- Near Miss

## Modeling

Starter code to create a data set for testing models.

```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.SVM import SVC

# define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
```

### Linear

[www.reed.edu/economics/parker/s11/312/notes/Notes2.pdf][12]

#### Assumptions

- [Classical assumptions][13]
- Linear relationship between independent and dependent variables.
- [No influential outliers](\\\#influential outliers)
- [NOT NEEDED][14] Multivariate normality (all variables to be multivariate normal)
  - Checked with a histogram or a Q-Q Plot. Normality can also be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov test.
  - When the data is not normally distributed a non-linear transformation (e.g., log-transformation) might fix this issue.
  - [stats.stackexchange.com/questions/148803/how-does-linear-regression-use-the-normal-distribution][15]
  - [www.wikiwand.com/en/Gauss%E2%80%93Markov\_theorem][16]
	- Does not require error normality to get the best OLS estimates.
- No or little multicollinearity
- The error terms are independent of one another. Since the error terms represent random influences, they are also centered around 0.

  - Violated when there is autocorrelation (Serial correlation) [link][17]
  - Autocorrelation occurs when the residuals are not independent of each other. Autocorrelation (serial correlation) is present whenever the value of one observation’s error term allows us to predict the value of the next. Autocorrelation of the errors violates the ordinary least squares assumption that the error terms are uncorrelated, meaning that the Gauss Markov theorem does not apply and that OLS estimators are no longer the Best Linear Unbiased Estimators (BLUE).
	- [Causes][18]
	- Detection
	  - A scatterplot allows you to check for autocorrelations.
	  - $e\_t = rho e\_t-1 + v\_t$, where $v\_t$ are independent from one another. If $rho$ does not equal to 1, there’s autocorrelation.
	- Consequence
	  - Coefficient value will still be unbiased.
	  - Variance of the coefficient estimate’s probability distribution may be biased. The standard errors tend to be underestimated (and the t-scores overestimated) when the autocorrelations of the errors at low lags are positive.

- Homoscedasticity
  - Check using the scatter plot (meaning the residuals are equal across the regression line). Heteroscedastic:
  - Heteroscedasticity does not cause ordinary least squares coefficient estimates to be biased, although it can cause ordinary least squares estimates of the variance (and, thus, standard errors) of the coefficients to be biased, possibly above or below the true or population variance. Thus, regression analysis using heteroscedastic data will still provide an unbiased estimate for the relationship between the predictor variable and the outcome, but standard errors and therefore inferences obtained from data analysis are suspect. Biased standard errors lead to biased inference, so the results of hypothesis tests are possibly wrong. For example, if OLS is performed on a heteroscedastic data set, yielding biased standard error estimation, a researcher might fail to reject a null hypothesis at a given significance level, when that null hypothesis was actually uncharacteristic of the actual population (making a type II error).
  - The common fix is to use a robust method for computing the covariance matrix aka standard errors. Which one you use is somewhat domain-dependent but White's method is a start.

#### Trade Offs

- Pro
  - Interpretability of linear parameters
  - Fast calculation
- Con
  - It is also important to check for outliers since linear regression is sensitive to outlier effects.
  - The solution we obtain is a flat hyperplane. If the data have a curvature or nonlinear structure, then regression will not be able to identify these characteristics.
	- Curvature in the predicted-versus-residual plot is a primary indicator that the underlying relationship is not linear. The larger the number of original predictors, the less practical including some transformations becomes. Taking this approach can cause the data matrix to have more predictors than observations, and we then again cannot invert the matrix.
  - Cannot invert the matrix when $p\>n$
	- If $n\>\>p$, least-squares estimates tend to also have low variance, and hence will perform well on test observations.
	- If n is not much larger than p, then there can be a lot of variability in the least-squares fit, resulting in overfitting and consequently poor predictions on future observations not used in model training.
	- Hard to get the cross-validated error on prediction when p close to n. Consider, for example, a data set where there are 100 samples and 75 predictors. If we use a resampling scheme that uses two-thirds of the data for training, then we will be unable to find a unique set of regression coefficients, since the number of predictors in the training set will be larger than the number of samples.
	- If $p \> n$, then there is no longer a unique least squares coefficient estimate: the variance is infinite so the method cannot be used at all.

#### Non-linear Relationship

Fit nonlinear relationship

$textwage = alpha + beta texteducation + u$: assumes change in wages is constant for all educational levels. E.g., increasing education from 5 to 6 years leads to the same dollar increase in wages as increasing education from 11 to 12, or 15 to 16, etc.

$log(textwage) = alpha + betatexteducation + u$: each year of education leads to a constant proportionate (i.e., percentage) increase in wages

$textwage = alpha + beta log(textyears of experience) + u$: If we increase x by one percent, we expect y to increase by ($beta$ 1/100) units of y.

$textsalary = alpha + beta log(textsales) + u$: For each 1% increase in sales, salary increase by $beta %$

Many nonlinear specifications can be converted to linear form by performing transformations on the variables in the model. Detect by scatterplots of the IV against the DV.

Polynomial model

If it is thought that the slope of the effect of Xi on E(Y) changes sign as Xi increases. For many such models, the relationship between Xi and E(Y) can be accurately reflected with a specification in which Y is viewed as a function of Xi and one or more powers of Xi

When M = 2, $beta\_1$ indicates the overall linear trend (positive or negative) in the relationship between X and Y across the observed data. $beta\_2$ indicates the direction of curvature. For example, a positive coefficient for $X$ and a negative coefficient for $X^2$ cause the curve to rise initially and then fall.

Concave upward, $\_2$ is positive

Concave downward, $\_2$ is negative.

A polynomial of order k will have a maximum of k-1 bends (k-1 points at which the slope of the curve changes direction). Note that the bends do not necessarily have to occur within the observed values of the Xs.

Drawback: They can perform poorly on the extremes of the predictor

Fractional polynomials differ from regular polynomials in that they allow logarithms, allow non-integer powers, and they allow powers to be repeated. [Link][19]

Exponential Model $ln(Y) = alpha + beta X + epsilon, Y = e^alpha + beta X + epsilon$

- Some variables might increase exponentially rather than arithmetically. For example, each year of education may be worth an additional 5% income, rather than, say, 2,000. Hence, for somebody who would otherwise make 20,000 a year, an additional year of education would raise their income 1,000. For those who would otherwise be expected to make 40,000, an additional year could be worth 2,000. Note that the actual dollar amount of the increase is different, but the percentage increase is the same. When $beta$ is positive, the curve has a positive slope throughout, but the slope gradually increases in magnitude as X increases.

- When $beta$ is negative, the curve has a negative slope throughout and the slope gradually decreases in magnitude as X increases, with the curve approaching the X-axis as Y gets infinitely large.

Exponential models – Power Models

- ln Y \~ ln X
  - Every 1% increase in X is associated with a $beta$ percentage change in E(Y)

**[Fourier Basis][20]**

$phi\_2n-1(x)=textsin(fracxn)$, $phi\_2n(x)=textcos(fracxn)$, tune integer value $n=1, 2, ldots, 5$

Gaussian Basis

$phi\_n(x)=e^-(x-5n)^2/a$

#### Interaction Terms

- Height = 35 + 4.2*Bacteria + 9*Sun + 3.2*Bacteria*Sun
- The effect of Bacteria on Height is 4.2 + 3.2\*Sun.
  - For plants in partial sun, Sun = 0, so the effect of Bacteria is 4.2 + 3.2\*0 = 4.2.

So for two plants in partial sun, a plant with 1000 more bacteria/ml in the soil would be expected to be 4.2 cm taller than a plant with fewer bacteria.

- For plants in full sun, however, the effect of Bacteria is 4.2 + 3.2\*1 = 7.4.

So for two plants in full sun, a plant with 1000 more bacteria/ml in the soil would be expected to be 7.4 cm taller than a plant with fewer bacteria.

- Because of the interaction, the effect of having more bacteria in the soil is different if a plant is in full or partial sun. Another way of saying this is that the slopes of the regression lines between height and bacteria count are different for the different categories of the sun. B3 indicates how different those slopes are.
- Interpreting B2 is more difficult. B2 is the effect of the Sun when Bacteria = 0. Since Bacteria is a continuous variable, it is unlikely that it equals 0 often, if ever, so B2 can be virtually meaningless by itself. Instead, it is more useful to understand the effect of the Sun, but again, this can be difficult. The effect of the Sun is B2 + B3\*Bacteria, which is different at every one of the infinite values of Bacteria. For that reason, often the only way to get an intuitive understanding of the effect of Sun is to plug a few values of Bacteria into the equation to see how Height, the response variable, changes.

### GLM

[Regression Overview][21]

The coefficient of GLM is always linked to the outcome $mu\_i$ by the link function.

$$frac1mu\_i=alpha+beta\_1 x\_1+beta\_2 x\_2 + epsilon$$

#### Distributions

##### Negative Binomial

A discrete probability distribution that models the number of failures in a sequence of independent and identically distributed Bernoulli trials before a specified (non-random) number of successes (denoted r) occurs. For example, we can define rolling a 6 on a die as a success, and rolling any other number as a failure, and ask how many failed rolls will occur before we see the third success (r = 3). In such a case, the probability distribution of the number of non-6s that appears will be a negative binomial distribution.

Coefficient

So if you have a negative $beta$ for a dummy variable x, you can say that "on average, x lowers the expected value of log(y) by $beta$ \*100 percent."

You can also get a multiplicative effect by exponentiating the coefficients. For example, if $beta$ D=.43, then folks with D=1 are expected to have exp.43=1.54 times higher outcome (holding everything else constant). This is the incidence rate ratio interpretation

Link functions

Parameters to tune

##### Poisson

##### Binomial (Logit when log

**Algorithm**

Logit function = log-odds = the logarithm of the odds $fracp1-p$.

Instead of fitting a straight line or hyperplane, the logistic regression model uses the logistic function to squeeze the output of a linear equation between 0 and 1. The logistic function will always produce an S-shaped curve of this form, and so regardless of the value of X, we will obtain a sensible prediction. On the other hand, if directly use linear, get a negative probability.

- $log(fracp(X)1-p(X))=beta\_0 + beta\_1 X$

- Why use maximum likelihood to fit a logistic regression model.
  - If you're fitting a binomial GLM with a logit link (i.e. a logistic regression model), then your regression equation is the log-odds that the response value is a '1' (or a 'success'), conditioned on the predictor values.

**Preprocessing**

Scale so that penalty applies evenly during regularization. 

**Hyperparameters**

Logistic regression does not really have any critical hyperparameters to tune.

Sometimes, you can see useful differences in performance or convergence with different solvers (solver).

- solver in [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]

Regularization (penalty) can sometimes be helpful.

- penalty in [‘none’, ‘l1’, ‘l2’, ‘elastic net’]

Note: not all solvers support all regularization terms.

The C parameter controls the penality strength, which can also be effective.

- C in [100, 10, 1.0, 0.1, 0.01]

**Coefficient Interpretation**

Exponentiating the log odds gives you the odds ratio for a one-unit increase in your variable. Here, the odds of your outcome for men are exp(0.014) = 1.01 times that of the odds of your outcome in women.

$ln(fracp(X)1-p(X))=beta\_0 + beta\_1 X$

$textodds(textfemale) = e^beta\_0 + beta\_1 \* 0$

$textodds(textmale) = e^beta\_0 + beta\_1 \* 1$

$textodds ratio(textmale) = textodds(textmale) / textodds(textfemale) = e^0.014 = 1.01$

$textodds ratio(textfemale) = 1 / 1.01 = 0.99$

**Multiclass Extension**

Multiple-class extensions of logit not used all that often.

**Assumptions**

- The outcome is a binary
- There is a linear relationship between the logit of the outcome and each predictor variable. (The true conditional probabilities are a logistic function of the independent variables.)
  - Check by visually inspecting the scatter plot between each predictor and the logit values. Create scatter plots.
  - If the scatter plot shows non-linearity, you need other methods to build the model such as including 2 or 3-power terms, fractional polynomials, and spline function.
- There is no influential values (extreme values or outliers) in the continuous predictors
- Second, logistic regression requires the observations to be independent of each other. In other words, the observations should not come from repeated measurements or matched data.
- Little or no multicollinearity among the independent variables. This means that the independent variables should not be too highly correlated with each other.
- Comparing to OLS
  - First, logistic regression does not require a linear relationship between the dependent and independent variables.
  - Second, the error terms (residuals) do not need to be normally distributed.
  - Third, homoscedasticity is not required. Finally, the dependent variable in logistic regression is not measured on an interval or ratio scale.

**Link Test**

The Stata command linktest can be used to detect a specification error, and it is issued after the logit or logistic command. The idea behind linktest is that if the model is properly specified, one should not be able to find any additional predictors that are statistically significant except by chance. After the regression command (in our case, logit or logistic), linktest uses the linear predicted value (hat) and linear predicted value squared (hatsq) as the predictors to rebuild the model. The variable hat should be a statistically significant predictor since it is the predicted value from the model. This will be the case unless the model is completely misspecified. On the other hand, if our model is properly specified, variable hatsq shouldn’t have much predictive power except by chance. Therefore, if hatsq is significant, then the linktest is significant. This usually means that either we have omitted relevant variable(s) or our link function is not correctly specified.

[link][22]

### Linear Mixed Effect Model

`Statsmodel` documentation provides an excellent explanation for LME models.

[Link][23]

### Linear Discriminant Analysis

A method for finding a linear combination of features that characterizes or separates two or more classes of objects or events. The resulting combination may be used as

- a linear classifier
- or, more commonly, for dimensionality reduction before later classification

**Relationship with Other Methods**

- ANOVA uses categorical independent variables and a continuous dependent variable

Questions along the line of – knowing the differences in categories they belong to, are there differences in the outcome variable (the number of products sold in the following case).

- Discriminant analysis has continuous independent variables and a categorical dependent variable (i.e. the class label).
  - Similar to Logistic regression and probit regression.
  - These other methods are preferable in applications where it is not reasonable to assume that the independent variables are normally distributed, which is a fundamental assumption of the LDA method.

LDA is also closely related to principal component analysis (PCA) and factor analysis in that they both look for linear combinations of variables that best explain the data.[4] LDA explicitly attempts to model the difference between the classes of data. PCA, in contrast, does not consider any difference in class, and factor analysis builds the feature combinations based on differences rather than similarities. Discriminant analysis is also different from factor analysis in that it is not an interdependence technique: a distinction between independent variables and dependent variables (also called criterion variables) must be made.

**Assumptions**

The assumptions of discriminant analysis are the same as those for MANOVA. The analysis is quite sensitive to outliers and the size of the smallest group must be larger than the number of predictor variables.

- Multivariate normality: Independent variables are normal for each level of the grouping variable.
- Homogeneity of variance/covariance (homoscedasticity): Variances among group variables are the same across levels of predictors. Can be tested with Box's M statistic. It has been suggested, however, that linear discriminant analysis be used when covariances are equal, and that quadratic discriminant analysis may be used when covariances are not equal.
- Multicollinearity: Predictive power can decrease with an increased correlation between predictor variables.
- Independence: Participants are assumed to be randomly sampled, and a participant's score on one variable is assumed to be independent of scores on that variable for all other participants.

It has been suggested that discriminant analysis is relatively robust to slight violations of these assumptions, and it has also been shown that discriminant analysis may still be reliable when using dichotomous variables (where multivariate normality is often violated).

### Additional Linear Topics

**Influential Outliers**

- Can be examined by visualizing the Cook’s distance values.
- Note that not all outliers are influential observations. To check whether the data contains potential influential observations, the standardized residual error can be inspected. Data points with absolute standardized residuals above 3 represent possible outliers and may deserve closer attention.

**Other methods**

- Note: Average and Standard Deviation are only valid for Gaussian distributions.
- Isolation forest
  - Isolation Forest Algorithm
  - Return the anomaly score of each sample using the IsolationForest algorithm
  - The IsolationForest 'isolates' observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
  - Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
  - This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
  - Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produces shorter path lengths for particular samples, they are highly likely to be anomalies.

**Multicollinearity**

- Occurs when the independent variables are too highly correlated with each other. One of the variables that are supposedly positively correlated with the dependent variable has a negative coefficient. Why?

_Test by_
  - Correlation matrix – Pairwise Pearson’s Bivariate Correlation among all independent variables need to be smaller than 1.
  - Variance Inflation Factor $textVIF = frac11 - R\_{j^2}$. The VIF equals 1 when the vector $X\_j$ is orthogonal to each column of the design matrix for the regression of Xj on the other covariates. With VIF \> 10 there is an indication that multicollinearity may be present; with VIF \> 100 there is certainly multicollinearity among the variables.
  - $R\_j^2$ is the multiple $R^2$ for the regression of $X\_j$ on the other covariates (a regression that does not involve the response variable Y). This identity separates the influences of several distinct factors on the variance of the coefficient estimate

- [Version 2][24]

TODO V2 to be combined 

  - Collinearity is the state where two variables are highly correlated and contain similar information about the variance within a given dataset. To detect colinearity among variables, simply create a correlation matrix and find variables with large absolute values. In R use the corr function and in python this can by accomplished by using numpy's corrcoef function.
  - Multicollinearity on the other hand is more troublesome to detect because it emerges when three or more variables, which are highly correlated, are included within a model. To make matters worst multicollinearity can emerge even when isolated pairs of variables are not colinear.
  - A common R function used for testing regression assumptions and specifically multicollinearity is "VIF()" and unlike many statistical concepts, its formula is straightforward:
  - $ V.I.F. = 1 / (1 - R^2). $
  - The Variance Inflation Factor (VIF) is a measure of colinearity among predictor variables within a multiple regression. It is calculated by taking the ratio of the variance of all a given model's betas divided by the variance of a single beta if it were fit alone.
  - Steps for Implementing VIF
	- Run a multiple regression.
	- Calculate the VIF factors.
	- Inspect the factors for each predictor variable, if the VIF is between 5-10, multicollinearity is likely present and you should consider dropping the variable.
- In Logistic regression
  - You can use whatever method you would use for ordinary regression. The dependent variable is irrelevant to multicollinearity issues, so it doesn't matter if you used logistic regression or regular regression or whatever.
- Depending on the situation, might/might not need to process further.
  - Can still be used for prediction when collinearity exists within the data. But since the regression coefficients to determine these predictions are not unique, we lose our ability to meaningfully interpret the coefficients.
  - Remove pairwise correlated predictors, the variance inflation factor. On many occasions, relationships among predictors can be complex and involve many predictors. In these cases, manual removal of specific predictors may not be possible and models that can tolerate collinearity may be more useful
  - Ridge regression approach should allow you to deal with collinear predictors. Just be careful to normalize your feature matrix X appropriately before using it, otherwise, you will risk regularising each feature disproportionately (yes, I mean the 0/1 columns, you should scale them such that each column has unit variance and mean 0).
  - Use data reduction method to put summary scores into the model instead of individual variables.
  - F-test for overall effect.
- A unique inverse of this matrix exists when
  - No predictor can be determined from a combination of one or more of the other predictors
  - The number of samples is greater than the number of predictors. If the data fall under either of these conditions, then a unique set of regression coefficients does not exist.

**Normality Assumption**

**Regression with Skewed Data**

E.g. Modeling user clicks

Skewness can be detected through QQ plot and histogram

Linear regression is not the right choice for your outcome, given:

1. The outcome variable is not normally distributed
2. The outcome variable being limited in the values it can take on (count data means the predicted values cannot be negative)
3. What appears to be a high frequency of cases with 0 visits

**Limited dependent variable models for count data**

The estimation strategy you can choose from is dictated by the "structure" of your outcome variable. That is, if your outcome variable is limited in the values it can take on (i.e. if it's a [limited dependent variable][25]), you need to choose a model where the predicted values will fall within the possible range for your outcome. While sometimes linear regression is a good approximation for limited dependent variables (for example, in the case of binary logit/probit), oftentimes it is not. Enter [Generalized Linear Models][26]. In your case, because the outcome variable is count data, you have several choices:

1. Poisson model
2. Negative Binomial model
3. Zero Inflated Poisson (ZIP) model
4. Zero Inflated Negative Binomial (ZINB) model

The choice is usually empirically determined. I will briefly discuss choosing between these options below.

Poisson vs. Negative Binomial

In general, Poisson is the go-to "general workhorse" model of the 4 count data models I mentioned above. A limitation of the model is the assumption that the conditional variance = the conditional mean, which may not always be true. If your model is overdispersed (conditional variance \> conditional mean), you will need to use the Negative Binomial model instead. Fortunately, when you run the Negative Binomial, the output usually includes a statistical test for the dispersion parameter (R calls this dispersion parameter "theta ( $theta$ )," which is called "alpha" in other packages). The null hypothesis in the choice between Poisson vs. Negative Binomial is H0: $theta$ =0, while the alternative hypothesis is H1: $theta$ $neq$ 0. If the coefficient on $theta$ is significant, there is evidence of overdispersion in the model, and you would choose Negative Binomial over Poisson. If the coefficient is not statistically significant, present Poisson results.

ZIP vs. ZINB

One potential complication is zero inflation, which might be an issue here. This is where the zero-inflated model's ZIP and ZINB come in. Using these models, you assume that the process generating the zero values is separate from the process generating the other, non-zero values. As with before, ZINB is appropriate when the outcome has excessive zeroes and is overdispersed, while ZIP is appropriate when the outcome has excessive zeroes but conditional mean = conditional variance. For the zero-inflated models, in addition to the model covariates you have listed above, you will need to think of variables that may have generated the excess zeroes you saw in the outcome. Again, there are statistical tests that come with the output of these models (sometimes you might have to specify them when you execute a command) that will let you empirically decide which model is the best one for your data. There are two tests of interest: The first is the test of the coefficient on the dispersion parameter.

$theta$ and the second is what is known as the Vuong test, which tells you whether the excess zeroes are generated by a separate process (i.e. whether there is, indeed, zero inflation in the outcome).

In comparing the choice between ZIP and ZINB, you will again look at the test of the dispersion parameter theta.

Other users can comment on the "usual" workflow, but my approach is to visualize the data and go from there. In your case, I would probably start with ZINB and run both the test on the coefficient on $theta$ and the Vuong test\_, since it's the test on the coefficient on $theta$ would tell you which one was better between ZIP and ZINB, and the Vuong test would tell you whether you should use zero-inflated models.


### Decision Tree

**Algorithm**

- Start with all variables in one group.
- Find the variable/split that best separates the outcomes (successive binary partitions based on the different predictors - explanatory variables).
  - Evaluation “homogeneity” within each group
  - Divide the data into two groups (“leaves”) on that split (“node”).
  - Within each split, find the best variable/split that separates the outcomes.
- Continue until the groups are too small or sufficiently “pure”.

**Algorithm with Pruning**

- A similar algorithm for both building and pruning
  - Partition the data in K1 groups.
  - Remove the first group, and train the data on the remaining K1 - 1 groups.
  - Use K2-fold cross-validation (on the K1 - 1 groups) to choose $alpha$. That is, divide the training observations into K2 folds and find $alpha$ that minimizes the error.
  - Using the subtree that corresponds to the chosen value of $alpha$, predicts the first of the K1 hold out samples.
  - Repeat steps 2-4 using the remaining K1 - 1 groups.

**Interactions**

The impact of predictor A on the outcome depends on the value of a different predictor B.

CARTs can easily model this by first splitting on B and then, on lower levels, splitting on A differently for low values of B than for high values of B.

If you have enough data, this will indeed happen automatically. Since CARTs consider all possible interactions (instead of the user having to explicitly model them), then need much more data to avoid fitting noise, i.e., spurious interactions.

**Hyperparameters**

- Node impurity measures
  - Gini is intended for continuous attributes, and Entropy for attributes that occur in classes.
  - Gini will tend to find the largest class, and entropy tends to find groups of classes that make up \~50% of the data.
  - Gini to minimize misclassification.
  - Entropy for exploratory analysis.
  - Some studies show this doesn’t matter – these differ less than 2% of the time.
  - Entropy may be a little slower to compute.
- Pruning parameter $alpha$. Cost complexity pruning—also known as weakest link pruning—gives us a way to do just this. Rather than considering every possible subtree, we consider a sequence of trees indexed by a nonnegative tuning parameter $alpha$. When $alpha$ = 0, then the subtree T will simply equal T0, because then (8.4) just measures the training error. However, as $alpha$ increases, there is a price to pay for having a tree with many terminal nodes, and so the quantity (8.4) will tend to be minimized for a smaller subtree. Here T(0) is the unpruned tree obtained when $T = T\_0$.
  - T: Number of terminal nodes of the tree.
  - Resubstitution Error Rate (R). The resubstitution rate is a measure of error. It is the proportion of original observations that were misclassified by various subsets of the original tree. The resubstitution rate decreases as you go down the list of trees. The largest tree will always yield the lowest resubstitution error rate. However, choosing the tree with the lowest resubstitution rate is not the optimal choice, as this tree will have a bias. Large trees will put random variation in the predictions as they overfit outliers.
	- $R\_alpha(T) = R(T) +alpha|T|$
	  - When $alpha =inf$, only stump left
	  - When $alpha = 0$, full tree
	  - When $alpha = 1$, it’s possible to recursively prune the current tree to reduce the cost function because doing this reduces $|T|$.
	  - When $alpha = 2$, it’s possible to recursively prune the current tree to reduce the cost function because doing this reduces $|T|$.
  - This is reminiscent of LASSO where a similar formulation was used to control the complexity.
  - The way to think about cost complexity is to consider $alpha$ increasing. As $alpha$ gets bigger, the “best" tree will be smaller. But the test error will not be monotonically related to the size of the training tree.
	- $alpha$ small: If $alpha$ is set to be small, we are saying that the risk is more worrisome than the complexity and larger trees are favored because they reduce the risk.
	- $alpha$ large: If $alpha$ is set to be large, then the complexity of the tree is more worrisome, and smaller trees are favored.

**Trade Offs**

_Pro_

  - They are easy to explain; trees are easy to display graphically (which make them easy to interpret). (They mirror the typical human decision-making process.)
  - Can handle categorical or numerical predictors of response variables (indeed, they can handle mixed predictors at the same time!).
  - Can handle more than 2 groups for categorical predictions
  - Easily ignore redundant variables.
  - Perform better than linear models in non-linear settings.
	- Classification trees are non-linear models, so they immediately use interactions between variables.
  - Data transformations may be less important (monotone transformations on the explanatory variables won’t change anything).

_Con_
  - Straight CART does not generally have the same predictive accuracy as other classification approaches. (we will improve the model - see random forests, boosting, bagging)
  - Difficult to write down / consider the CART “model”
  - Without proper pruning, the model can easily lead to overfitting.
  - With lots of predictors, (even greedy) partitioning can become computationally unwieldy

### Tree-based ensemble algorithms

- Their algorithms are easy to understand and visualize: describing and sketching a decision tree is arguably easier than describing Support Vector Machines to your grandma
- They are non-parametric and don’t assume or require the data to follow a particular distribution: this will save you time transforming data to be normally distributed
- They can handle mixed data types: categorical variables do not necessarily have to be one hot encoded
- Multi-collinearity of features does not affect the accuracy and prediction performance of the model: features do not need to be removed or otherwise engineered to decrease the correlations and interactions between them
- They are robust against overfitting: because they use many weak learners that under-fit (high bias) and combine those predictions into a stronger learner, they reduce the overfitting (variance) of the model
- They are relatively robust against outliers and noise: in general, they will handle noisy data (e.g. features with no effect on the target) or outliers (e.g. extreme values) well with little effect on the overall performance (this point is debated for AdaBoost; more on that below)
- Inputs do not need to be scaled: preprocessing and transforming the features with MinMaxScaler or StandardScaler are not necessary
- They are computationally relatively inexpensive: compared to algorithms such as Support Vector Machines or neural networks they are faster
- They usually perform much better than their weak learners: decision trees will be less accurate due to their high variance/overfitting compared with boosting and bagging algorithms

### Bagging Trees

**Algorithm**

**Tuning Parameter**

- n\_estimators in [10, 100, 1000]
  - Good values might be a log scale from 10 to 1,000.
  - ELI5: Good explanation for log vs linear scale

**Trade Offs**

_Pro_

  - Can handle categorical or numerical response variables (indeed, they can handle mixed predictors at the same time!).
  - Can handle more than 2 groups for categorical predictions
  - Easily ignore redundant variables.
  - Perform better than linear models in non-linear settings.
  - Classification trees are non-linear models, so they immediately use interactions between variables.
  - Data transformations may be less important (monotone transformations on the explanatory variables won’t change anything).
  - Similar bias to CART, but reduced variance (can be proved).
  - OOB samples can be used as test data to estimate the error rate of the tree.	
	- Why is OOB 1/3?

_Con_

  - Model is even harder to “write-down” (than CART). The model improves prediction accuracy at the expense of interpretability.
  - With lots of predictors, (even greedy) partitioning can become computationally unwieldy - now the computational task is even harder! (because of the number of trees grown for each bootstrap sample)

### RF

**Algorithms**

As with bagging, each tree in the forest casts a vote for the classification of a new sample, and the proportion of votes in each class across the ensemble is the predicted probability vector.

- 1. Bootstrap sample from the training set.
- 2. Grow an unpruned tree on this bootstrap sample.
  - At each split, select m variables and determine the best split using only these predictors.
  - Typically m = sqrt(p) or log\_2(p), where p is the number of features. Random forests are not overly sensitive to the value of m. [splits are chosen as with trees: according to either squared error or Gini index/cross-entropy / classification error.]
  - Do not prune the tree. Save the tree as is! (We don’t need to because this allows us to grow very different trees - and the potential of overfitting are mitigated by averaging trees built on bootstrapped data, and by decorrelating trees through selecting from a subset of predictors.)
- 3. For each tree grown on a bootstrap sample, predict the OOB samples. For each tree grown, 1/3 of the training samples won’t be in the bootstrap sample – those are called out of bootstrap (OOB) samples. OOB samples can be used as test data to estimate the error rate of the tree.
- 4. Combine the OOB predictions to create the “out-of-bag” error rate (either majority vote or the average of predictions/class probabilities).
- 5. All trees together represent the model that is used for new predictions (either majority vote or average).

**Hyperparameters**

- To tune max\_features, we recommend starting with five values that are somewhat evenly spaced across the range from 2 to P, where P is the number of predictors. We likewise recommend starting with an ensemble of 1,000 trees and increasing that number if performance is not yet close to a plateau. [‘sqrt’, ‘log2’]
- n\_estimators in [10, 100, 1000]
  - Ideally, this should be increased until no further improvement is seen in the model.

**Trade Offs**

- Pro
  - Note that by bootstrapping the samples and the predictor variables, we add another level of randomness over which we can average to again decrease the variability.
	- Comparing to bagging, de-correlate the trees in the forest. Reduces the variance further. Each tree has the same expectation, but the average will again reduce the variability.
  - Subset of predictors makes random forests much faster to search through than all predictors
  - The model is relatively insensitive to values of mtry.
  - As with most trees, the data pre-processing requirements are minimal - does not require standardizing data, etc.
  - Out-of-bag measures of performance can be calculated, including accuracy, sensitivity, specificity, and confusion matrices.
  - Automatically capture the interaction between variables
  - Random forests are quite accurate
  - Generally, models do not overfit the data, and CV is not needed. However, CV can be used to fit the tuning parameters (m, node size, max number of nodes, etc.).
	- Roughly speaking, some of the potential over-fitting that might happen in a single tree (which is a reason you do pruning generally) is mitigated by two things in a Random Forest:
	  - The fact that the samples used to train the individual trees are "bootstrapped".
	  - The fact that you have a multitude of random trees using random features and thus the individual trees are strong but not so correlated with each other.
  - Advantage over GBT
	- Although RF is also based on tree averaging, it has several clear advantages over GBRT:
	  - It is particularly insensitive to parameter choices
	  - It is known to be very resistant to overfitting
	  - It is very parallelizable.
- Con
  - Model is even harder to “write-down” (than CART). The model improves prediction accuracy at the expense of interpretability.
  - With lots of predictors, (even greedy) partitioning can become computationally unwieldy - now the computational task is even harder!

### GBM

- Boosting refers to a family of algorithms in which a set of weak learners (learners that are only slightly correlated with the true process) are combined to produce a strong learner.
- Boosting can be applied to any classification technique, but classification trees are a popular method for boosting since these can be made into weak learners by restricting the tree depth to create trees with few splits (also known as stumps).
- Also called Gradient Boosting Machine (GBM) or named for the specific implementation, such as XGBoost.
- Gradient Boosting \> Random Forest \> Bagging \> Single Trees
  - In his talk titled “Gradient Boosting Machine Learning” at H2O, Trevor Hastie made the comment that in general gradient boosting performs better than a random forest, which in turn performs better than individual decision trees.
- It works because: Since classification trees are a low bias/high variance technique, the ensemble of trees helps to drive down variance, producing a result that has low bias and low variance.

**Hyperparameters**

The gradient boosting algorithm has many parameters to tune.

- Learning\_rate (also called shrinkage or eta (learning\_rate)) in [0.001, 0.01, 0.1]
- n\_estimators [10, 100, 1000]
- subsample in [0.5, 0.7, 1.0]
  - number of rows or subset of the data to consider for each tree (subsample)
- max\_depth in [3, 7, 9]
  - the depth of each tree

**Best Practices**

[How to configure GBT][27]

- Learning rate (shrinkage parameter): The “shrinkage” parameter 0 ? v ? 1 controls the learning rate of the procedure. Empirically …, it was found that small values (v ?= 0.1) lead to much better generalization error.
- Smaller values of v lead to larger values of M for the same training risk so that there is a tradeoff between them. … In fact, the best strategy appears to be to set v to be very small (v ? 0.1) and then choose M by early stopping.
  - In the case of GBT, set the value for v, then gradually increase M until the performance starts to degrade. Monitoring the performance on a validation dataset to calibrate the number of trees and to use an early stopping procedure once performance on the validation dataset begins to degrade.
  - Neural Network context
	- Train the model once for a large number of training epochs.
	- During training, the model is evaluated on a holdout validation dataset after each epoch.
	- If the performance of the model on the validation dataset starts to degrade (e.g. loss begins to increase or accuracy begins to decrease), then the training process is stopped.
  - Additional considerations for early stopping:
	- More elaborate triggers may be required in practice. This is because the training of a neural network is stochastic and can be noisy. Plotted on a graph, the performance of a model on a validation dataset may go up and down many times. This means that the first sign of overfitting may not be a good place to stop training.
	- Performance of the model is evaluated on the validation set at the end of each epoch, which adds computational cost during training. This can be reduced by evaluating the model less frequently, such as every 2, 5, or 10 training epochs.
  - In general
	- Early stopping requires that you configure your network to be under constrained, meaning that it has more capacity than is required for the problem.
	- When training the network, a larger number of training epochs is used than may normally be required, to give the network plenty of opportunities to fit, then begin to overfit the training dataset.
	- There are three elements to using early stopping; they are:
	  - Monitoring model performance.
	  - Trigger to stop training.
	  - The choice of model to use.
- Sampling fraction: best is approximately 40% (f=0.4) – 50%. However, sampling only 30% or even 20% of the data at each iteration still gives a considerable improvement over no sampling at all, with a corresponding computational speed-up by factors of 3 and 5 respectively.
- Number of nodes in the tree: Friedman found values like 3 and 6 better than larger values like 11, 21, and 41. ESL Ch 10 recommends 6, with generally good values in the range of 4-to-8.

**Trade Offs**

- Pro
  - This method is very accurate. The training examples that were misclassified have their weights boosted, and a new tree is formed. This procedure is then repeated consecutively for the new trees. The final score is taken as a weighted sum of the scores of the individual leaves from all trees.
  - No normalization is needed when using different types of data (e.g., categorical and count data).
  - Trading off runtime efficiency and accuracy (i.e., relevance for search) can be easily achieved by truncating the number of trees used in the boosted trees model.
  - From the perspective of feature selection, a more interesting property of the boosted trees is that (a greedy) feature selection already happens in the algorithm when selecting splitting features (e.g., for regression trees, splitting features and splitting points are found to minimize the squared-error loss for any given partition of the data).
  - Moreover, as a byproduct, a sorted list of the relative importance of features (i.e., a feature importance list) is automatically generated for each boosted tree model.
- Con
  - Computationally expensive
	- Start by reducing the number of candidates through filtering. In a typical ranking setup, we need to evaluate the same trained model on multiple instances of feature vectors. For example, we need to rank \~1,000 different potential candidates for a given person and pick only the most relevant ones.
	  - Randomly sample 0.5 \~ 0.8 of the training data to decorrelate trees and prevent overfitting.
- Parameters
  - Tree depth (or interaction depth), 1 makes it an addictive model and usually gives good results
  - Number of iterations, build 100 - 1000 trees
  - shrinkage/learning rate, 0.01 \~ 0.001, smaller the slower
- Pearson residual
  - The raw residual divided by the square root of the variance function $V(
	mu)$.
  - A standardized Pearson residual has N(0,1) distribution. A value that exceeds 2 or 3 in absolute value is a sign of lack of fit.

**AdaBoost, Gradient Boosting, and XGBoost**

AdaBoost:

- additional penalty weight on error observations
- AdaBoost is not optimized for speed, therefore being significantly slower than XGBoost.
- noisy data leads to poor performance due to the algorithm spending too much time on learning extreme cases and skewing results.
- Compared to random forests and XGBoost, AdaBoost performs worse when irrelevant features are included in the model
- Application
  - AdaBoost is best used in a dataset with low noise, when computational complexity or timeliness of results is not the main concern and when there are not enough resources for broader hyperparameter tuning due to lack of time and knowledge of the user.
	Gradient Boosting: no penalty
	XGBoost: Node splitting in parallel

## Other Models

### Support Vector Machines

#### Algorithm

#### Trade Offs
- Pro
  - Can always fit a linear separating hyperplane in a high enough dimensional space.
  - The kernel trick makes it possible to not know the transformation functions, $phi$.
  - Because the optimization is on a convex function, the numerical process for finding solutions is extremely efficient.
  - It works really well with a clear margin of separation
  - It is effective in high dimensional spaces.
  - It is effective in cases where the number of dimensions is greater than the number of samples.
  - It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
- Con
  - Overall, doesn’t work when large data/noisy data/hard data/multi-class/non-numeric/need probability estimates
  - Can only classify binary categories (response variable).
  - All predictor variables must be numeric.
	- A great differential in range will allow variables with a large range to dominate the predictions. Either linearly scale each attribute to some range [ e.g., (-1, +1) or (0,1)] or divide by the standard deviation.
	- Categorical variables can be used if formatted as binary factor variables.
	- Whatever is done to the training data must also be done to the test data!
  - Another problem is the kernel function itself.
	- With primitive data (e.g., 2d data points), good kernels are easy to come by.
	- With harder data (e.g., MRI scans), finding a sensible kernel function may be much harder.
  - With really large data, it doesn't perform well because of the large amount of required training time
  - It also doesn’t perform very well when the data set has a lot of noise i.e., target classes are overlapping
  - SVM doesn’t directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.

### Ensemble Methods

- Stacking
  - GBT and Logistic

Similar to online and offline learning needed in the Facebook newsfeed to provide the most tailored recommendation, there also could be nimble online learning models that constantly monitor risks using live data stream, and a more sophisticated model that updates the feature selection and encoding daily.

- Bagging
- Boosting
  - How to use boosting for other models?
  - Hypothesis
	- Fit the initial model, obtain residual
	- Use the same model to fit the residual, and repeat this process
	- Stop until a criterion is reached. Obtain an ensemble model that’s the combination of multiple models, each with their own weights (depending on the model)

### Bayesian Models

The goal is to estimate the posterior distribution for the probability of observing each species, $p$, conditioned on the data, and hyperparameters, such as $alpha$.

In more formal terms, we assign probability distributions to unknown quantities $P(theta)$. Then, we use Bayes' theorem to transform the prior probability distribution into a posterior distribution $P(theta|y)$.

For instance, in a readability study,

- $theta$ measure the coefficient -- the effectiveness of each font on readability, and $P(theta)$ is its distribution.

#### Coin-flipping Problem (beta-binomial model)

**Set up:** We toss a coin several times and record how many heads and tails we get. Based on this data, we try to answer questions such as is the coin fair? Or, more generally, how biased is the coin?

**y:** number of heads.
**N:** number of tosses.

**Prior:** will land half of the time heads and half of the time tails; in Bayesian statistics, every time we do not know the value of a parameter, we put a prior on it.

$$P(theta)sim textBeta(alpha, beta)=fracGamma(alpha+beta)Gamma(alpha)Gamma(beta)theta^alpha-1(1-theta)beta-1$$

_The beta distribution looks similar to the binomial except for the first term, which is a normalizing constant that ensures the distribution integrates to 1, and $Gamma$ is the Greek uppercase gamma letter and represents gamma function. We can see from the preceding formula that the beta distribution has two parameters, $alpha$, and $beta$._

We use Beta for prior because

- beta distribution is restricted to be between 0 and 1, in the same way, $theta$ is. In general, we use the beta distribution when we want to model proportions of a binomial variable.
- versatility. As we can see in the preceding figure, the distribution adopts several shapes (all restricted to the [0, 1] interval), including a uniform distribution, Gaussian-like distributions, and U-like distributions.
- the beta distribution is the conjugate prior of the binomial distribution (which we are using as the likelihood). A **conjugate prior**[^or something?] of a likelihood is a prior that, when used in combination with a given likelihood, returns a posterior with the same functional form as the prior. Every time we use a beta distribution as the prior and a binomial distribution as the likelihood, we will get a beta as the posterior distribution.
  - There are other pairs of conjugate priors; for example, the Normal distribution is the conjugate before itself. For many years, Bayesian analysis was restricted to the use of conjugate priors. Conjugacy ensures mathematical tractability of the posterior, which is important given that a common problem in Bayesian statistics is ending up with a posterior we cannot solve analytically. This was a deal-breaker before the development of suitable computational methods to solve probabilistic methods.

**Likelihood:** chance of observing data given the prior. We need to choose a reasonable distribution here – in this case, Binomial.

Given a coin that's perceived as fair, what's the probability of observing the actual data.
$P(theta)sim textBeta(alpha, beta)$
$P(y)sim textBernoulli(p=theta)$

- The only unobserved variable in our model is $theta$.
- y is an observed variable representing the data; we do not need to sample that because we already know those values

```python
np.random.seed(123)
trials = 4
theta_real = 0.35 # known only because we are generating data
data = stats.bernoulli.rvs(p=theta_real, size=trials)

with pm.Model() as our_first_model:
    theta = pm.Beta('theta', alpha=1., beta=1.)  # prior
    y = pm.Bernoulli('y', p=theta, observed=data)  # likelihood
    trace = pm.sample(1000, random_seed=123)

az.plot_trace(trace)  # kde and individually sampled values
az.summary(trace)
az.plot_posterior(trace)
```

Sometimes, describing the posterior is not enough. Sometimes, we need to make decisions based on our inferences. We have to reduce a continuous estimation of a dichotomous one: yes-no, health-sick, contaminated-safe, and so on. We may need to decide if the coin is fair or not. A fair coin is one with a value of exactly 0.5. We can compare the value of 0.5 against the HPD interval. In Figure 2.2, we can see that the HPD goes from ≈0.02 to ≈0.71 and hence 0.5 is included in the HPD. According to our posterior, the coin seems to be tail-biased, but we cannot completely rule out the possibility that the coin is fair. If we want a sharper decision, we will need to collect more data to reduce the spread of the posterior or maybe we need to find out how to define a more informative prior.

#### Toy Model

$alpha$ comes from the hyperparameter specified in the Dirichlet distribution. It specifies our prior belief in the number of event occurrences.

**Prior:** Probability of observing any of the three animals is the same

**Expected value:** The mean of the posterior distribution. This represents the expected value taking into account the pseudo counts which corporate our initial belief about the situation. We can adjust our level of confidence in this prior belief by increasing the magnitude of the pseudo counts. This forces the expected values closer to our initial belief that the prevalence of each species is equal. For Dirichlet-Multinomial, it is $$E[p\_i|mathcalX, alpha]=fracc\_i+a\_iN+sum\_k alpha\_k$$

```python
species = ['lions', 'tigers', 'bears']

# Observations
c = np.array([3, 2, 1])

# Pseudocounts - our prior belief of the likelihood of events
alphas = np.array([1, 1, 1])

# Expected values of posterior
expected = (alphas + c) / (c.sum() + alphas.sum())
```

#### PyMC3

A good intro to pymc3
[Intro to PYMC3][28]

The toy example only allows us to get a point estimate using the posterior calculation. To get a range of estimates, we use Bayesian inference by constructing a model of the situation and then sampling from the posterior to approximate the posterior. This is implemented through Markov Chain Monte Carlo (or a more efficient variant called the No-U-Turn Sampler) in PyMC3.

```python
import pymc3 as pm
import numpy as np

alphas = np.array([1, 1, 1])
c = np.array([3, 2, 1])

with pm.Model() as model:

    # Parameters of the Multinomial are from a Dirichlet
    parameters = pm.Dirichlet('parameters', a=alphas, shape=3)

    # Observed data is from a Multinomial distribution
    observed_data = pm.Multinomial(
        'observed_data', n=6, p=parameters, shape=3, observed=c)

    # sample from prior
    trace = pm.sample(draws=1000, chains=2, tune=500,
      *discard_tuned_samples=True)
```

```python
import pandas as pd
import numpy as np

import pymc3 as pm
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot
%matplotlib inline

df = pd.read_csv('data3.csv')  # replace with random observation
df.head()

np.random.seed(109)
with pm.Model() as model:
   # Set priors for unknown model parameters
   alpha = pm.Normal('alpha',mu=0,tau=1000)

   # Likelihood (sampling distribution) of observations
   tau_obs = pm.Gamma('tau', alpha=0.001, beta=0.001)
   obs = pm.Normal('observed',
      *mu = alpha,
      *tau = tau_obs,
      *observed = df['y'])
   # create trace plots
   trace = pm.sample(2000, tune=2000)
   pm.traceplot(trace, compact=False);

# posterior means
np.mean(trace['alpha']) , np.mean(trace['tau'])
```

### Naive Bayes

Example

$P(textSunny|textYes)=3/9=0.33$

$P(textSunny)=5/14=0.36$

$P(textYes)=9/14=0.64$

$P(textYes|textSunny)=P(textSunny|textYes)times P(textYes)/P(textSunny)$

$P(textYes|textSunny)=0.33times 0.64/0.36=0.60$

$P(textNo|textSunny)=0.40$

- Applications
- Pro
  - It is easy and fast to predict the class of test data set. It also performs well in multi-class prediction
  - When the assumption of independence holds, a Naive Bayes classifier performs better compared to other models like logistic regression and you need less training data.
  - It performs well for categorical input variables compared to the numerical variable(s). For numerical variables, a normal distribution is assumed (bell curve, which is a strong assumption).
- Con
  - If the categorical variable has a category (in the test data set), which was not observed in the training data set, then the model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as “Zero Frequency”. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.
  - On the other side naive Bayes is also known as a bad estimator, so the probability outputs from `predict_proba` are not to be taken too seriously.
  - Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors that are completely independent.

### GAM

[Stitchfix][29]

[ML Notebook from Christoper][30]

### kNN

#### Algorithm

Search for linear or nonlinear boundaries that optimally separate the data. These boundaries are then used to predict the classification of new samples. “Closeness” is determined by a distance metric, like Euclidean and Minkowski, and choice of metric depends on predictor characteristics.

- Decide on a distance metric (e.g., Euclidean distance, 1 - correlation, etc.) and find the distances from each point in the test set to each point in the training set. The distance is measured in the feature space, that is, with respect to the explanatory variables (not the response variable).n.b. In most machine learning algorithms that use “distance” as a measure, the “distance” is not required to be a mathematical distance metric. Indeed, 1-correlation is a very common distance measure, and it fails the triangle inequality.
- Consider a point in the training set. Find the k closest points in the test set to the one test observation.
- Using majority vote, find the dominant class of the k closest points. Predict that class label to the test observation. Values for k are typically odd to prevent ties. If the response variable is continuous (instead of categorical), find the average response variable of the k training point to be the predicted response for the one test observation.

#### Preprocessing

- Normalizing the data is a method that allows giving every attribute the same influence in identifying neighbors when computing certain types of distances like the Euclidean one. You should normalize your data when the scales have no meaning and/or you have inconsistent scales like centimeters and meters. It implies prior knowledge of the data to know which one is more important. The algorithm automatically normalizes the data when both numeric and categorical variable is provided.

#### Hyperparameters

```python
# define model and parameters
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']

# define grid search
grid = dict(kernel = kernel, C = C, gamma = gamma)
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
grid_search = GridSearchCV(estimator = model, param_grid = grid, n_jobs = -1, cv = cv,

      *scoring = 'accuracy', error_score = 0)

grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

- K: The number of neighbors to look for. It is recommended to take an odd k for binary classes to avoid ties.
  - A low k will increase the influence of noise and the results are going to be less generalizable.
  - A high k will tend to blur local effects which are exactly what we are looking for.
- Aggregation method: The aggregation method to use. Here we allow for the arithmetic mean, median, and mode for numeric variables and mode for categorical ones.
- Distance metric:
  - Numeric attribute distances: among the various distance metrics available, we will focus on the main ones, Euclidean and Manhattan. Euclidean is a good distance measure to use if the input variables are similar in type (e.g. all measured widths and heights). Manhattan distance is a good measure to use if the input variables are not similar in type (such as age, height, etc…).
	- Euclidean
	- 1 - correlation
	- Cosine
	- manhattan
	- Minkowski
	- Haversine distance (geographical)
  - Categorical attribute distances: without prior transformation, applicable distances are related to frequency and similarity. Here we allow the use of two distances: Hamming distance and the Weighted Hamming distance.
	- Hamming distance: take all the categorical attributes and for each, count one if the value is not the same between two points. The Hamming distance is then the number of attributes for which the value was different.
	- Weighted Hamming distance: also return one if the value is different, but returns the frequency of the value in the attribute if they are matching, increasing the distance when the value is more frequent. When more than one attribute is categorical, the harmonic mean is applied. The result remains between zero and one but the mean value is shifted toward the lower values compared to the arithmetic mean.
  - Binary attribute distances: those attributes are generally obtained via categorical variables transformed into dummies. As already mentioned for the continuous variables, the Euclidean distance can also be applied here. However, there is also another metric based on dissimilarity that can be used, the Jaccard distance.
	- Jaccard distance

#### Trade Offs

- Pro
  - Can produce decent predictions, especially when the response is dependent on the local predictor structure.
  - it can easily work for any number of categories
  - it can predict a quantitative response variable
  - the bias of 1-NN is often low (but the variance is high)
  - any distance metric can be used (so the algorithm models the data appropriately)
  - the method is simple to implement/understand
  - model is nonparametric (no distributional assumptions on the data)
  - great model for imputing missing data
- Con
  - Euclidean distance is dominated by scale. Need to standardize. The distance value between samples will be biased towards predictors with larger scales. To allow each predictor to contribute equally to the distance calculation, we recommend centering and scaling all predictors before performing KNN.
  - One class can dominate if it has a large majority
  - It can be computationally unwieldy (and unneeded!!) to calculate all distances (there are algorithms to search smartly)
  - the output doesn’t provide any information about which explanatory variables are informative.
  - Any method with tuning parameters can be prone to overfitting, and KNN is especially susceptible to this problem. Too few neighbors lead to highly localized fitting (i.e., over-fitting), while too many neighbors lead to boundaries that may not locate necessary separating structure in the data. Therefore, we must take the usual cross-validation or resampling approach for determining the optimal value of K.
  - The new sample’s predicted class is the class with the highest probability estimate; if two or more classes are tied for the highest estimate, then the tie is broken at random or by looking ahead to the K + 1 closest neighbor. As the number of neighbors increases, the probability of ties also increases.
  - The KNN method can have poor predictive performance when the local predictor structure is not relevant to the response. Irrelevant or noisy predictors are one culprit since these can cause similar samples to be driven away from each other in the predictor space. Hence, removing irrelevant, noise-laden predictors is a key pre-processing step for KNN. Another approach to enhancing KNN predictivity is to weight the neighbors’ contribution to the prediction of a new sample based on their distance to the new sample. In this variation, training samples that are closer to the new sample contribute more to the predicted response, while those that are farther away contribute less to the predicted response.

#### Applications

##### Page Ranking/Document Retrieval

[Link][31] to incorporate

- TF-IDF model (basically a kNN)

  - Can be used to compare the similarity between documents and retrieve the relevant ones.
  - We used the vector space model, which entails assigning to each executable (i.e., document) a vector of size equal to the total number of distinct n-grams (i.e., terms) in the collection. The components of each vector were the weights of the top n-grams present in the executable. For the $j$th n-gram of the ith executable, the method computes the weight $w\_ij$, defined as $w\_ij = tf\_ij
	times idf\_j$
	- $tf\_ij$, the number of times the ith n-gram appears in the jth executable
	- $idf\_j = log
		fracddf\_j$, where $d$ is the total number of executables and $df\_j$ is the number of executables that contain the $j$th n-gram.
	- Use log because it is not necessarily the case that the less the occurrence of a term across documents, the more relevant a term is no matter the number of documents the term is in - sub-linear function better approximates this relationship.
  - Intuition: A high weight in TF-IDF is reached by a high term frequency (in the given document) and a low document frequency of the term in the whole collection of documents; the weights hence tend to filter out common terms.
  - To classify an unknown instance, the method uses the top n-grams from the executable, as described previously, to form a vector, u, the components of which are each n-gram’s inverse document frequency (i.e., $u\_j = idf\_j$).
  - Once formed, the classifier computes a similarity coefficient (SC) between the vector for the unknown executable and each vector for the executables in the collection using the cosine similarity measure:

  - After selecting the top five closest matches to the unknown, the method takes a weighted majority vote of the executable labels, and returns the class with the least weight as the prediction. It uses the cosine measure as the weight.

##### Imputation

It can be used for data that are continuous, discrete, ordinal, and categorical which makes it particularly useful for dealing with all kinds of missing data.

The assumption behind using KNN for missing values is that a point value can be approximated by the values of the points that are closest to it, based on other variables.

Let’s keep the previous example and add another variable, the income of the person. Now we have three variables, gender, income, and the level of depression that has missing values. We then assume that people of similar income and same gender tend to have the same level of depression. For a given missing value, we will look at the gender of the person, its income, look for its k nearest neighbors, and get their level of depression. We can then approximate the depression level of the person we wanted.

## Model Explainability

### Permutation Importance

 [ Permutation Importance Code Example][32]

For single trees, variable importance can be determined by aggregating the improvement in the optimization objective for each predictor. For random forests, the improvement criteria (default is typically the Gini Impurity) is aggregated across the ensemble to generate an overall variable importance measure. Alternatively, predictors’ impact on the ensemble can be calculated using a permutation approach.

Variable importance can be measured by two different metrics

- (permutation) accuracy:
  - For each tree, the prediction error on the out-of-bag portion of the data is recorded (error rate for classification, MSE for regression).
  - Within the OOB values, permute the jth variable and recalculate the prediction error.
  - The difference between the two is then averaged over all trees (with the jth variable) to give the importance for the jth variable.
- Purity:
  - The decrease (or increase, depending on the plot) in node purity: root sum of squares (RSS) [deviance/Gini for classification trees]. That is, the amount of total decrease in RSS from splitting on that variable averaged over all trees.

Difference between Variable Importance and Latent Variable Importance

Often, especially in the case of GBT, two variants of the correlated variables are included in tree models to achieve a better fit. This may affect variable inferences that solely focus on either variant. Shapley values of the correlated siblings could be added to better approximate the Shapley values of the latent feature.

### Shapley Values

### LIME

### Partial Dependence

## Clustering

### K-means

- Intuition
  - In words, this formula says that we want to partition the observations into K clusters such that the total within-cluster variation, summed over all K clusters, is as small as possible.
  - To make it actionable we need to define the within-cluster variation. There are many possible ways to define this concept, but by far the most common choice involves squared [Euclidean distance](\\\#distance metrics).
  - Therefore,
- Algorithm
  - Guaranteed to decrease the value of the objective function at each step.
  - The algorithm will continually improve until the result no longer changes.
- Notes
  - Because the K-means algorithm finds a local rather than a global optimum, the results obtained will depend on the initial (random) cluster assignment of each observation in Step 1of Of Algorithm 10.1. For this reason, it is important to run the algorithm multiple times from different random initial configurations. Then one selects the best solution, i.e. that for which the objective function is smallest.
  - Presence of outliers: For instance, suppose that most of the observations truly belong to a small number of (unknown) subgroups, and a small subset of the observations are quite different from each other and all other observations. Then since Kmeans and hierarchical clustering force every observation into a cluster, the clusters found may be heavily distorted due to the presence of outliers that do not belong to any cluster. Mixture models are an attractive approach for accommodating the presence of such outliers.
  - Not very robust to perturbations to the data: Also, clustering methods generally are not very robust to perturbations to the data. For instance, suppose that we cluster n observations, and then cluster the observations again after removing a subset of the n observations at random. One would hope that the two sets of clusters obtained would be quite similar, but often this is not the case!
  - Small decisions in how clustering is performed, such as how to standardize data and what linkage to use can have a large effect on the results. Therefore, we recommend performing clustering with different choices of these parameters and looking at the full set of results to see what patterns consistently emerge. Since clustering can be non-robust, we recommend clustering subsets of the data to get a sense of the robustness of the clusters obtained. Most importantly, we must be careful about how the results of a clustering analysis are reported. These results should not be taken as the absolute truth about a data set. Rather, they should constitute a starting point for the development of a scientific hypothesis and further study, preferably on an independent data set.
- Application to outlier detection
  - Calculate distance
  - Calculate the mean and standard deviation
  - Calculate the p-value to determine whether it’s significantly different from the null hypothesis
  - Calculate as an outlier
  - Decision: whether categorize as a global or local outlier. This depends on domain knowledge on the result k clusters and their distinctions from each other.

### PCA

This approach involves projecting the p predictors into an M-dimensional subspace, where M ? p. This is achieved by computing M different linear combinations, or projections, of the variables. Then these M projections are used as predictors to fit a linear regression model by least squares.

Intuition:

- Often, a small number of principal components are enough to explain most of the variability in the data. We assume that the directions in which X1, . . . , Xp show the most variation are the directions that are associated with Y.
- PCA is mathematically defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.
- The second component $Z\_2$ should have zero correlation with $Z\_1$. This is equivalent to saying that the direction must be perpendicular/orthogonal.
- Better than OLS because:
  - If the assumption underlying PCR holds, then fitting a least-squares model to Z1, . . . , ZM will lead to better results than fitting a least-squares model to X1, . . . , Xp, since most or all of the information in the data that relates to the response is contained in Z1, . . . , ZM, and by estimating only M`<`p coefficients we can mitigate overfitting.
- Not a feature selection method because each of the M principal components used in the regression is a linear combination of all p of the original features.
- Similar to Ridge.
- The number of principal components, M, is typically chosen by cross-validation.
- Standardize: When performing PCR, we generally recommend standardizing each predictor, using (6.6), before generating the principal components. This standardization ensures that all variables are on the same scale. In the absence of standardization, the high-variance variables will tend to play a larger role in the principal components obtained, and the scale on which the variables are measured will ultimately affect the final PCR model. However, if the variables are all measured in the same units (say, kilograms, or inches), then one might choose not to standardize them.

PCR will tend to do well in cases when the first few principal components are sufficient to capture most of the variation in the predictors as well as the relationship with the response.

Algorithm

- Construct the first M principle components Z1, … Zm.
  - Z1 = 0.839 × (pop – pop\_bar) + 0.544 × (ad – ad\_bar).
	- Where 0.839^2 + 0.544^2 = 1, and this linear combination is generated in an unsupervised way, in the sense that it was generated without knowing the dependent variable.
	- Out of every possible linear combination of pop and ad such that a^2+b^2 = 1, this particular linear combination yields the highest variance.
	- Another interpretation – the first PC vector defines the line that is as close as possible to the data.
- Use these components as the predictors in a linear regression model that is fit using least square. The first principal component line minimizes the sum of the squared perpendicular distances between each point and the line.
  Check the importance of the features and how to plot a biplot

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target
# In general a good idea is to scale the data
scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

pca = PCA()
x_new = pca.fit_transform(X)

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r', alpha = 0.5)
        if labels is None:
      *plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
      *plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

# Call the function. Use only the 2 PCs.
myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()
```
## Model Comparisons

### Cross-validation

We can compute the validation set error or the cross-validation error for each model under consideration, and then select the model for which the resulting estimated test error is smallest. See error metrics.

#### Tuning

1. set k in kNN
2. build the model using the k value set above
   1. remove 10% of the data
   2. build the model using the remaining 90%
   3. predict class membership / continuous response for the 10% of the observations which were removed
   4. repeat by removing each decile one at a time
3. measure the CV prediction error for the k value at hand
4. repeat steps 1-3 and choose the k for which the prediction error is the lowest

#### Tuning & Assessment

1. Partition the data in K1 groups.
2. Remove the first group and train the data on the remaining K1 - 1 groups.
3. Divide the training observations into K2 folds and use CV to find $alpha$ that minimizes the error.
4. Using the subtree that corresponds to the chosen value of $alpha$, predict the first of the K1 hold out samples.
5. Repeat steps 2-4 using the remaining K1 - 1 groups.

### Classification Performance Metrics

#### Overall Classification Metrics

Metric Evaluation Based On Tip

MCC Class good for imbalanced data

Matthews's correlation coefficient. Come from bioinformatics.

The coefficient takes into account true and false positives and negatives and is generally regarded as a balanced measure that can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient between the observed and predicted binary classifications; it returns a value between −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better than random prediction and −1 indicates total disagreement between prediction and observation. The statistic is also known as the phi coefficient.

[Wikipedia][33]

- F1 Class
- F0.5 Class good when you want to give more weight to precision
- F2 Class good when you want to give more weight to recall
- Accuracy Class highly interpretable
- Logloss Probability
- AUC Class
  - Sensitivity vs. Specificity
- AUCPR Class good for imbalanced data
  - Precision $TP/(TP+FP)$ vs. Recall $TP/(TP+FN)$ incorporates all possible class combinations.

#### F1-score vs ROC/AUC

[link][34]

- Using ROC/AUC: As a rule of thumb, if the cost of having False negative is high, we want to increase the model sensitivity (recall). We want to increase the proportion of actually positive populations that are categorized into true positive, relative to a false negative.
  - False negatives are worse
  - Sensitivity vs Specificity (True Negative Rate)
- Using F1: On the other hand, if the cost of having False Positive is high, then we want to increase the model specificity and precision!.
  - False positives are worse (falsely labeled an email as spam for example)
  - Precision vs Recall

#### ROC / AUC (Sensitivity vs. Specificity)

- Varying the classifier threshold changes its true positive and false positive rate. These are also called the sensitivity and (1 - specificity) of our classifier. They characterize the performance of a classifier.
- The ROC curve simultaneously displays the two types of errors for all possible thresholds. The overall performance of a classifier, summarized over all possible thresholds, is given by the area under the (ROC) curve (AUC). An ideal ROC curve will hug the top left corner, so the larger area under the (ROC) curve the AUC the better the classifier.
- ROC curves are useful for comparing different classifiers since they consider all possible thresholds.
- ROC is a graphical plot that illustrates the performance of a binary classifier (Sensitivity vs. 1- Specificity = FP rate vs. FP rate). Not sensitive to unbalanced classes.
  - One often overlooked aspect of sensitivity and specificity is that they are conditional measures. Sensitivity is the accuracy rate for only the event population (and specificity for the non-events). Using the sensitivity and specificity, the obstetrician can make statements such as “assuming that the fetus does not have Down syndrome, the test has an accuracy of 95 %.” However, these statements might not be helpful to a patient since, for new samples, all that is known is the prediction. The person using the model prediction is typically interested in unconditional queries such as “what are the chances that the fetus has the genetic disorder?” This depends on three values: the sensitivity and specificity of the diagnostic test and the prevalence of the event in the population.
- AUC is the area under the ROC curve. Perfect classifier: AUC=1, fall on (0,1); 100% sensitivity (no FN) and 100% specificity (no FP)

False-positive: Improperly reporting the presence of a condition when it’s not in reality. Example: HIV positive test when the patient is actually HIV negative

False-negative: Improperly reporting the absence of a condition when in reality it’s the case. Example: not detecting a disease when the patient has this disease.

- FP is worse than FN
  - It is better that ten guilty persons escape than that one innocent suffer.
- FN is worse than FP
  - If early treatment is important for good outcomes
  - Software testing: a test to catch a virus has failed
- Medical trial/credit card fraud, reduce false-negative rate.

#### Balanced Accuracy (Class Imbalance)

$frac(text{Sensitivity + textSpecificity)}2$

#### Precision & Recall

Understanding and measure of relevance.

high recall + high precision: the class is perfectly handled by the model

low recall + high precision: the model can’t detect the class well but is highly trustable when it does

high recall + low precision: the class is well detected but the model also includes points of other classes in it

low recall + low precision: the class is poorly handled by the model

#### Cross Entropy

### Regression Performance Metrics

- RMSE: gives a relatively high weight to large errors, most useful when large errors are particularly undesirable.
- MAE: all the individual differences are weighted equally in the average, thus more robust to outliers than MSE.

#### AIC, BIC, and Adjusted R2

- Why do we need them?
  - Training set RSS or $R^2$ cannot be used to select from among a set of models with different numbers of variables because the training error will decrease as more variables are included in the model, but the test error may not.
  - Therefore, we need to use these metrics to adjust the training error for the model sizes (specifications) available. This adjustment helps us select among a set of models with different numbers of variables.
- The intuition behind the adjusted R2 is that once all of the correct variables have been included in the model, adding additional noise variables will lead to only a very small decrease in RSS. Therefore, in theory, the model with the largest adjusted R2 will have only correct variables and no noise variables.
- AIC, BIC can also be defined for more general types of models. The adjusted R2 is not as well motivated in statistical theory as AIC, BIC, and Cp. The best model has the lowest AIC and BIC.
- R-squared: the proportion of the information in the data that’s explained by the model. 0.75 means that 3-quarter of the variation can be explained. It’s a measurement of correlation, not accuracy.

The performance of a model here depends on the variation of the outcome – the less variable the outcome is, the lower is the value of SS Total, and lower is R squared, given that RMSE is the same.

- In either case, R2 indicates the proportion of variation in the y-variable that is due to variation in the x-variables.
- For simple linear regression, R2 is just the square of the sample correlation.
- Adjusted R2 penalizes for having a large number of parameters:

$R^2 = 1-
fracSS_{res}SS_{total}$

$textAdjusted  R^2 = 1-fracfrac{SS_{res}n - k}frac{SS_{total}n - 1}$

[1]:	https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf
[2]:	https://www.youtube.com/watch?v=taA3r7378gU&list=PLPW2keNyw-usgvmR7FTQ3ZRjfLs5jT4BO&t=664s
[3]:	https://www.coursera.org/lecture/deep-neural-network/rmsprop-BhJlm
[4]:	http://cs229.stanford.edu/notes/cs229-notes4.pdf
[5]:	https://en.wikipedia.org/wiki/Vector_Space_Model
[6]:	https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize
[7]:	https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
[8]:	https://towardsdatascience.com/understand-data-normalization-in-machine-learning-8ff3062101f0
[9]:	https://towardsdatascience.com/handling-missing-values-in-machine-learning-part-1-dda69d4f88ca
[10]:	https://towardsdatascience.com/handling-missing-values-in-machine-learning-part-2-222154b4b58e
[11]:	https://medium.com/coinmonks/smote-and-adasyn-handling-imbalanced-data-set-34f5223e167
[12]:	https://www.reed.edu/economics/parker/s11/312/notes/Notes2.pdf
[13]:	https://stats.stackexchange.com/questions/149110/assumptions-to-derive-ols-estimator/149111#149111
[14]:	https://stats.stackexchange.com/questions/149226/does-linear-regression-assume-all-variables-predictors-and-response-to-be-mult
[15]:	https://stats.stackexchange.com/questions/148803/how-does-linear-regression-use-the-normal-distribution
[16]:	https://www.wikiwand.com/en/Gauss%E2%80%93Markov_theorem
[17]:	http://www3.amherst.edu/~fwesthoff/webpost/Old/Econ_360/Econ_360-11-14-Chap.pdf
[18]:	https://en.wikibooks.org/wiki/Econometric_Theory/Serial_Correlation#Causes_of_Autocorrelation
[19]:	https://www.stata.com/features/overview/fractional-polynomials/
[20]:	http://irl.cs.brown.edu/fb.php
[21]:	http://dept.stat.lsa.umich.edu/~kshedden/Courses/Stat504/posts/regression_overview
[22]:	https://stats.idre.ucla.edu/stata/webbooks/logistic/chapter3/lesson-3-logistic-regression-diagnostics/
[23]:	https://www.statsmodels.org/stable/mixed_linear.html
[24]:	https://docs.google.com/document/d/1e9dfm3-JHy9JE5eRrcW1YUVo09ORPjnnozq05i_aBZ4/edit#
[25]:	https://en.wikipedia.org/wiki/Limited_dependent_variable
[26]:	https://en.wikipedia.org/wiki/Generalized_linear_model
[27]:	https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
[28]:	https://juanitorduz.github.io/intro_pymc3/
[29]:	https://multithreaded.stitchfix.com/assets/files/gam.pdf
[30]:	https://christophm.github.io/interpretable-ml-book/extend-lm.html
[31]:	https://www.geeksforgeeks.org/tf-idf-model-for-page-ranking/
[32]:	https://www.kaggle.com/dansbecker/permutation-importance#Code-Example
[33]:	https://www.wikiwand.com/en/Matthews_correlation_coefficient
[34]:	https://stackoverflow.com/questions/44172162/f1-score-vs-roc-auc