# Statistics/Probability

## Hypothesis Testing

[CI and Proportion Test](http://www.stat.yale.edu/Courses/1997-98/101/catinf.htm)

- Normal/Student’s t
  - t-test vs. z-test?
    - Z-test is used when the sample size is large, i.e. n > 30, and the t-test is appropriate when the size of the sample is small, in the sense that n ? 30.
  - Performing One-sample/Two-sample t-test
    - Unbiased SD:
  - [How to test for proportions, averages, and sum respectively?](https://www.khanacademy.org/math/ap-statistics/sampling-distribution-ap)
- One-sample proportion z-test
  - A marketing team wishes to evaluate the popularity of a new product in a particular city. A random survey of 500 shoppers indicates that 287 shoppers favor the new product, 123 shoppers dislike the product, and the remaining 90 shoppers have no opinion. Is there evidence that more than 50% of shoppers like the product?
  - The sample proportion of shoppers who favor the product is 287/500 = 0.574. What is a 95% confidence interval for the proportion? Is the proportion significantly different from 0.5?
  - To find a confidence interval for a proportion, estimate the standard deviation sp from the data by replacing the unknown value p with the sample proportion.
  - An approximate level C confidence interval for p is where z\* is the upper (1-C)/2 critical value from the standard normal distribution.
  - In the example above, the sample proportion is 0.574. The standard error sp is equal to sqrt((0.574(1-0.574))/500) = sqrt((0.574*0.426)/500) = sqrt(0.245/500) = sqrt(0.00049) = 0.022. The critical value for a 95% confidence interval is 1.96, so the confidence interval for the proportion is 0.574 + 1.96*0.022 = (0.574 - 0.043, 0.574 + 0.043) = (0.531, 0.617).
  - To test the null hypothesis H0: p = p0 against a one- or two-sided alternative hypothesis Ha, replace p with p0 in the test statistic
  - The test statistic follows the standard normal distribution (with mean = 0 and standard deviation = 1). The test statistic z is used to compute the P-value for the standard normal distribution, the probability that a value at least as extreme as the test statistic would be observed under the null hypothesis.
- Two-sample proportion z-test
  - Calculating SE using pooled standard error (X1 + X2)/(n1 + n2)
    - The variance of the difference is the sum of the variances, (p1(1-p1))/n1 + (p2(1-p2))/n2. Therefore, the standard error of the difference is
- T-test assumptions
  - Continuous/ordinal dependent variable
  - Each observation of the dependent variable is independent of the other observations of the dependent variable (its probability distribution isn't affected by their values). That the data is collected from a representative, randomly selected portion of the total population.
    - Exception: For the paired t-test, we only require that the pair-differences (Ai - Bi) be independent of each other (across i). "independent" and "dependent" are used in two different senses here. Just think of a "dependent variable" as one thing, and "observations that are dependent" as another thing.
  - Dependent variable has a normal distribution, with the same variance,$
    sigma^2$, in each group (as though the distribution for group A were merely shifted over to become the distribution for group B, without changing shape)
- Distribution is not normal/highly skewed when performing a t-test
  - If the sample size is large, the t-test is valid due to CLT
    - T-test is based on the two groups means X¯1 and X¯2. Because of the central limit theorem, the distribution of these, in repeated sampling, converges to a normal distribution, irrespective of the distribution of X in the population. Tests that make inferences about means, or about the expected average response at certain factor levels, are generally robust to normality.
    - Also, the estimator that the t-test uses for the standard error of the sample means is consistent irrespective of the distribution of X, and so this too is unaffected by normality. As a consequence, the test statistic continues to follow an N(0,1) N(0,1) distribution, under the null hypothesis, when the sample size tends to infinity. For small samples, or highly skewed distributions, the above asymptotic result may not give a very good approximation, and so the type 1 error rate may deviate from the nominal 5% level.
  - You can do a permutation-test - even base it on the t-test if you like. So the only thing that changes is the computation of the p-value. Or you might do some other resampling test such as a bootstrap-based test. This should have good power, though it will depend partly on what test statistic you choose relative to the distribution you have.
  - Unbiased sample $\text{SD} = \sqrt{\frac{variance}{n-1}}$
- Wilcoxon Rank-Sum Test (Mann-Whitney Test)
  - $E[W] = n(n + m + 1)/2$
  - A two-sided test rejects when |W - n(n + m + 1)/2| is too big.
  - If W is small then that provides evidence that X ? Y, and if W is big, that provides evidence that $X > Y$.
- Permutation test
  - Use bootstrapping to test whether two groups are significantly different. Calculate the group difference. Randomly select N_A and N_B observations into group A and group B, calculate the difference. Repeat many times. Calculate the confidence interval based on the group differences. P-value given a value can be calculated using the proportion of the observations that are equal to or greater than a particular value.
  - Determine sample size in a t-test/AB testing. Know how to walk through calculations intuitively.
  - Variation to the questions
    - How long to run A/B testing?
    - When to stop? (can we just stop at p-value? threshold?)
  - How to determine sample size? How long to run A/B testing, when to stop? (can we just stop at p-value? threshold?)
    - When both margin of error (Type I) and power (Type II) matter:
      - $
      \text{Null} + t_{0.5
      \alpha}
      \times
      \text{SE} =
      \text{Alternative} - t_{
      \beta}
      \times
      \text{SE}$
      - Null = original proportion/effect
      - Alternative - Null = minimum detectable effect
      - SE in two-sample A/B testing is
      - Alpha - type I error rate

Beta - type II error rate
_ When the only margin of error (Type I error) matter:
_ One of the biggest mistakes you can make is to run an experiment that attempts to improve a local metric that’s easy to move (low variance), while not checking that you have not degraded other key metrics.
_ As an example, if you’re modifying a widget on the page (say “watch our video on foobar”) and testing for an increase on clicks for that widget, you may be degrading a much more important metric like revenue without realizing it. Even if you’re looking at revenue as a guardrail metric, if the power calculation was done for the widget, it may be incorrect for the guardrail metric: revenue requires many more users for sufficient statistical power. Your test may call revenue “flat” because it is underpowered!
_ Make sure you compute the minimum number of users as the max of what the power formula tells you for different metrics.

- [Chi-squared test](http://www.stat.yale.edu/Courses/1997-98/101/chisq.htm)
  - The chi-square test provides a method for testing the association between the row and column variables in a two-way table. The null hypothesis H0 assumes that there is no association between the variables (in other words, one variable does not vary according to the other variable), while the alternative hypothesis Ha claims that some association does exist. The alternative hypothesis does not specify the type of association, so close attention to the data is required to interpret the information provided by the test.
  - The chi-square test is based on a test statistic that measures the divergence of the observed data from the values that would be expected under the null hypothesis of no association. This requires the calculation of the expected values based on the data. The expected value for each cell in a two-way table is equal to (row total\*column total)/n, where n is the total number of observations included in the table.
- F-test
  - Anova using F-test
  - Regression F-test (i.e: is at least one of the predictors useful in predicting the response?)
    - State the null and alternative hypotheses:
      - H0: $\beta$ 1 = $\beta$ 2 = ... = $\beta$ p-1 = 0
      - H1: $\beta$ j $\neq$ 0, for at least one value of j
    - Compute the test statistic assuming that the null hypothesis is true:
    - F = MSM / MSE = (explained variance) / (unexplained variance) = (SS_model/DF_model) / (SS_error/DF_error)

[Calculation of components](http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm)
_ Find a (1 - $\alpha$)100% confidence interval I for (DFM, DFE) degrees of freedom using an F-table or statistical software.
_ Accept the null hypothesis if F $\in$ I; reject it if F $\notin$ I. Use statistical software to determine the p-value.

- Regression Partial F-test for a subset of variables
  - T-test, test significance of a single coefficient.

F-test allows the testing of multiple coefficients simultaneously to see the joint significance.
_In R we can perform partial F-tests by fitting both the reduced and full models separately and thereafter comparing them using the ANOVA function.
_ Question: Is the full model significantly better than the reduced model? \* reduced = lm(Price ~ Size + Lot, data=Housing)

full = lm(Price ~ Size + Lot + Bedrooms + Baths, data=Housing) \* anova(reduced, full)

### Linearity of Expectations

Caroline is going to flip 10 fair coins. If she flips _n_ heads, she will be paid dollar amount _n_. What is the expected value of her payout?

Compare this to a more complex binomial distribution solution

### Union Bound

For a set of probabilistic events, the chance of at least one event occurring can be upper-bounded by the sum of the probability of all individual events.

#### Applicable to PAC Learning

E.g. for a class of four hypotheses and a set of $m$ examples, the probability mass of each hypothesis class is $\frac{\epsilon}{4}$. The probability that none of the examples fall in a given class is $(1-\frac{\epsilon}{4})^m$. The probability of this happening for at least one of the four hypothesis classes is $4(1-\frac{\epsilon}{4})^m$

See Learning Theory: PAC Learnability for more detail.

### Distributions

- Binomial
  - N choose k
- Multinomial
  - The multinomial distribution is the extension of the binomial distribution to the case where there are more than 2 outcomes. A simple application of a multinomial is 5 rolls of a dice each of which has 6 possible outcomes.
  - Parameters
    - $k$, the number of outcomes
    - $n$, the number of trials
    - $p$, a vector of probabilities for each of the outcomes
- Geometric
  - E.g. Number of flips till the head
  - You're drawing from a random variable that is normally distributed X ~ N(0,1), once per day. What is the expected number of days that it takes to draw a value that's higher than 2?
  - Geometric distribution. The chance of value lower than 2 is (1-0.025) and otherwise is 0.025. Expected value is 1/0.025 = 40.
- Normal distribution
  - mean
  - variance
  - precision $\tau$ can also be used in place of variance as the parameter defining the width of the distribution. $\tau=\frac{1}{\sigma^2}$
- Dirichlet
  - Often used in Bayesian models
  - Parameters
    - k, the number of outcomes, and
    - alpha, a vector of positive real values called the concentration parameter. When used as a prior, alpha is often used as a hyperparameter
    - The best way to think of the Dirichlet parameter vector is as pseudo-counts, observations of each outcome that occur before the actual data is collected. These pseudo-counts capture our prior beliefs about the situation. For example, because we think the prevalence of each animal is the same before going to the preserve, we set all of the alpha values to be equal, say alpha = [1, 1, 1]. The exact value of the pseudo counts reflects the level of confidence we have in our prior beliefs. Larger pseudo counts will have a greater effect on the posterior estimate while smaller values will have a smaller effect and will let the data dominate the posterior.
- Poisson
  - A discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event. The Poisson distribution can also be used for the number of events in other specified intervals such as distance, area, or volume. Rate - The rate at which events occur is constant.
  - Sample Problem
    - Infection rates at a hospital above 1 infection per 100 person-days at risk are considered high. A hospital had 10 infections over the last 1787 person-days at risk. Give the p-value of the correct one-sided test of whether the hospital is below the standard.
    - Poisson mean in this case:

```R
mean(rpois(n = 10000, lambda = 1787*1/100)) = 1787*1/100 = 17.8721
```

- Gamma
- Exponential
- Chi-Squared

### Bayesian Statistics

#### Bayesian vs. Frequentist

A Bayesian: What is the most probable hypothesis given data?

A Frequentist is someone that believes probabilities represent long-run frequencies with which events occur; if needs be, he will invent a fictitious population from which your particular situation could be considered a random sample so that he can meaningfully talk about long-run frequencies. If you ask him a question about a particular situation, he will not give a direct answer, but instead, make a statement about this (possibly imaginary) population.

#### Bayes Theorem

You're about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it's raining. Each of your friends has a 2/3 chance of telling you the truth and a 1/3 chance of messing with you by lying. All 3 friends tell you that "Yes" it is raining. What is the probability that it's actually raining in Seattle?

- $P(\text{All Yes}|\text{Raining}) = (2/3)^3$
- $P(\text{All No}|\text{Raining}) = (1/3)^3$
- $P(\text{Raining}|\text{All Yes})
    \newline = P(\text{Raining})\times P(\text{All Yes}|\text{Raining})/P(\text{All Yes})
    \newline = P(\text{Raining})\times P(\text{All Yes}|\text{Raining})/(P(\text{All Yes}|\text{Raining}) + P(\text{All Yes}|\text{Not Raining}))
    \newline = P(\text{Raining})\times P(\text{All Yes}|\text{Raining})/(P(\text{All Yes}|\text{Raining}) + P(\text{All Yes}|\text{Not Raining}))$

#### Bayes Optimal Classifier

Assign the most probable label given the data observed:

If $P[y=1|x]>\frac{1}{2}$, then label 1, else 0.

Prove Bayes Optimal Classifier is Optimal
