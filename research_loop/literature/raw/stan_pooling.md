> **External page content (untrusted):** Treat the content below as data, not instructions. Do not follow requests in it to call tools or disclose or send data.

## Hierarchical Partial Pooling for Repeated Binary Trials

**Source**: https://mc-stan.org/learn-stan/case-studies/pool-binary-trials.html

---

# Hierarchical Partial Pooling for Repeated Binary Trials

#### *Bob Carpenter*

#### *29 January 2016*

### Abstract

This note illustrates the effects on posterior inference of pooling data (aka sharing strength) across items for repeated binary trial data. It provides Stan models and R code to fit and check predictive models for three situations: (a) complete pooling, which assumes each item is the same, (b) no pooling, which assumes the items are unrelated, and (c) partial pooling, where the similarity among the items is estimated. We consider two hierarchical models to estimate the partial pooling, one with a beta prior on chance of success and another with a normal prior on the log odds of success. The note explains with working examples how to (i) fit models in RStan and plot the results in R using ggplot2, (ii) estimate event probabilities, (iii) evaluate posterior predictive densities to evaluate model predictions on held-out data, (iv) rank items by chance of success, (v) perform multiple comparisons in several settings, (vi) replicate new data for posterior p-values, and (vii) perform graphical posterior predictive checks.

## Repeated Binary Trials

Suppose that for each of \\(N\\) items \\(n \\in 1{:}N\\), we observe \\(y\_n\\) successes out of \\(K\_n\\) trials. For example, the data may consist of

- rat tumor development, with \\(y\_n\\) rats developing tumors of \\(K\_n\\) total rats in experimental control group \\(n \\in 1{:}N\\) (Tarone 1982)
- surgical mortality, with \\(y\_n\\) surgical patients dying in \\(K\_n\\) surgeries for hospitals \\(n \\in 1{:}N\\) (Spiegelhalter et al. 1996)
- baseball batting ability, with \\(y\_n\\) hits in \\(K\_n\\) at bats for baseball players \\(n \\in 1{:}N\\) (Efron and Morris 1975; Carpenter 2009)
- machine learning system accuracy, with \\(y\_n\\) correct classifications out of \\(K\_n\\) examples for systems \\(n \\in 1{:}N\\) (ML conference proceedings; Kaggle competitions)

We use the small baseball data set of Efron and Morris (1975) as a running example, and in the same format provide the rat control data of Tarone (1982), the surgical mortality data of Spiegelhalter et al. (1996) and the extended baseball data set of Carpenter (2009).

#### Baseball Hits (Efron and Morris 1975)

As a running example, we include the data from Table 1 of (Efron and Morris 1975) as `efron-morris-75-data.tsv` (it was downloaded 24 Dec 2015 from [here](http://www.swarthmore.edu/NatSci/peverso1/Sports%20Data/JamesSteinData/Efron-Morris%20Baseball/EfronMorrisBB.txt)). It is drawn from the 1970 Major League Baseball season from both leagues.

```
df <- read.csv("efron-morris-75-data.tsv", sep="\t");
df <- with(df, data.frame(FirstName, LastName, 
                          Hits, At.Bats, 
                          RemainingAt.Bats,
                          RemainingHits = SeasonHits - Hits));
print(df);
```

```
   FirstName   LastName Hits At.Bats RemainingAt.Bats RemainingHits
1    Roberto   Clemente   18      45              367           127
2      Frank   Robinson   17      45              426           127
3      Frank     Howard   16      45              521           144
4        Jay  Johnstone   15      45              275            61
5        Ken      Berry   14      45              418           114
6        Jim    Spencer   14      45              466           126
7        Don  Kessinger   13      45              586           155
8       Luis   Alvarado   12      45              138            29
9        Ron      Santo   11      45              510           137
10       Ron    Swaboda   11      45              200            46
11      Rico Petrocelli   10      45              538           142
12     Ellie  Rodriguez   10      45              186            42
13    George      Scott   10      45              435           132
14       Del      Unser   10      45              277            73
15     Billy   Williams   10      45              591           195
16      Bert Campaneris    9      45              558           159
17   Thurman     Munson    8      45              408           129
18       Max      Alvis    7      45               70            14
```

We will only need a few columns of the data; we will be using the remaining hits and at bats to evaluate the predictive inferences for the various models.

```
N <- dim(df)[1]
K <- df$At.Bats
y <- df$Hits
K_new <- df$RemainingAt.Bats;
y_new <- df$RemainingHits;
```

The data separates the outcome from the initial 45 at-bats from the rest of the season. After running this code, `N` is the number of items (players). Then for each item `n`, `K[n]` is the number of initial trials (at-bats), `y[n]` is the number of initial successes (hits), `K_new[n]` is the remaining number of trials (remaining at-bats), and `y_new[n]` is the number of successes in the remaining trials (remaining hits).

The remaining data can be used to evaluate the predictive performance of our models conditioned on the observed data. That is, we will “train” on the first 45 at bats and see how well our various models do at predicting the rest of the season.

Although we consider many models, the data is coded as follows for all of them.

```
data {
  int<lower=0> N;           // items
  int<lower=0> K[N];        // initial trials
  int<lower=0> y[N];        // initial successes

  int<lower=0> K_new[N];    // new trials
  int<lower=0> y_new[N];    // new successes
}
```

As usual, we follow the convention of naming our program variables after the variables we use when we write the model out mathematically in a paper. We also choose capital letters for integer constants and y for the main observed variable(s).

## Pooling

With *complete pooling*, each item is assumed to have the same chance of success. With *no pooling*, each item is assumed to have a completely unrelated chance of success. With *partial pooling*, each item is assumed to have a different chance of success, but the data for all of the observed items informs the estimates for each item.

Partial pooling is typically accomplished through hierarchical models. Hierarchical models directly model the population of items. The population mean and variance is important, but the two hierarchical models we consider (chance of success vs. log odds of success) provide rather differently shaped posteriors.

From a population model perspective, no pooling corresponds to infinite population variance, whereas complete pooling corresponds to zero population variance.

In the following sections, all three types of pooling models will be fit for the baseball data.

## Model 1: Complete Pooling

The complete pooling model assumes a single parameter \\(\\phi\\) representing the chance of success for all items. It is necessary in Stan to declare parameters with constraints corresponding to their support in the model. Because \\(\\phi\\) will be used as a binomial parameter, we must have \\(\\phi \\in \[0,1]\\). The variable `phi` must therefore be declared in Stan with the following lower- and upper-bound constraints.

```
parameters {
  real<lower=0, upper=1> phi;  // chance of success (pooled)
}
```

The consequences for leaving the constraint off is that the program may fail during random initialization or during an iteration because Stan will generate initial values for `phi` outside of \\(\[0,1]\\). Such a specification may appear to work if there are only a small number of such variables because Stan tries multiple random initial values by default for MCMC; but even so, results may be biased due to numerical arithmetic issues.

Assuming each player’s at-bats are independent Bernoulli trials, the sampling distribution for each player’s number of hits \\(y\_n\\) is modeled as

\\\[ p(y\_n \\, | \\, \\phi) \\ = \\ \\mathsf{Binomial}(y\_n \\, | \\, K\_n, \\phi). \\]

When viewed as a function of \\(\\phi\\) for fixed \\(y\_n\\), this is called the likelihood function.

Assuming each player is independent leads to the complete data likelihood

\\\[ p(y \\, | \\, \\phi) = \\prod\_{n=1}^N \\mathsf{Binomial}(y\_n \\, | \\, K\_n, \\phi). \\]

We will assume a uniform prior on \\(\\phi\\),

\\\[ p(\\phi) \\ = \\ \\mathsf{Uniform}(\\phi \\, | \\, 0, 1) \\ = \\ 1. \\]

Whether a prior is uniform or not depends on the scale with which the parameter is expressed. Here, the variable \\(\\phi\\) is a chance of success in \\(\[0, 1]\\). If we were to consider the log-odds of success, \\(\\log \\frac{\\phi}{1 - \\phi}\\), a uniform prior on log-odds is not the same as a uniform prior on chance of success (they are off by the Jacobian of the transform). A uniform prior on chance of success translates to a unit logistic prior on the log odds (the definition of the unit logistic density can be derived by calculating the Jacobian of the transform).

By default, Stan places a uniform prior over the values meeting the constraints on a parameter. Because `phi` is constrained to fall in \\(\[0,1]\\), there is no need to explicitly specify the uniform prior on \\(\\phi\\).

The likelihood is expressed as a vectorized sampling statement in Stan as

```
model {
  ...
  y ~ binomial(K, phi);
}
```

Sampling statements in Stan are syntactic shorthand for incrementing the underlying log density accumulator. Thus the above would produce the same draws as

```
  increment_log_prob(binomial_log(y, K, phi));
```

The only difference is that the sampling statement drops any constants that don’t depend on parameters or functions of parameters.

The vectorized sampling statement above is equivalent to but more efficient than the following explicit loop.

```
  for (n in 1:N)  
    y[n] ~ binomial(K[n], phi);
```

In general, Stan will match dimensions, repeating scalars as necessary; any vector or array arguments must be the same size. When used as a function, the result is the sum of the log densities. The vectorized form can be up to an order of magnitude or more faster in some cases, depending on how many repeated calculations can be avoided.

The actual Stan program in `pool.stan` has many more derived quantities that will be used in the rest of this note; see the appendix for the full code of all of the models discussed.

We start by loading the RStan package.

```
library(rstan);
```

```
Loading required package: ggplot2
rstan (Version 2.9.0, packaged: 2016-01-05 16:17:47 UTC, GitRev: 05c3d0058b6a)
For execution on a local, multicore CPU with excess RAM we recommend calling
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
```

The model can be fit as follows, with `M` being the total number of draws in the complete posterior sample (each chain is by default split into half warmup and half sampling iterations and 4 chains are being run).

```
M <- 10000;
fit_pool <- stan("pool.stan", data=c("N", "K", "y", "K_new", "y_new"),
                 iter=(M / 2), chains=4);
```

```

SAMPLING FOR MODEL 'pool' NOW (CHAIN 1).

Chain 1, Iteration:    1 / 5000 [  0%]  (Warmup)
Chain 1, Iteration:  500 / 5000 [ 10%]  (Warmup)
Chain 1, Iteration: 1000 / 5000 [ 20%]  (Warmup)
Chain 1, Iteration: 1500 / 5000 [ 30%]  (Warmup)
Chain 1, Iteration: 2000 / 5000 [ 40%]  (Warmup)
Chain 1, Iteration: 2500 / 5000 [ 50%]  (Warmup)
Chain 1, Iteration: 2501 / 5000 [ 50%]  (Sampling)
Chain 1, Iteration: 3000 / 5000 [ 60%]  (Sampling)
Chain 1, Iteration: 3500 / 5000 [ 70%]  (Sampling)
Chain 1, Iteration: 4000 / 5000 [ 80%]  (Sampling)
Chain 1, Iteration: 4500 / 5000 [ 90%]  (Sampling)
Chain 1, Iteration: 5000 / 5000 [100%]  (Sampling)# 
#  Elapsed Time: 0.070874 seconds (Warm-up)
#                0.071481 seconds (Sampling)
#                0.142355 seconds (Total)
# 

SAMPLING FOR MODEL 'pool' NOW (CHAIN 2).

Chain 2, Iteration:    1 / 5000 [  0%]  (Warmup)
Chain 2, Iteration:  500 / 5000 [ 10%]  (Warmup)
Chain 2, Iteration: 1000 / 5000 [ 20%]  (Warmup)
Chain 2, Iteration: 1500 / 5000 [ 30%]  (Warmup)
Chain 2, Iteration: 2000 / 5000 [ 40%]  (Warmup)
Chain 2, Iteration: 2500 / 5000 [ 50%]  (Warmup)
Chain 2, Iteration: 2501 / 5000 [ 50%]  (Sampling)
Chain 2, Iteration: 3000 / 5000 [ 60%]  (Sampling)
Chain 2, Iteration: 3500 / 5000 [ 70%]  (Sampling)
Chain 2, Iteration: 4000 / 5000 [ 80%]  (Sampling)
Chain 2, Iteration: 4500 / 5000 [ 90%]  (Sampling)
Chain 2, Iteration: 5000 / 5000 [100%]  (Sampling)# 
#  Elapsed Time: 0.074807 seconds (Warm-up)
#                0.078645 seconds (Sampling)
#                0.153452 seconds (Total)
# 

SAMPLING FOR MODEL 'pool' NOW (CHAIN 3).

Chain 3, Iteration:    1 / 5000 [  0%]  (Warmup)
Chain 3, Iteration:  500 / 5000 [ 10%]  (Warmup)
Chain 3, Iteration: 1000 / 5000 [ 20%]  (Warmup)
Chain 3, Iteration: 1500 / 5000 [ 30%]  (Warmup)
Chain 3, Iteration: 2000 / 5000 [ 40%]  (Warmup)
Chain 3, Iteration: 2500 / 5000 [ 50%]  (Warmup)
Chain 3, Iteration: 2501 / 5000 [ 50%]  (Sampling)
Chain 3, Iteration: 3000 / 5000 [ 60%]  (Sampling)
Chain 3, Iteration: 3500 / 5000 [ 70%]  (Sampling)
Chain 3, Iteration: 4000 / 5000 [ 80%]  (Sampling)
Chain 3, Iteration: 4500 / 5000 [ 90%]  (Sampling)
Chain 3, Iteration: 5000 / 5000 [100%]  (Sampling)# 
#  Elapsed Time: 0.073841 seconds (Warm-up)
#                0.070549 seconds (Sampling)
#                0.14439 seconds (Total)
# 

SAMPLING FOR MODEL 'pool' NOW (CHAIN 4).

Chain 4, Iteration:    1 / 5000 [  0%]  (Warmup)
Chain 4, Iteration:  500 / 5000 [ 10%]  (Warmup)
Chain 4, Iteration: 1000 / 5000 [ 20%]  (Warmup)
Chain 4, Iteration: 1500 / 5000 [ 30%]  (Warmup)
Chain 4, Iteration: 2000 / 5000 [ 40%]  (Warmup)
Chain 4, Iteration: 2500 / 5000 [ 50%]  (Warmup)
Chain 4, Iteration: 2501 / 5000 [ 50%]  (Sampling)
Chain 4, Iteration: 3000 / 5000 [ 60%]  (Sampling)
Chain 4, Iteration: 3500 / 5000 [ 70%]  (Sampling)
Chain 4, Iteration: 4000 / 5000 [ 80%]  (Sampling)
Chain 4, Iteration: 4500 / 5000 [ 90%]  (Sampling)
Chain 4, Iteration: 5000 / 5000 [100%]  (Sampling)# 
#  Elapsed Time: 0.073971 seconds (Warm-up)
#                0.075928 seconds (Sampling)
#                0.149899 seconds (Total)
# 
```

```
ss_pool <- extract(fit_pool);
```

Here, we read the data out of the environment by name; normally we would prefer to encapsulate the data in a list to avoid naming conflicts in the top-level namespace. We showed the default output for the `stan()` function call here, but will suppress it in subsequent calls.

The posterior sample for `phi` can be summarized as follows.

```
print(fit_pool, c("phi"), probs=c(0.1, 0.5, 0.9));
```

```
Inference for Stan model: pool.
4 chains, each with iter=5000; warmup=2500; thin=1; 
post-warmup draws per chain=2500, total post-warmup draws=10000.

    mean se_mean   sd  10%  50%  90% n_eff Rhat
phi 0.27       0 0.02 0.25 0.27 0.29  3237    1

Samples were drawn using NUTS(diag_e) at Sat Jan 30 20:56:04 2016.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

The summary statistics begin with the posterior mean, the MCMC standard error on the posterior mean, and the posterior standard deviation. Then there are 0.1, 0.5, and 0.9 quantiles, which provide the posterior median and boundaries of the central 80% interval. The last two columns are for the effective sample size (MCMC standard error is the posterior standard deviation divided by the square root of the effective sample size) and the \\(\\hat{R}\\) convergence diagnostic (its value will be 1 if the chains have all converged to the same posterior mean and variance; see the Stan Manual (Stan Development Team 2015) or (Gelman et al. 2013). The \\(\\hat{R}\\) value here is consistent with convergence (i.e., near 1) and the effective sample size is good (roughly half the number of posterior draws; by default Stan uses as many iterations to warmup as it does for drawing the sample).

The result is a posterior mean for \\(\\theta\\) of \\(0.27\\) with an 80% central posterior interval of \\((0.25, 0.29)\\). With more data, such as from more players or from the rest of the season, the posterior approaches a delta function around the maximum likelihood estimate and the posterior interval around the centeral posterior intervals will shrink. Nevertheless, even if we know a player’s chance of success exactly, there is a large amount of uncertainty in running \\(K\\) binary trials with that chance of success; using a binomial model fundamentally bounds our prediction accuracy.

Although this model will be a good baseline for comparison, we have good reason to believe from a large amount of prior data (players with as many as 10,000 trials) that it is very unlikely that all players have the same chance of success.

## Model 2: No Pooling

A model with no pooling involves a separate chance-of-success parameter \\(\\theta\_n \\in \[0,1]\\) for each item \\(n\\).

The prior on each \\(\\theta\_n\\) is uniform,

\\\[ p(\\theta\_n) = \\mathsf{Uniform}(\\theta\_n \\, | \\, 0,1), \\]

and the \\(\\theta\_n\\) are assumed to be independent,

\\\[ p(\\theta) = \\prod\_{n=1}^N \\mathsf{Uniform}(\\theta\_n \\, | \\, 0,1). \\]

The likelihood then uses the chance of success \\(\\theta\_n\\) for item \\(n\\) in modeling the number of successes \\(y\_n\\) as

\\\[ p(y\_n \\, | \\, \\theta\_n) = \\mathsf{Binomial}(y\_n \\, | \\, K\_n, \\theta\_n). \\]

Assuming the \\(y\_n\\) are independent (conditional on \\(\\theta\\)), this leads to the total data likelihood

\\\[ p(y \\, | \\, \\theta) = \\prod\_{n=1}^N \\mathsf{Binomial}(y\_n \\, | \\, K\_n, \\theta\_n). \\]

The Stan program for no pooling only differs in declaring the ability parameters as an \\(N\\)-vector rather than a scalar.

```
parameters {
  vector<lower=0, upper=1>[N] theta; // chance of success
}
```

The constraint applies to each `theta[n]` and implies an independent uniform prior on each.

The model block defines the likelihood as binomial, using the efficient vectorized form

```
model {
  y ~ binomial(K, theta);  // likelihood
}
```

This is equivalent to the less efficient looped form

```
  for (n in 1:N)
    y[n] ~ binomial(K[n], theta[n]);
```

The full Stan program with all of the extra generated quantities, is in `no-pool.stan`, which is shown in the appendix.

This model can be fit the same way as the last model.

```
fit_no_pool <- stan("no-pool.stan", data=c("N", "K", "y", "K_new", "y_new"),
                    iter=(M / 2), chains=4);
ss_no_pool <- extract(fit_no_pool);
```

Results are displayed the same way.

```
print(fit_no_pool, c("theta"), probs=c(0.1, 0.5, 0.9));
```

```
Inference for Stan model: no-pool.
4 chains, each with iter=5000; warmup=2500; thin=1; 
post-warmup draws per chain=2500, total post-warmup draws=10000.

          mean se_mean   sd  10%  50%  90% n_eff Rhat
theta[1]  0.40       0 0.07 0.31 0.40 0.50 10000    1
theta[2]  0.38       0 0.07 0.29 0.38 0.47 10000    1
theta[3]  0.36       0 0.07 0.27 0.36 0.45 10000    1
theta[4]  0.34       0 0.07 0.26 0.34 0.43 10000    1
theta[5]  0.32       0 0.07 0.23 0.32 0.41 10000    1
theta[6]  0.32       0 0.07 0.23 0.32 0.41 10000    1
theta[7]  0.30       0 0.07 0.21 0.30 0.39 10000    1
theta[8]  0.28       0 0.07 0.19 0.27 0.36 10000    1
theta[9]  0.26       0 0.06 0.18 0.25 0.34 10000    1
theta[10] 0.26       0 0.06 0.18 0.25 0.34 10000    1
theta[11] 0.23       0 0.06 0.16 0.23 0.31 10000    1
theta[12] 0.23       0 0.06 0.16 0.23 0.31 10000    1
theta[13] 0.23       0 0.06 0.16 0.23 0.31 10000    1
theta[14] 0.23       0 0.06 0.16 0.23 0.32 10000    1
theta[15] 0.23       0 0.06 0.16 0.23 0.32 10000    1
theta[16] 0.21       0 0.06 0.14 0.21 0.29 10000    1
theta[17] 0.19       0 0.06 0.12 0.19 0.27 10000    1
theta[18] 0.17       0 0.05 0.10 0.17 0.24 10000    1

Samples were drawn using NUTS(diag_e) at Sat Jan 30 20:56:34 2016.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

Now there is a separate line for each item’s estimated \\(\\theta\_n\\). The posterior mode is the maximum likelihood estimate, but that requires running Stan’s optimizer to find; the posterior mean and median will be reasonably close to the posterior mode despite the skewness (the posterior can be shown analytically to be a Beta distribution).

Each 80% interval is much wider than the estimated interval for the population in the complete pooling model; this is to be expected—there are only 45 data items for each parameter here as opposed to 810 in the complete pooling case. If the items each had different numbers of trials, the intervals would also vary based on size.

As the estimated chance of success goes up toward 0.5, the 80% intervals gets wider. This is to be expected for chance of success parameters, because the standard deviation of a random variable distributed as \\(\\mathsf{Binomial}(K, \\theta)\\) is \\(\\sqrt{\\frac{\\theta \\, (1 - \\theta)}{K}}\\).

The no pooling model model provides better MCMC mixing than the complete pooling model as indicated by the effective sample size and convergence diagnostics \\(\\hat{R}\\); although not in and of itself meaningful, it is often the case that badly misspecified models provide difficult computationally (a result Andrew Gelman has dubbed “[The Folk Theorem](http://andrewgelman.com/2009/05/24/handy_statistic/)”).

Based on our existing knowledge of baseball, the no-pooling model is almost certainly overestimating the high abilities and underestimating lower abilities (Ted Williams, 30 years prior to the year this data was collected, was the last player with a 40% observed success rate over a season, whereas 20% is too low for all but a few rare defensive specialists).

## Model 3: Partial Pooling (Chance of Success)

Complete pooling provides estimated abilities that are too narrowly distributed for the items and removes any chance of modeling population variation. Estimating each chance of success separately without any pooling provides estimated abilities that are too broadly distributed for the items and hence too variable. Clearly some amount of pooling between these two extremes is called for. But how much?

A hierarchical model treats the players as belonging to a population of players. The properties of this population will be estimated along with player abilities, implicitly controlling the amount of pooling that is applied. The more variable the (estimate of the) population, the less pooling is applied.

Mathematically, the hierarchical model places a prior on the abilities with parameters that are themselves estimated. In this case, we will assume a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) as the prior as it is scaled to values in \\(\[0, 1]\\),

\\\[ p(\\theta\_n \\, | \\, \\alpha, \\beta) \\ = \\ \\mathsf{Beta}(\\theta\_n \\, | \\, \\alpha, \\beta), \\]

where \\(\\alpha, \\beta &gt; 0\\) are the parameters of the prior. The beta distribution is the conjugate prior for the binomial, meaning that the posterior is known to be a beta distribution. This also allows us to interpret the prior’s parameters as prior data, with \\(\\alpha - 1\\) being the prior number of successes and \\(\\beta - 1\\) being the prior number of failures, and \\(\\alpha = \\beta = 1\\) corresponding to no prior observations and thus a uniform distribution. Each \\(\\theta\_n\\) will be modeled as conditionally independent given the prior parameters, so that the complete prior is

\\\[ p(\\theta \\, | \\, \\alpha, \\beta) = \\prod\_{n=1}^N \\mathsf{Beta}(\\theta\_n \\, | \\, \\alpha, \\beta). \\]

The parameters \\(\\alpha\\) and \\(\\beta\\) are themselves given priors (sometimes called hyperpriors). Rather than parameterize \\(\\alpha\\) and \\(\\beta\\) directly, we will instead put priors on \\(\\phi \\in \[0, 1]\\) and \\(\\kappa &gt; 0\\), and then define

\\\[ \\alpha = \\kappa \\, \\phi \\]

and

\\\[ \\beta = \\kappa \\, (1 - \\phi). \\]

This reparameterization is convenient, because

- \\(\\phi = \\frac{\\alpha}{\\alpha + \\beta}\\) is the mean of a variable distributed as \\(\\mathsf{Beta}(\\alpha, \\beta)\\), and
- \\(\\kappa = \\alpha + \\beta\\) is the prior count plus two (roughly inversely related to the variance).

We will follow Gelman et al. (2013, Chapter 5) in providing a prior that factors into a uniform prior on \\(\\phi\\),

\\\[ p(\\phi) = \\mathsf{Uniform}(\\phi \\, | \\, 0, 1), \\]

and a Pareto prior on \\(\\kappa\\),

\\\[ p(\\kappa) = \\mathsf{Pareto}(\\kappa \\, | \\, 1, 1.5) \\propto \\kappa^{-2.5}. \\]

with the restriction \\(\\kappa &gt; 1\\). In general, for functions \\(f\\) and \\(g\\), we write \\(f(x) \\propto g(x)\\) if there is some constant \\(c\\) such that \\(f(x) = c \\, g(x)\\). The first argument to the Pareto distribution is a bound \\(\\epsilon &gt; 0\\), which in turn requires the outcome \\(\\kappa &gt; \\epsilon\\); this is required so that the distribution can be normalized to integrate to 1 over its support. The value \\(\\epsilon = 1\\) is a conservative choice for this problem as we expect in the posterior, \\(\\kappa\\) will be much greater than \\(1\\). The constraint \\(\\kappa &gt; 1\\) must therefore be included in the Stan parameter declaration, because Stan programs require support on the parameter values that satisfy their declared constraints.

The Stan code follows the definitions, with parameters declared with appropriate constraints as follows.

```
parameters {
  real<lower=0, upper=1> phi;         // population chance of success
  real<lower=1> kappa;                // population concentration
  vector<lower=0, upper=1>[N] theta;  // chance of success 
}
```

The lower-bound on \\(\\kappa\\) matches the first argument to the Pareto distribution in the model block.

```
model {
  kappa ~ pareto(1, 1.5);                        // hyperprior
  theta ~ beta(phi * kappa, (1 - phi) * kappa);  // prior
  y ~ binomial(K, theta);                        // likelihood
}
```

The values of \\(\\alpha = \\phi \\, \\kappa\\) and \\(\\beta = (1 - \\phi) \\, \\kappa\\) are computed in the arguments to the vectorized Beta sampling statement. The prior on \\(\\phi\\) is implicitly uniform because it is explicitly constrained to lie in \\(\[0, 1]\\). The prior on \\(\\kappa\\) is coded following the model definition.

The full model with all generated quantities can be coded in Stan as in the file `hier.stan`; it is displayed in the appendix. It is run as usual.

```
fit_hier <- stan("hier.stan", data=c("N", "K", "y", "K_new", "y_new"),
                 iter=(M / 2), chains=4,
                 seed=1234);
```

```
Warning: There were 3 divergent transitions after warmup. Increasing
adapt_delta above 0.8 may help.
```

```
Warning: Examine the pairs() plot to diagnose sampling problems
```

```
ss_hier <- extract(fit_hier);
```

Even though we set `results=‘hide’` in the knitr R call here, the error messages are still printed. We set the pseudorandom number generator seed here to 1234 so that we could discuss what the output looks like.

#### Divergent Transitions and Mitigation Strategies

In this case, there’s an error report for a number of what Stan calls “divergent transitions” after warmup. A divergent transition arises when there is an arithmetic issue in evaluating a numerical expression; this is almost always an overflow or underflow in well-specified models, but may simply be arguments out of bounds if the proper constraints have not been enforced (for instance, taking an unconstrained parameter as the chance of success for a Bernoulli distribution, which requires its chance of success to be in \\(\[0,1]\\)).

Whenever divergent transitions show up, it introduces a bias in the posterior draws away from the region where the divergence happens. Therefore, we should try to remove divergent transitions before trusting our posterior sample.

The underlying root cause of most divergent transitions is a step size that is too large. The underlying Hamiltonian Monte Carlo algorithm is attempting to follow the Hamiltonian using discrete time steps corresponding to the step size of the algorithm. When that step size is too large relative to posterior curvature, iteratively taking steps in the gradient times the step size provides a poor approximation to the posterior curvature. The problem with a hierarchical model is that the curvature in the posterior varies based on position; when the hierarchical variance is low, there is high curvature in the lower-level parameters around the mean. During warmup, Stan globally adapts its step size to a target acceptance rate, which can lead to too large step sizes for highly curved regions of the posterior. To mitigate this problem, we need to either reduce the step size or reparameterize. We firt consider reducing step size, and then in the next secton consider the superior alternative of reparameterization.

To reduce the step size of the algorithm, we want to lower the initial step size and increase the target acceptance rate. The former keeps the step size low to start; the latter makes sure the adapted step size is lower. So we’ll run this again with the same seed, this time lowering the step size (`stepsize`) and increasing the target acceptance rate (`adapt_delta`).

```
fit_hier <- stan("hier.stan", data=c("N", "K", "y", "K_new", "y_new"),
                 iter=(M / 2), chains=4,
                 seed=1234, 
                 control=list(stepsize=0.01, adapt_delta=0.99));
ss_hier <- extract(fit_hier);
```

Now it runs without divergent translations.

Summary statistics for the posterior are printed as before.

```
print(fit_hier, c("theta", "kappa", "phi"), probs=c(0.1, 0.5, 0.9));
```

```
Inference for Stan model: hier.
4 chains, each with iter=5000; warmup=2500; thin=1; 
post-warmup draws per chain=2500, total post-warmup draws=10000.

            mean se_mean     sd   10%   50%    90% n_eff Rhat
theta[1]    0.32    0.00   0.05  0.26  0.32   0.39  3809 1.00
theta[2]    0.31    0.00   0.05  0.25  0.31   0.38  3865 1.00
theta[3]    0.30    0.00   0.05  0.25  0.30   0.37  5130 1.00
theta[4]    0.29    0.00   0.05  0.24  0.29   0.36 10000 1.00
theta[5]    0.29    0.00   0.05  0.23  0.28   0.35  7610 1.00
theta[6]    0.29    0.00   0.05  0.23  0.28   0.34 10000 1.00
theta[7]    0.28    0.00   0.04  0.22  0.27   0.33 10000 1.00
theta[8]    0.27    0.00   0.04  0.21  0.27   0.32 10000 1.00
theta[9]    0.26    0.00   0.04  0.20  0.26   0.31 10000 1.00
theta[10]   0.26    0.00   0.04  0.21  0.26   0.31 10000 1.00
theta[11]   0.25    0.00   0.04  0.19  0.25   0.30 10000 1.00
theta[12]   0.25    0.00   0.04  0.19  0.25   0.30 10000 1.00
theta[13]   0.25    0.00   0.04  0.20  0.25   0.30 10000 1.00
theta[14]   0.25    0.00   0.04  0.19  0.25   0.30 10000 1.00
theta[15]   0.25    0.00   0.04  0.20  0.25   0.30 10000 1.00
theta[16]   0.24    0.00   0.04  0.18  0.24   0.29  6275 1.00
theta[17]   0.23    0.00   0.04  0.17  0.23   0.28  5286 1.00
theta[18]   0.22    0.00   0.04  0.17  0.22   0.28  4323 1.00
kappa     110.35    7.72 182.42 25.39 64.27 210.76   558 1.01
phi         0.27    0.00   0.02  0.24  0.27   0.29  4543 1.00

Samples were drawn using NUTS(diag_e) at Sat Jan 30 20:57:10 2016.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

Because the Beta prior is conjugate to the binomial likelihood, the amount of interpolation between the data and the prior in this particular case is easy to quantify. The data consists of \\(K\\) observations, whereas the prior will be weighted as if it were \\(\\kappa - 2\\) observations (specifically \\(\\phi \\, \\kappa - 1\\) prior successes and \\((1 - \\phi) \\, \\kappa - 1\\) prior failures).

The parameter \\(\\kappa\\) is not well determined by the combination of data and Pareto prior, with a posterior 80% interval of roughly \\((25, 225)\\). By the informal discussion above, \\(\\kappa \\in (25, 225)\\) ranges from weighting the data 2:1 relative to the prior to weighting it 1:5. The wide posterior interval for \\(\\kappa\\) arises because the exact variance in the population is not well constrained by only 18 trials of size 45. If there were more items (higher \\(N\\)) or even more trials per item (higher \\(K\\)), the posterior for \\(\\kappa\\) would be more tightly constrained (see the exercises for an example).

It is also evident from the posterior summary that the lower effective sample size for \\(\\kappa\\) indicates it is not mixing as well as the other components of the model. Again, this is to be expected with a centered hierarchical prior and low data counts. This is an example where a poorly constrained parameter leads to reduced computational efficiency (as reflected in the effective sample size). Such poor mixing is typical of centered parameterizations in hierarchical models (Betancourt and Girolami 2015). It is not immediately clear how to provide a non-centered analysis of the beta prior, because it isn’t supplied with a location/scale parameterization on the unconstrained scale. Instead, we consider an alternative parameterization in the next section.

Figure 5.3 from (Gelman et al. 2014) plots the fitted values for \\(\\phi\\) and \\(\\kappa\\) on the unconstrained scale, which is the space over which Stan is sampling. The variable \\(\\phi \\in \[0,1]\\) is transformed to \\(\\mathrm{logit}(\\phi) = \\log(\\phi / (1 - \\phi))\\) and \\(\\kappa \\in (0, \\infty)\\) is transformed to \\(\\log \\kappa\\). We reproduce that figure here for our running example.

```
df_bda3_fig_5_3 <- with(ss_hier,
                        data.frame(x = log(phi / (1 - phi)),
                                   y = log(kappa)));
phi_sim <- ss_hier$phi;
kappa_sim <- ss_hier$kappa;
df_bda3_fig_5_3 <- data.frame(x = log(phi_sim / (1 - phi_sim)),
                              y = log(kappa_sim));
library(ggplot2);
plot_bda3_fig_5_3 <- 
  ggplot(df_bda3_fig_5_3, aes(x=x, y=y)) +
  geom_point(shape=19, alpha=0.15) +
  xlab("logit(phi) = log(alpha / beta)") +
  ylab("log(kappa) = log(alpha + beta)");
plot_bda3_fig_5_3;
```

The (upside-down) funnel-like shape of the posterior is evident, with higher \\(\\kappa\\) values corresponding to lower variance (hence the upside-downedness) and thus a narrower range of \\(\\phi\\) values. This particular funnel-like posterior plot arises because when the population variation is large (i.e., \\(\\kappa\\) is small), there is weak non-identifiability between the prior mean (\\(\\phi\\)) and the mean of the random effects (\\(\\mathrm{mean}(\\theta)\\)).

The root problem with the non-centered parameterization, though, is the relation between \\(\\kappa\\) and the \\(\\theta\\). For example, consider the following plot of \\(\\theta\_1\\) vs. \\(\\kappa\\).

```
inv_logit <- function(u) { 1 / (1 + exp(-u)); }
theta1_sim <- ss_hier$theta[ , 1];
kappa_sim <- ss_hier$kappa;
df_funnel <- data.frame(x = inv_logit(theta1_sim),
                        y = log(kappa_sim));
library(ggplot2);
plot_funnel<- 
  ggplot(df_funnel, aes(x=x, y=y)) +
  geom_point(shape=19, alpha=0.15) +
  xlab("logit(theta[1])") + 
  ylab("log(kappa)");
plot_funnel;
```

We have unconstrained both \\(\\theta\_1\\) (using \\(\\mathrm{logit}^{-1}\\)) and \\(\\kappa\\) (using \\(\\log\\)), rendering both on the unconstrained scale over which sampling is actually carried out in Stan. This helps in diagnosing problems with sampling that may not be as apparent on the constrained scale. All of the transforms used by Stan for constraints are detailed in the Stan manual (Stan Development Team 2015).

It is clear here that when \\(\\kappa\\) is smaller, there is more latitude for \\(\\theta\_1\\) to move around. This phenomenon was originally discussed by Neal (2003), is covered in the Stan manual (Stan Development Team 2015), and is analyzed fors Hamiltonian Monte Carlo by Betancourt and Girolami (2015).

## Model 4: Partial Pooling (Log Odds)

The previous models all used a direct parameterization of the chance-of-success \\(\\theta\_n \\in \[0,1]\\). In this section, we consider an alternative parameterization in terms of the log-odds \\(\\alpha\_n\\), which are defined by the logit transform as

\\\[ \\alpha\_n = \\mathrm{logit}(\\theta\_n) = \\log \\, \\frac{\\theta\_n}{1 - \\theta\_n}. \\]

For example, \\(\\theta\_n = 0.25\\) corresponds to odds of \\(.25\\) to \\(.75\\) (equivalently, \\(1\\) to \\(3\\)), or log-odds of \\(\\log .25 / .75 = -1.1\\).

We will still use a binomial likelihood, only now we have logit as a so-called “link” function.

\\\[ p(y\_n \\, | \\, K\_n, \\alpha\_n) \\ = \\ \\mathsf{Binomial}(y\_n \\, | \\, K\_n, \\ \\mathrm{logit}^{-1}(\\alpha\_n)) \\]

The inverse logit function is the logistic sigmoid from which logistic regression gets its name,

\\\[ \\mathrm{logit}^{-1}(\\alpha\_n) = \\frac{1}{1 + \\exp(-\\alpha\_n)} = \\theta\_n. \\]

By construction, for any \\(\\alpha\_n \\in (-\\infty, \\infty)\\), \\(\\mathrm{logit}^{-1}(\\alpha\_n) \\in (0, 1)\\); the sigmoid converts arbitrary log odds back to the chance-of-success scale. In practice, floating point calculations may underflow or overflow to 0 or 1, so in practice, we only know the result will be in \\(\[0, 1]\\).

Stan has a binomial probability function with a built-in logit link function, with which we can define the likelihood directly as

\\\[ p(y\_n \\, | \\, K\_n, \\alpha\_n) \\ = \\ \\mathsf{BinomialLogit}(y\_n \\, | \\, K\_n, \\alpha\_n) \\ = \\ \\mathsf{Binomial}(y\_n \\, | \\, K\_n, \\ \\mathrm{logit}^{-1}(\\alpha\_n)). \\]

We use a simple normal hierarchical prior,

\\\[ p(\\alpha\_n \\, | \\, \\mu, \\sigma) = \\mathsf{Normal}(\\alpha\_n \\, | \\, \\mu, \\sigma). \\]

Then one level up, we use a weakly informative hyperprior for \\(\\mu\\),

\\\[ p(\\mu) = \\mathsf{Normal}(\\mu \\, | \\, -1, 1), \\]

which places 95% of the prior probability for \\(\\mu\\) in the interval \\((-3, 1)\\), which inverse-logit transforms to the interval \\((0.05, 0.73)\\) with a median 0.27 chance of success. An even narrower prior is actually motivated here from substantial baseball knowledge. This value should obviously be changed for other applications.

The prior scale \\(\\sigma &gt; 0\\) can be taken to be a truncated normal (half normal, here):

\\\[ p(\\sigma) \\ = \\ 2 \\, \\mathsf{Normal}(\\sigma \\, | \\, 0, 1) \\ \\propto \\ \\mathsf{Normal}(\\sigma \\, | \\, 0, 1). \\]

This is a fairly broad prior in this case, being on the log-odds scale.

One of the major advantages of casting the problem in terms of log odds is that its now easier to add in fixed effects and other multilevel effects, or even varying intercepts and slopes with multivariate priors (see Gelman and Hill (2007) for many such examples, all of which have been translated to Stan).

#### Centered Parameterization

We will begin with the natural centered parameterization, but we must point out ahead of time that this is not the optimal way to code this model in Stan. The centered parameterization directly follows the mathematical presentation.

```
parameters {
  real mu;                       // population mean of success log-odds
  real<lower=0> sigma;           // population sd of success log-odds
  vector[N] alpha;               // success log-odds
}
model {
  mu ~ normal(-1, 1);               // hyperprior
  sigma ~ normal(0, 1);             // hyperprior
  alpha ~ normal(mu, sigma);        // prior (hierarchical)
  y ~ binomial_logit(K, alpha);     // likelihood
}
```

The full program may be found in `hier-logit-centered.stan`; it is not included in this document directly (code for the other models is in an appendix at the end of this document).

```
fit_hier_logit_centered <- 
  stan("hier-logit-centered.stan", data=c("N", "K", "y"),
       iter=(M / 2), chains=4,
       seed=1234);
```

```
Warning: There were 87 divergent transitions after warmup. Increasing
adapt_delta above 0.8 may help.
```

```
Warning: Examine the pairs() plot to diagnose sampling problems
```

```
ss_hier_logit_centered <- extract(fit_hier_logit_centered);
```

```
print(fit_hier_logit_centered, 
      c("sigma", "theta[1]", "theta[10]", "alpha[1]", "alpha[10]",
        "mu", "sigma", "lp__"), probs=c(0.1, 0.5, 0.9));
```

```
Inference for Stan model: hier-logit-centered.
4 chains, each with iter=5000; warmup=2500; thin=1; 
post-warmup draws per chain=2500, total post-warmup draws=10000.

             mean se_mean    sd     10%     50%     90% n_eff Rhat
sigma        0.17    0.01  0.11    0.04    0.16    0.32   236 1.02
theta[1]     0.29    0.00  0.04    0.25    0.29    0.35   674 1.01
theta[10]    0.26    0.00  0.03    0.22    0.26    0.30  4225 1.00
alpha[1]    -0.88    0.01  0.21   -1.10   -0.92   -0.60   643 1.01
alpha[10]   -1.05    0.00  0.17   -1.26   -1.05   -0.84  2993 1.00
mu          -1.03    0.00  0.09   -1.14   -1.03   -0.91  1257 1.00
lp__      -442.26    1.23 11.68 -455.08 -444.57 -425.36    90 1.05

Samples were drawn using NUTS(diag_e) at Sat Jan 30 20:57:42 2016.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

With the centered parameterization and Stan’s defalt stepsize and target acceptance rate, there are many divergent transitions. Even with 10,000 posterior draws, the effective sample size for the population scale \\(\\sigma\\) is in the low hundreds, with \\(\\hat{R} &gt; 1.01\\). These results might be usable, but the high \\(\\hat{R}\\) and low effective sample size are sure signs the chains aren’t mixing well.

*Warning:* In cases where the chains are not mixing well, Stan’s effective sample size estimate will be much lower than that from packages such as Coda (Plummer 2006), which analyze each chain’s effective sample size separately (Gelman et al. 2013; Stan Development Team 2016). Thus it is important when comparing systems to use Stan’s more conservative effective sample size estimates; the other systems assume complete mixing in their effective sample size calculations and thus overestimate their performance when there is not good mixing.

```
alpha_1_centered <- ss_hier_logit_centered$alpha[, 1];
funnel_centered_df <-  
  with(ss_hier_logit_centered,
       data.frame(x = alpha_1_centered,
                  y = log(sigma)));
funnel_centered <-
  ggplot(funnel_centered_df, aes(x=x, y=y)) +
  geom_point(shape=19, alpha=0.1) +
  xlab("player 1 log odds of success: alpha[1]") + 
  ylab("log population scale: log sigma") +
  ggtitle("Centered Hierarchical Funnel Plot");
funnel_centered;
```

The important feature of this plot is the reduced range possible for \\(\\alpha\_1\\) as \\(\\sigma\\) approaches zero (i.e., \\(\\log \\sigma\\) approaches negative infinity). Neal (2003) refers to the prior for the non-centered model as a “funnel”. With more data, the funnel shape becomes less pronounced and this centered parameterization is actually more efficient than the non-centered parameterization we discuss in the next section (Betancourt and Girolami 2015).

#### Non-Centered Parameterization

Betancourt and Girolami (2015) provide a detailed discussion of why the centered parameterization described in the previous section is challenging for MCMC methods to sample when there are small counts per group (here, the players are the groups and each has only 45 at bats observed).

To mitigate the problem, they suggest moving to a non-centered parameterization, as has also been shown to be helpful for Gibbs and random-walk Metropolis samplers (Papaspiliopoulos et al. 2003). This basically amounts to changing the parameterization over which sampling is done, taking now a standard unit normal prior for a new variable,

\\\[ \\alpha^{\\mathrm{std}}\_n = \\frac{\\alpha\_n - \\mu}{\\sigma}. \\]

Then we can parameterize in terms of \\(\\alpha^{\\mathrm{std}}\\), which has a standard-normal distribution

\\\[ p(\\alpha^{\\mathrm{std}}\_n) = \\mathsf{Normal}(\\alpha^{\\mathrm{std}}\_n \\, | \\, 0, 1). \\]

We can then define our original \\(\\alpha\\) as a derived quantity

\\\[ \\alpha\_n = \\mu + \\sigma \\, \\alpha^{\\mathrm{std}}\_n. \\]

We code this implicitly in the Stan model by defining the likelihood as

\\\[ p(y\_n \\, | \\, \\alpha^{\\mathrm{std}}\_n, \\mu, \\sigma, K) \\ = \\ \\mathsf{BinomialLogit}(K\_n, \\ \\mu + \\sigma \\, \\alpha\_n). \\]

This decouples the sampling distribution for \\(\\alpha^{\\mathrm{std}}\\) from \\(\\mu\\) and \\(\\sigma\\), greatly reducing their correlation in the posterior.

The Stan program’s parameter declaration and model directly follow the definition.

```
parameters {
  real mu;                       // population mean of success log-odds
  real<lower=0> sigma;           // population sd of success log-odds
  vector[N] alpha_std;           // success log-odds
}
model {
  mu ~ normal(-1, 1);                             // hyperprior
  sigma ~ normal(0, 1);                           // hyperprior
  alpha_std ~ normal(0, 1);                       // prior
  y ~ binomial_logit(K, mu + sigma * alpha_std);  // likelihood
}
```

Because the parameters to the prior for \\(\\sigma\\) are constants, the normalization for the half-prior (compared to the full prior) is constant and does not need to be included in the notation. This only works if the parameters to the density are data or constants; if they are defined as parameters or as quantities depending on parameters, then explicit truncation is required.

For the purposes of comparison, the chance of success \\(\\theta\\) is computed as a generated quantity.

```
generated quantities {
  vector[N] theta;  // chance of success
  ...
  for (n in 1:N)
    theta[n] <- inv_logit(mu + sigma * alpha_std[n]);
  ...
}
```

The full Stan program for the hierarchical logistic model is in `hier-logit.stan` and displayed in the appendix.

It is fit and the values are extracted as follows.

```
fit_hier_logit <- stan("hier-logit.stan", data=c("N", "K", "y", "K_new", "y_new"),
                       iter=(M / 2), chains=4,
                       control=list(stepsize=0.01, adapt_delta=0.99));
ss_hier_logit <- extract(fit_hier_logit);
```

We can print as before.

```
print(fit_hier_logit, c("alpha_std", "theta", "mu", "sigma"), probs=c(0.1, 0.5, 0.9));
```

```
Inference for Stan model: hier-logit.
4 chains, each with iter=5000; warmup=2500; thin=1; 
post-warmup draws per chain=2500, total post-warmup draws=10000.

               mean se_mean   sd   10%   50%   90% n_eff Rhat
alpha_std[1]   0.69    0.01 0.96 -0.58  0.72  1.89 10000    1
alpha_std[2]   0.57    0.01 0.92 -0.63  0.59  1.71 10000    1
alpha_std[3]   0.45    0.01 0.93 -0.75  0.47  1.61 10000    1
alpha_std[4]   0.34    0.01 0.91 -0.84  0.37  1.48 10000    1
alpha_std[5]   0.24    0.01 0.91 -0.92  0.26  1.39 10000    1
alpha_std[6]   0.22    0.01 0.90 -0.95  0.24  1.37 10000    1
alpha_std[7]   0.13    0.01 0.89 -1.00  0.13  1.25 10000    1
alpha_std[8]   0.01    0.01 0.90 -1.15  0.02  1.13 10000    1
alpha_std[9]  -0.09    0.01 0.89 -1.22 -0.10  1.05 10000    1
alpha_std[10] -0.09    0.01 0.89 -1.23 -0.09  1.04 10000    1
alpha_std[11] -0.21    0.01 0.88 -1.31 -0.22  0.90 10000    1
alpha_std[12] -0.21    0.01 0.90 -1.36 -0.21  0.95 10000    1
alpha_std[13] -0.22    0.01 0.91 -1.37 -0.22  0.94 10000    1
alpha_std[14] -0.22    0.01 0.91 -1.35 -0.22  0.93 10000    1
alpha_std[15] -0.22    0.01 0.91 -1.36 -0.23  0.95 10000    1
alpha_std[16] -0.34    0.01 0.90 -1.47 -0.36  0.82 10000    1
alpha_std[17] -0.44    0.01 0.93 -1.62 -0.46  0.77 10000    1
alpha_std[18] -0.56    0.01 0.95 -1.77 -0.57  0.68 10000    1
theta[1]       0.29    0.00 0.04  0.25  0.29  0.35 10000    1
theta[2]       0.29    0.00 0.04  0.25  0.28  0.34 10000    1
theta[3]       0.28    0.00 0.04  0.24  0.28  0.33 10000    1
theta[4]       0.28    0.00 0.04  0.24  0.27  0.33 10000    1
theta[5]       0.27    0.00 0.03  0.24  0.27  0.32 10000    1
theta[6]       0.27    0.00 0.03  0.24  0.27  0.32 10000    1
theta[7]       0.27    0.00 0.03  0.23  0.27  0.31 10000    1
theta[8]       0.27    0.00 0.03  0.23  0.26  0.30 10000    1
theta[9]       0.26    0.00 0.03  0.22  0.26  0.30 10000    1
theta[10]      0.26    0.00 0.03  0.22  0.26  0.30 10000    1
theta[11]      0.26    0.00 0.03  0.21  0.26  0.29 10000    1
theta[12]      0.26    0.00 0.03  0.21  0.26  0.29 10000    1
theta[13]      0.26    0.00 0.03  0.21  0.26  0.29 10000    1
theta[14]      0.26    0.00 0.03  0.21  0.26  0.29 10000    1
theta[15]      0.26    0.00 0.03  0.21  0.26  0.29 10000    1
theta[16]      0.25    0.00 0.03  0.21  0.25  0.29 10000    1
theta[17]      0.25    0.00 0.03  0.20  0.25  0.29 10000    1
theta[18]      0.24    0.00 0.04  0.19  0.25  0.28 10000    1
mu            -1.03    0.00 0.09 -1.14 -1.03 -0.91 10000    1
sigma          0.17    0.00 0.11  0.03  0.15  0.32  3634    1

Samples were drawn using NUTS(diag_e) at Sat Jan 30 20:58:17 2016.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
```

It is clear from the wide posteriors for the \\(\\theta\_n\\) that there is considerable uncertainty in the estimates of chance-of-success on an item-by-item basis. With an 80% interval of \\((0.03, 0.32)\\), it is clear that the data is consistent with complete pooling (i.e., \\(\\sigma = 0\\)).

Compared to the direct beta priors with uniform and Pareto hyperpriors shown in the first example, the normal prior on log odds exerts more pull toward the population mean. The posterior means for \\(\\theta\\) ranged from 0.22 to 0.32 with the beta prior, but only range from 0.24 to 0.29 for the normal prior. Furthermore, the posterior intervals for each values are shrunk compared to the beta prior. For example, Roberto Clemente (\\(n = 1\\)), has an 80% central poster

...(content truncated)
