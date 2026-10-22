# Exchangeability as a model choice

AnySearch located Pitman's 1996 *Some Developments of the Blackwell–MacQueen Urn Scheme* (DOI10.1214/lnms/1215453576). A follow-up academic search did not locate the original text; a primary-domain search found the [author-hosted PDF](https://www.stat.berkeley.edu/users/pitman/blmq.pdf), which was read directly. The DOI tool request failed; no claim is based on that failed extraction.

Source-supported: equation19 gives the Blackwell–MacQueen existing/new-cluster prediction rule; section3.2 defines exchangeable partition probabilities. Multiplying this rule yields the finite partition prior proportional to alpha^K times the product of(cluster_size-1)!.

Project hypothesis, not a theorem from the paper: a current cohort's subject success rates may be represented by a mixture over shared-rate clusters, including all-singleton independence. Put a Uniform(0,1) rate in each cluster, integrate the Bernoulli likelihood, and average posterior rates. Marginal rate priors remain uniform across concentrations, separating dependence from one-subject prior shape. Equal-rate clusters and conditionally independent outcomes are modelling assumptions; the literature does not guarantee that they hold for PAIEC or improve its Brier ALC.
