# R40: input geometry is a hypothesis, not label information by itself

AnySearch academic.search retrieved the primary COLT2008 paper by Shai Ben-David, Tyler Lu and David Pál, *Does Unlabeled Data Provably Help? Worst-case Analysis of the Sample Complexity of Semi-Supervised Learning*.

Primary text read: https://www.learningtheory.org/colt2008/papers/92-Ben-David.pdf (abstract, introduction and section3 discussion). The paper separates knowledge of an input marginal from assumptions relating labels to that marginal. It establishes restrictions for specified hypothesis classes and gives examples where cluster-based reasoning is misleading. It does not establish a PAIEC-specific bound, nor prove that all unlabeled representations fail.

Project inference: current-support PCA can cheaply propose an input partition, but high variance is not proof of useful label strata. R40 therefore uses no query labels for its partition, retains an unsplit count model, and compares random directions and disrupted input-label correspondence. Its exact Beta sequence likelihood is a direct elementary calculation, not a performance result imported from this paper. No additional unlabeled test pool is accessed.

Search completed successfully this turn; AnySearch API recovery is not currently needed. Queries contained only public methodological terms, not local data or credentials.
