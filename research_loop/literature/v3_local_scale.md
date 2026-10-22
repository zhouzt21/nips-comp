# Local-scale calibration: source check, not performance evidence

2026-10-22. Question: if a later oracle audit finds under-sized local corrections, can current-support leave-one-out checks be computed cheaply without exposing evaluation labels?

AnySearch academic discovery/search succeeded; raw search saved in `raw/v3_local_scale_search.txt`. Extraction of a guessed arXiv HTML version failed (request 27fbbdfc-37a9-4b17-96b6-1f60e2983982). The primary [paper PDF, arXiv:2005.09301v2](https://arxiv.org/pdf/2005.09301) was then read directly with the web tool. This was a URL extraction failure, not evidence of a broken AnySearch API.

Source-supported: van de Wiel, van Nee and Rauschenberger (2021), *Fast cross-validation for multi-penalty high-dimensional ridge regression*, sections 3–4, exploit sample-space algebra for ridge cross-validation and allow distinct feature-block penalties. Their scope includes unpenalized covariates. This supports computational feasibility, not a PAIEC performance claim.

Our conditional inference: an unpenalized intercept plus fixed ridge slope admits exact leave-one-out fitted values; shrinkage of the local prevalence must be handled separately. Any proposed formula must agree with explicit refits, including re-centering after deleting each support item. Neither that equality nor this paper establishes that a scale estimated from 3–31 labels will generalize. A future experiment must test that separately. No algorithm or parameter is selected from this literature search.
