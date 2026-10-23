# Track B primary evidence and implementation mapping

Retrieved 2026-10-23 with AnySearch discovery (academic.search/code.snippet; exact logs in R75/literature) and direct public source downloads. Search snippets are discovery only.

1. Evgeniou and Pontil, **Regularized Multi-Task Learning**, peer-reviewed KDD 2004, DOI 10.1145/1014052.1014067. Author-hosted full text: http://www0.cs.ucl.ac.uk/staff/m.pontil/reading/mt-kdd.pdf . Read Section 2, equation (1) w_t=w_0+v_t; objective (2), and the equivalent mean-deviation penalty (6)-(10). Original experiment is SVM/hinge loss. Our squared-loss, per-Subject centered Brier regression and smoothed intercept are adaptations; no claim to reproduce the paper's experiments or numerical results.

2. Zhou, Chen and Ye, **MALSAR: Multi-task learning via structural regularization**, author-maintained repository https://github.com/jiayuzhou/MALSAR , pinned commit `fb9751594983df020ddc4f7e4a40520ee7c37989`. Read author manual Section 4.4 (pp. 15-16), equations (10)-(13), and `MALSAR/functions/SRMTL/Least_SRMTL.m`. With R=I-11'/T, no L1 term, this gives mean-deviation quadratic regularization and least squares. This is the MALSAR authors' implementation of the family, not a claim that KDD 2004 authors released this repository.

3. Local R72 `peer_current_predictor.py`, `run_peer_current_outer.py`, and frozen full probabilities are the immediate implementation/reference. Their same3/pooled3/diff3 scores retain the old hash-ordered cohorts. Our random-cohort results must receive distinct labels.

## Exact quadratic mapping for the conditional candidate

Our objective is SSE + ||b||² + 8 sum_s ||d_s||², w_s=b+d_s. Eliminating b gives penalty 8 sum ||w_s||² - 64/(1+8T)||sum w_s||². For unit SSE this equals a graph penalty with coefficients a=8/(1+8T), g=8-a. The actual MALSAR source uses **0.5*SSE**, although the manual's equation (13) omits that factor; therefore its matching inputs are rho_L2=a/2, rho_graph=g/2, rho_L1=0, R=I-11'/T. Our dual kernel is dot(x_i,x_j)*(1+I_same_subject/8). The analytical direct solve avoids introducing a new iterative optimizer. Synthetic primal/dual/graph-objective equivalence is checked before acceptance.

The fixed value 8 is our preregistered shrinkage design, not a literature-claimed optimal PAIEC setting. Partial pooling runs only after the source support-only heterogeneity entry criterion; no claim is made from low-sample raw coefficient angle alone. The fitted-null diagnostic is exploratory and model-dependent.

Manifest records exact downloaded bytes and SHA256. Initial HTTPS UCL request failed; original HTTP author URL succeeded with PDF magic verification. No access controls were bypassed.
