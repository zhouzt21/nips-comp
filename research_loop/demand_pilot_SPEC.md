# R74 No-Peer demand pilot: frozen before reading outcome results

Date: 2026-10-23. Parent contract: `DUAL_TRACK_SPEC.md`; original authority: `doc/plan_8.md`.

## Claim and falsifier

Exploratory PAIEC adaptation of ADeLe's original demand annotation and Benchmark-heldout informational test. Four Luna teacher grades may add transferable response information; a small CPU ridge student may retain useful inductive bias. No faithful reproduction claim for the paper's subject-specific 18-demand+UG RandomForest, GPT-4o snapshot or AUROC/ECE protocol. Original author code at commit 816321f327dfe66910b72b514156aaa296b15f00 is read and archived. No author battery responses are training data.

Falsification: teacher fails matched source B0 controls, student fails grade mean baseline, or deployable grade features fail paired source Brier controls. Do not expand annotation, models or hyperparameters on failure.

## Data and isolation

Use final frozen R73 annotations, 128 distinct original R53 query items, AS/MCr/QLl/KNf. A null is missing supervision, never grade zero. Three original source pools, each original `inner_halves` in both directions; excluded outer Benchmarks remain excluded. Pools overlap and are not independent replicates. No new outer evaluation unless the predeclared entry tests pass. Student preprocessing and all response models fit only the fit half. IDs/BM labels select splits and audit; they never enter tensors. Each grade head uses all valid fit-half item grades for that dimension. Teacher diagnostic uses complete four-grade rows only, matching author complete-case semantics.

Item features are unchanged 384 MiniLM + 256 legal metadata. Source episodes are seed73, cap25 per BM, query<=64; all repeated responses contribute via counts/successes. The original 640-dimensional source caches are reused. Historical R59 R56 and R66 item-only OOF probabilities are matched by query item and response variant/count; they are historical contextual controls with different training data/model capacity, not the same-data demand contrast. Preserve variant probabilities rather than replacing nonlinear Brier with Brier of mean probabilities. Check R25 item/variant score equivalence.

## Models and fixed parameters

- Ridge student: four independent heads, each fit on distinct annotated items with valid grade, item-equal weights. Center and standardize 640 inputs on that head's fit items only (std floor 0.05), divide by sqrt(640). Unpenalized intercept, mean weighted squared loss + **0.1** times squared coefficient norm. Predict continuous grades clipped to [0,5]; nearest integer only for reporting agreement. Constant comparator is per-head fit grade mean. No tuning.
- Response ridge: pair-equal risk; within each pair use response counts normalized on the relevant query set. Intercept unpenalized; mean weighted Brier + **0.01** coefficient norm. Fit-only weighted input standardization (std floor 0.05), divide 640 original coordinates by sqrt(640), four grade coordinates by sqrt(4). Probabilities clipped [0,1]. This is a CPU adaptation, not the author RF.
- Same 644 input slots for base (last four zeros), augmented and permuted models. Demand-only sets first 640 coordinates to zero. No extra metadata/subject fields. Extra active demand coordinates are evaluated against identical-capacity permuted-demand controls.
- Frozen permutation seed **20261023**: jointly permute four teacher grades within each BM among complete-case items, preserving per-BM means/missingness. For student negative control, permute each dimension's valid grades within BM before fitting that head. No outcome values enter RNG seeds. Constant grade groups may yield unchanged permutation; report effective changed fractions.
- Local support model: retain R25's exact semantic/metadata scaling and ridge1; append continuous student grades divided by **5**, no additional coefficient search. Demand-only local has 640 zeros + grades/5. Permuted-student local uses the control student's outputs. Each local solve uses only current own labels, own centering and smoothed2 intercept; B0/B1 remain exact R25.

## Stages and controls

1. Teacher information B0 only: train response models on complete annotated fit-query rows; evaluate complete annotated held-out-query rows. Base, +teacher, +within-BM-permuted teacher, demand-only, +student (same response subset), constant0.5, fit-source constant, historical R56 B0 and R66 item OOF B0. Report missing/usable row, pair, BM and item counts. Teacher B1 and teacher support-local six-budget effects are not claimed because original selected items are query-only and most support grades are absent.
2. Mandatory grade quality: all valid held-out annotation dimensions; continuous MAE/RMSE, rounded exact and >1 error versus fit-mean, by rubric/BM, fit/validation IDs, null exclusions. Quantify coverage, overlap and grade variance.
3. Deployable six-budget source pilot: response base/+student/+permuted-student/demand-only priors train on **all fit source query responses**, a distinct full-source-data stage. B0=q(query), B1=clip(q(query)+(y0-q(support0))/3), B3 onward exact R25. Local controls replace only B3/B7/B15/B31 as above. Include constant0.5, frozen smoothed2, training-selected mean, R25, historical R56 segment and R66 item-only segment. Save all probabilities/counts/successes/item/subject identities and fit hashes. Report prediction accuracy at fixed p>=0.5 as secondary, not subject raw success rate.

## Entry / stopping and resource contract

Teacher entry requires augmented B0 to beat base and permutation in at least two pools, pooled pair-equal and BM-equal, and excluding alignment_faking. A deployable candidate must beat R25 in at least two pools, pooled pair-equal, BM-equal and excluding alignment; it must also beat its capacity-matched base and permutation in pooled pair-equal and BM-equal. Report BM-cluster 95% intervals (10,000 draws seed20261023), all source slices and max regressions. Existing public-data reuse remains exploratory. If teacher entry and any deployable entry both pass, notify main agent before selecting exactly one frozen outer candidate (lowest source ALC among passing methods, deterministic lexical tie-break). Otherwise stop Track A at pilot, preserve R25. Do not enlarge 128 items or four dimensions without new evidence/authorization.

CPU threadpool limit2; intended runtime <=20min, record actual separately. Six split students and response models saved as NumPy artifacts, no GPU/new downloads/paid API. Reading source caches and annotation quality is allowed before freeze; no new response-outcome scores are inspected before the contract snapshot. Root owns R73, global state and todo; this lane owns only R74 and demand_pilot/demand_student code/reports.
