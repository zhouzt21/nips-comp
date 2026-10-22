# R13 / V2-04: a factorial test of two identified probability components

Registered2026-10-22 before results. Primary conditional cohort4, seeds0/1/2; stress cohorts1/6. The original isolated-pair gate remains intact.

## Updated high-level state

R10 identifies actual within-pair item discrimination from local labels. R12 rejects a stronger attribution: R11's more complicated Bayesian integration does not beat a strength-matched plug-in. Thus separate the probability mean (peer evidence) and the local item slope (current matching-subject evidence). This composition is justified by distinct measured components, not by method names.

## Dependencies / alternatives

Audit -> component identification (R10) -> estimator simplification (R12) -> factorial composition (R13). If mean and slope interact negatively, retain the separate simpler methods. No new architecture or representation, no coefficient tuning after outer results.

## Frozen predictor

Semantic vectors exactly R7, same model revision/window. Local ridge uses each fold's R7 training-selected lambda. Peer intercept uses R12 training-selected alpha and kappa (4/8 in every fold). Base intercept `(1+Sown)/(2+nown)`; peer intercept `(alpha*qpeer+Sown)/(alpha+nown)` with `qpeer=(kappa/2+Speer)/(kappa+npeer)`; no distinct peers gives base intercept.

All labels with the identical visible subject dict and benchmark are pooled. For aliases in public source IDs, form a single local support matrix from all their allowed labels; never use private subject_id to decide what predict can access. Use centered support ridge slope `(x-mean(A)) Ac' (Ac Ac'+lambda I)^-1 (y-mean(y))`. Add this un-clipped slope to the chosen intercept then clip once. No label from outside the current prefix or from evaluation enters inference.

Four cells: base mean; local slope only; peer mean only; peer mean + local slope. All receive identical admissible evidence and target evaluationresponses. Cohort1 must replay R7, peer-only cohort4 must replay R12, including visible-config alias behavior. The local-only shared result may differ from isolated R7 where aliases create additional matched evidence; explain rather than silently comparing incompatible scopes.

## Mechanism tests and rejection

- Combined beats both single components with expected budget localization: evidence for complementary information.
- Combined loses peer-only: reject straightforward additive decomposition, inspect slope mean/calibration interaction before more components.
- Similar scores but extra complexity: do not claim useful combination.
- Report interaction `ALC(combined)-ALC(local)-ALC(peer)+ALC(base)`, each budget and fold, benchmark bootstrap.
- Gate comparisons include fixed smoothed2 and the stronger training-selected peer baseline in this conditional scenario. No conditional-only result counts as hosted or isolated-pair success; no promotion from cohort6 alone.

The semantic backend remains a research dependency. No deployment, external publication or competition submission is part of this experiment.
