# R30: evaluation coverage audit, not a new performance benchmark

Frozen before computing the diagnostic. The current proxy caps each benchmark at100 pairs and each evaluation pool at64 items. Audit whether these caps substantially alter the weighting of the observed benchmark collection before more model fitting.

Use only eligibility counts from the previously frozen full public pair-rate cache plus frozen R25 three-seed benchmark mean scores. Report cap coverage and benchmark mixture. Reweight each observed benchmark mean by its full eligible pair count as a STRATIFIED EXTRAPOLATION, not a score from evaluating all pairs. No additional target labels enter any predictor, no parameters selected, and the frozen primary metric/promotion gate is unchanged. This sensitivity cannot establish performance on omitted pairs/items or hidden tests. Do not use it to promote a model.

Input-only follow-up condition: if caps substantially change weights, a subsequent actual larger evaluation could reduce finite-sampling uncertainty; if improvement remains small under both weightings, evidence favors a mechanism bottleneck over a simple aggregate-weight explanation. This remains observational evidence rather than a causal proof.
