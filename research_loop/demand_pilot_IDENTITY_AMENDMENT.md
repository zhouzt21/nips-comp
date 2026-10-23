# R74 identity-check amendment, before any fitted model or new result

The first invocation failed at `reference_fold` on pool0 before `save_student` or any response fit. The original `contract/` remains immutable. No teacher/student effect score was produced or inspected.

The failed assertion equated R66 `query_ids` metadata with R59 `query_groups` variant array order. Inspection of the original `mlp/training_mechanism_data.py:85` raw sampler establishes that `query_ids` always records randomized unique eid order, while `include_mlp=True` query/counts/successes arrays are overwritten with `groupby(item_id, interactor_variant, sort=False)` order. R66's `run_transferable_prior_ablations.py` uses precisely these arrays and explicitly checks R59 counts/successes and R25 scoring. Thus the metadata has a different ordering contract; this is not evidence of a historical probability ordering error.

Repair: assert R66 `query_ids` equals source episode unique query item order; retain exact R59/R66 counts/successes equality and support identity equality; use R59 explicit `query_groups` to map new item probabilities to response variants; check the R25 re-evaluated probabilities and scoring. Freeze repaired code plus original sampler and original R66 runner under `contract_v2/` before execution. Model/data/hyperparameters/entry rules unchanged.
