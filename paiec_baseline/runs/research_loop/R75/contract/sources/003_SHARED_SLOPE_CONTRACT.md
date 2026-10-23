# R75/R76 Track B frozen execution contract

2026-10-23. Exploratory `conditional_simulation`; no Hosted assertion. Original authority: `doc/plan_8.md` and `DUAL_TRACK_SPEC.md`.

## Data, comparison and selection

Keep R64 item/response identities, counts, successes, all query variants and three split seeds. Read R72 context records; only support labels enter predictors. R25 geometry is MiniLM/sqrt(source ridge) concatenated with metadata/4. Brier/accuracy and ALC use unchanged historical scoring. No external labels, new training data, or parameter search.

For each outer fold, source pilot uses only that fold's source Benchmark list, at most 8 eligible unique visible Subject configurations per Benchmark, ordered by a label-independent hash, with seeds 0/1/2. Source pools overlap: folds are parameter-selection contexts, not independent replications. Deduplicate Benchmark when reporting combined source evidence. Source outcomes cannot promote themselves as outer evidence.

## B2 frozen cohorts and uncertainty

Main random cohort: per Benchmark, permute eligible full visible Subject keys with SHA256-derived seed `7501|benchmark`; partition into groups of up to four. Second composition uses `7502|benchmark`. This uses no outcomes. For an own Subject, peers have the fixed order in its group. Full scenarios: 0, first 1, first 2, first 3 peers; independently missing peers (hash coin, fixed across budgets); uneven peer budgets with prefix lengths n, floor(n/2), floor(n/4); second composition 3 peers. Full visible duplicate configurations are excluded from sharing, as in R72. Duplicate/conflicting identities also receive explicit synthetic adapter cases; forced ambiguity full scenario is exact R25.

Heterogeneity is diagnosed at B31 on source seed 0, first complete selected group per Benchmark. Split each Subject's acquired sequence into even/odd halves (16/15). On common input anchors, compute centered half-slope predictions. Statistic H is mean across Subject pairs of the cross-half product of their prediction differences. Positive within-Subject half cross-products indicate repeatability. Compare H with 32 parametric Bernoulli bootstrap replicates generated from a fitted common slope plus separate empirical Subject intercepts. This is a small exploratory model-based noise diagnostic, not a consistent test of true latent heterogeneity; fitted-null misspecification remains.

B4 entry for each outer fold: at least two distinct source Benchmarks have observed H above the 95th percentile fitted-null H AND positive within-Subject repeatability; additionally source Benchmark-equal mean H exceeds the 95th percentile of the corresponding null mean. No outer query result enters this decision. No statement of identical slopes follows from a failure to enter.

## B3 finite factor controls

Same source cohorts, data and queries: own/pool under semantic-only and semantic+metadata representations at R25 scaling; pooled per-Subject centering versus global centering; and ridge 1 versus original R36 ridge expressed in the same feature units (r20/r25). The global-centering control retains own smoothed mean but centers pooled X/y globally; it can learn between-Subject shifts. Original R36 unscaled-semantic predictor is checked against its equivalent scaled-semantic ridge. No grid and no outer selection.

## B4 fixed conditional candidate

If entry criterion passes, exactly one model: minimize sum squared centered residuals plus ||beta_shared||^2 + 8*sum ||delta_s||^2. Dual kernel is XX' * (1 + I_same_subject/8), with observation ridge 1. Predict own smoothed mean + own-centered query*(shared+own delta), clip [0,1]. If fold entry fails use fully shared slope; if no peers or own B0 use R25. Peer-label permutation control uses fixed outcome-independent support index permutations. No shrinkage scan. Stop this structure if source paired Brier does not improve fully shared; full conditional candidate is evaluated only when both entry and source gain hold.

## B5 full evaluation and artifacts

All predeclared cohort scenarios receive 5242 pairs x 3 seeds x 6 budgets. R72 same3/pooled3/diff3 are reused as the explicitly old composition, never relabeled as random cohorts. Persist every new probability matrix plus index/counts/successes, source-selection records and cohort mappings. Independently replay Brier/accuracy and identity hashes. Report pair/Benchmark equal ALC, folds, per-budget, clustered 10000-replicate CI, alignment versus rest, worst regression, actual extra labels, CPU wall time and process memory. Frozen means, R25 and R56 remain controls. No favorable-cohort picking.

Raw adapter uses only current input and labeled entries, exact full visible Subject configuration, current item content/metadata/interactors and injected stateless encoder/fallback. Missing/incomplete or conflicting configuration evidence causes own-only fallback; unknown external collisions cannot be detected from identical visible dictionaries and remain an explicit contract boundary. It ignores Benchmark names and internal IDs, has no persistent label state, and is smoke-tested only on synthetic inputs.

CPU threads <=4. Target source stage <=15 minutes and full prediction <=45 minutes; actual times recorded, no human time inferred. Frozen sources and commands captured before execution. No R1-R72 files modified.
