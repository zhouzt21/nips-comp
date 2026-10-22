# PAIEC-LOOP-R3: support-conditioned item slope

Registered2026-10-22 before any R3 training/outer scoring. R2 first outer fold already fails the all-fold gate (structured0.187594 versus smoothed2 0.177406), while other folds continue unchanged. R1 and historical diagnostics show that global prior transfer remains unreliable.

## Mechanism

Source L4 (Bertinetto et al., arXiv1805.08136 sections3.1-3.2) motivates a cheap ridge base learner. This diagnostic does not claim to reproduce its learned image embeddings/meta-training.

Keep original item hash representation; subtract empty-subject contribution, normalize item vector to unit norm. No supervised global predictor and no anonymous ID feature. For each current support prefix of n examples, center features and labels within support. Let A be support features, Q query features, H=I-11'/n, K=A A', C=Q A'.

Fit a regularized local slope through the dual system:

`beta = (H K H + lambda I)^(-1) (y - mean(y))`

`p(query) = clip((1 + sum(y))/(2+n) + (Q-mean(A))(A-mean(A))' beta, 0, 1)`.

At n=0 or1, no slope is learned and prediction equals smoothed mean strength2. Thus this isolates information in **relative item features associated with current labels**; it is not another fixed mix of the failed global MLP prior. Normalization and support centering are fixed, not outer-tuned. Slope uncertainty can cause negative transfer; include an off option.

## Selection and paired evaluation

For each frozen outer fold, only its training benchmarks supply selection episodes. Inner item-split seed73, maximum25pairs/BM and64evaluation items/pair. Select ridge from off,16,4,1,0.25 by pair-equal six-budget ALC; ties prefer the earlier/stronger shrinkage option. Alpha remains2 for the ridge predictor. Also select a stronger simple baseline alpha from2,1,4,8,16 using these same training episodes. Finish and persist all3 selections before reading any R3 outer metric.

Outer: existing seed0,max100pairs/BM,64evaluation items. Compare frozen smoothed2, inner-selected smoothed, historical training-selected smoothed, empirical mean and old MLP. Assert every per-benchmark baseline Brier equals the existing replay within1e-7. Acquisition/evaluation distinct-item separation and one acquired trial per item exactly match existing evaluator.

Only support labels enter kernel_predict; evaluation labels enter scoring after predictions. B0/B1 must equal baseline exactly. Cache on D:/paiec_research_cache/R3 contains separately named support/evaluation labels and input-only kernels; version key includes code hashes and source revision. No external API inference or pretrained weights.

Gate remains unchanged. A meaningful high-budget benefit without gate completion would justify a later episodic representation objective; no benefit or training selection of off would question whether raw text similarity expresses local difficulty. Do not tune ridge using outer results.

## R3 verification extension (registered before new seeds)

Seed0 ALC0.207545 vs0.208331, fold deltas -0.001346,-0.001133,+0.000125; all training selections chose ridge1. Most benefit occurs at budgets15/31, not3. Freeze all selections and evaluate seeds1,2 in addition to0, same pairs/caps; bootstrap benchmark clusters after averaging seeds. This is trajectory robustness, not new benchmark evidence. No new model or parameter is selected from this extension.
