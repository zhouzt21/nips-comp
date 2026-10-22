# R41: equivalence classes, rather than another encoder variant

Frozen after R40. Existing R18 sample already reports3355 distinct raw texts,3243 canonical texts,3108 token windows among3600items; merely repeating that count has little value. New question: do these collisions create measurable disagreement in the actual paired evaluation responses, and at which input-processing stage? Operator: Audit information representation -> decompose deterministic equivalence classes. No model, tokenizer or feature change.

Reuse all5242eligiblepairs, original three item split seeds, first64evaluationitems and ALL their responses. No support outcomes are needed; these are evaluation-label diagnostics explicitly separated from deployed scores. For each pair/seed form nested keys:
1. full visible item content, raw item_features and interactors (subject fixed within pair);
2. original canonicalized content plus raw metadata/interactors;
3. exact original tokenizer IDs (126content limit, first96+last30) plus raw metadata/interactors;
4. same token IDs plus exact R25 metadata vector; interactors not used by R25.
No item_id, source path or anonymous benchmark identity enters a model. IDs here only recover the original official data rows/split. Tokenization uses original offline revision; no new weights or external data.

For each key class with response count m and positives s, the best constant prediction fitted to THESE evaluation labels has empirical SSE s-s^2/m. Sum and divide by total responses. Compare successive class coarsenings; assert nonnegative loss increments. This is an in-sample oracle approximation cost, not a deployable Brier/ALC, population Bayes noise floor, or performance estimate. The difference of two oracle risks does not equal the gain a finite-label estimator can realize. Raw-input repeated-response noise and distinct-input collision loss remain separate.

Save per-pair/seed and per-benchmark summaries, counts of unique inputs and merged evaluation items; summarize pair-equal and benchmark-equal costs. Repeat raw/token key construction is deterministic. Tests: known group SSE, coarsening identity, row/response aggregation equivalence. Also record original R31 scores only for context, never mix this label-fitted diagnostic with them. No token-length/feature-field choices are selected by this diagnostic.

Decision: a substantial collision cost concentrated at a specific deterministic stage motivates one repair that preserves that missing observable information. Small cost weakens exact-collision explanation, but does NOT prove the embedding geometry retains all label information. Then reframe toward measurable neighborhood/sufficient-statistic quality rather than enlarging the encoder or invoking an irreducible-noise claim.
