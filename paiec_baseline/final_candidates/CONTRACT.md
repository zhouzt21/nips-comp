# Final candidate acceptance contract — 2026-10-23

## Frozen decision

User fixed semantic ridge 4 for both candidates before new evaluation. Metadata ridge 16; solve ridge 1; Partial8 private/shared regularization 8. No training, tuning, R79 gate, selection by outer outcomes, or Codabench submission.

## Three distinct evidence layers

1. Historical replay: every query probability, 5,242 pairs × 3 seeds × 6 budgets, original 4/4/1 scaling. Compare R25 and main cohort 7501 Partial8 against saved R76, R77 and R81 outputs. Historical R25 verifier scores are retained as an additional reference. Core absolute probability tolerance 1e-10; aggregate Brier/ALC tolerance 1e-10. Accuracy ties at 0.5 require explicit audit if floating-point roundoff changes thresholding.
2. Release local evaluation: same inputs, supports, queries, counts, successes, seed and main cohort 7501; fixed 4/4/4. Save new raw predictions separately. Report equal-pair Brier ALC, equal-Benchmark ALC, each fold, each budget Brier and accuracy (p>=0.5), seed variability, Benchmark-cluster interval and regressions >=0.005/0.03 ALC. Main cohort scope is conditional sharing simulation, not Hosted availability. Other cohort compositions are historical robustness evidence, not newly re-evaluated scores.
3. ZIP acceptance: bundled MiniLM revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41; exact structured text and metadata encoders; frozen operator function bodies. No dataset, embedding DB, external absolute paths or label persistence. Real text fixture sample chosen deterministically (first eligible seed-0 pair in each Benchmark; first four unique query variants, all budgets), plus synthetic interface cases. Compare cached historical features to CPU encoding (2e-6 absolute tolerance), and final ZIP probabilities to fixed-4 reference (2e-6). Fresh process repetition tolerance 1e-12. Record sample coverage rather than imply all raw texts were re-encoded.

## Adapter boundary

The frozen adapter requires a trusted visible Subject roster for sharing; infer no such roster from labels/hidden IDs. The public predict(input,labeled) signature supplies none. Candidate B therefore conservatively falls back in that interface. An explicit predict_with_roster diagnostic entry exercises the unchanged Partial8 adapter when legitimate roster evidence is provided; this is NOT an official Hosted API. Record NOT_READY for shared deployment until its runtime contract is resolved. Repeated modeled support entries disable sharing but retain own-response multiplicity in R25.

## Clean environment and resources

Independent Python 3.12 venv with exact declared dependencies, ZIP extracted outside repository, empty HF cache, blocked Python network/file audit, no inherited PYTHONPATH. Run official ZIP checker and accessible public smoke/client tooling; real Starting Kit/Hosted checks remain separate. Measure cold load, warm requests, full 31+3×31 support, fresh-process overhead and peak RSS; local reference ceilings: 60 seconds cold/request and 2 GiB RSS, not asserted organizer limits. Windows and Linux acceptance are labeled separately.

## Preservation and deliverables

New files only under final_candidates, dist, new report and external acceptance area. Snapshot hashes before/after for historical source/artifacts and protected fallback ZIPs. Large model/data/ZIP assets remain ignored by Git; source, hashes, metrics, reports and build commands committed/pushed. Missing acceptance yields NOT_READY, hosted_score=null. All official submissions=0.
