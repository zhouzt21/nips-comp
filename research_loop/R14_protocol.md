# R14 / V2-05: audit dependence on which peers are available

Registered2026-10-22 before new cohort assignments. R13cohort4 combined ALC0.203771 vsbase0.208503 and peer0.204959. Its incremental item gain is consistent acrossfolds, but peer contribution regresses in fold2 relative to local-only. Cohort6 looks stronger; it is a stress condition, not permission to pick a better test scenario.

High-level uncertainty is now **information availability/composition**, not representation capacity. Previous trajectory seeds alter acquisition/evaluation items but all shared runs fixed cohortseed193. They do not establish robustness to peer composition.

Operator: audit the uncertainty unit -> intervene on allowed peer composition -> separate information-access sensitivity from predictor improvement.

Freeze R13 predictors and training selections. Primary cohortsize4 only; cohortseeds193(existing),271,811; itemseeds0/1/2. Equal average across these nine scenario draws within each benchmark, then benchmark-cluster paired bootstrap. Same total737targetpairs and36benchmarks. Report each cohortseed separately, allfolds, allbudgets, severe regressions. More draws do not create more independent benchmarks.

If peer gains reverse across cohortseeds or severe regressions emerge, prioritize input/evidence-dependent reliability over stronger global pooling. If complementary gains remain, candidate merits engineering work but still must pass0.005 against the strong simple peer control, original isolated gate and actual hosted evidence constraints. No retuning or choosing the best cohortseed.
