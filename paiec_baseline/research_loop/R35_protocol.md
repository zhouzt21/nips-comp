# R35: identify the gap between oracle capacity and support evidence

Frozen after R34, before new diagnostics. R34 gains only0.00004856 overR25 despite R33 max4 headroom0.005824. Source-selected tau=[1,10,10]; constant scales=[1,1,0.5]. No grid expansion.

Core competing explanations: (A) support-only scale statistic contains little information about which corrections should be amplified; (B) it contains useful high-budget signal but source ALC selection of one regularizer across budgets suppresses it because low-budget variance dominates. These may coexist.

Operator: Diagnostic -> Decompose -> decide whether differentiated treatment is justified. First join existing R33/R34 records by benchmark,subject,seed,budget. Measure rank correlation between the support-only coefficient and bounded oracle coefficient, separately by budget and benchmark; report amplification-decision agreement and actual R34-R25 losses. Oracle still uses evaluation labels; this is retrospective diagnosis, not deployed calibration. Seed rows are not independent inference units.

Then replay the unchanged R34 SOURCE episodes only to obtain per-budget Brier for the already frozen four tau choices. Verify their ALC against saved selections. Identify whether low/high-budget preference differs consistently across source folds. No outer tau sweep or new candidate scoring. Do not promote a per-budget best-tau table using this diagnosis.

Continue to one uncertainty-aware calibration experiment only if both source budget conflict and support-oracle association support it. Otherwise stop a scale-calibration parameter search and revisit the missing information. Even positive correlation does not prove that31 labels suffice to realize the oracle gap.
