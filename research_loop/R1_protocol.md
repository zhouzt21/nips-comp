# PAIEC-LOOP-R1: field interference versus model capacity

Preregistered 2026-10-22 before candidate training/scoring.

## Why

Existing encoder already namespaces tokens. It is incorrect to claim it lacks field identity. It does mix all namespaces into the same 2048 coordinates, and repeated JSON/punctuation can consume substantial mass. An inspection of input-only examples found image paths, SQL schemas, and long system prompts. Structured parsing is a separate subsequent hypothesis; do not bundle it into this round.

## Candidates and controls

- R1-block: replace shared hashing by disjoint subject512/content1024/item-features256/interactors256 blocks. Same total dimension, tokens, normalization, two hidden layers128/64, Brier loss, AdamW.
- R1-linear: original encoding with a single linear sigmoid head. Tests whether nonlinear capacity is necessary or harmful. It does not isolate a single type of regularization.
- Existing original MLP weights per fold supply the historical controlled comparison. Compare training rows, inner validation directories, seeds and hyperparameters with metadata; report any mismatch.
- Constants and empirical, smoothed2, training-selected smoothed baseline remain mandatory.

## Frozen settings

Data revision54074453da7b9911be0f48e4c560be547bc84d21. Existing three `runs/mlp_cv/fold{0,1,2}_split.json` files. Train only outer-training benchmarks, with existing training-only internal validation. Seed2026, response cap400/BM, batch512, lr0.001, wd0.0001, max30 epochs, patience5. Early stopping optimizes inner pair B0 as historical code, not outer ALC.

Outer evaluation role=development, seed0, max100 pairs/BM, fixed64 evaluation items/pair, six official budgets, nested acquisition and response-level scoring; residual strength2 fixed. B0 and feature-only prior are diagnostics. No outer parameter search. If candidate suggests gain, confirm seeds1,2 without further tuning, then larger evaluation samples.

Outputs: `runs/research_loop/R1/{block,linear}_foldN/` weights, training metadata, logs, evaluations; summary includes every budget, fold, benchmark and paired bootstrap. Benchmark bootstrap is descriptive under repeated public dataset reuse.

Accept only gate in RESEARCH_STATE. Failure actions: if block fails, do not enlarge hash space by default; inspect whether semantic text extraction can recover question information. If linear fails, do not attribute all failure to nonlinear capacity. No architecture sweep.
