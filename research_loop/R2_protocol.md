# PAIEC-LOOP-R2: structured content extraction

Registered after R1 results, before R2 training/scoring, 2026-10-22.

R1 blocks ALC0.212291, linear0.211939, baseline0.208331. Both lose all3 folds. Block versus old improves all3 but CI includes zero; linear's average benefit over old is fold-dependent. Neither field interference nor nonlinear capacity alone explains the baseline gap.

Input-only audit: 2731 sampled training-pool items,1429 JSON,640 exceeding640tokens,160 exceeding20000chars,495 with opaque resource references. See `runs/research_loop/content_audit.json`.

Single intervention: parse JSON content into textual leaves; prioritize question/user/prompt text, cap long contextual/system leaves, remove opaque location/ID leaves. Plain text unchanged. Retain all original subject/item-metadata/interactor encoding, shared2048 hash,128/64 MLP, response cap400, Brier, AdamW and residual2. This jointly tests structural text extraction, not separate effects of each parser rule; any benefit requires parser ablation later.

Same frozen outer folds, internal benchmark early stopping, seed2026, batch512, lr.001, wd.0001, max30/patience5; original MLP stopped before its historical12-epoch cap, so larger ceiling is not active for that comparator. Same seed0,100pairs/BM,64eval items, all response trials on selected items.

Primary comparison residual2 against smoothed2 and training-selected strengths2/2/4. Gate unchanged. B0, original MLP comparison, individual benchmarks and budget curves diagnose why. If this also fails, next question is whether local labels can learn an item-relative relationship instead of correcting only a global intercept. No embedding download or architecture expansion is justified by these results yet.

Research variants are evaluated with explicit feature metadata; no new submission ZIP is promoted before portable predict support and gate checks.
