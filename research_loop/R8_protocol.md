# PAIEC-LOOP-R8: semantic MLP prior versus matched retained-text MLP

Registered2026-10-22 before R8 training. R7 semantic local ridge0.207180 improves all3 folds versus0.208331 but by only0.001151. Semantic versus retained-text hash delta-0.000575 has a benchmark-cluster CI crossing0. Semantic local evidence is promising but not proven sufficient; B0/B1 remain unchanged by construction.

Question: does the same frozen semantic representation support a transferable zero-label MLP prior? This differs from R7, which learned only from current local labels.

Two matched modes: subject512/item-content1024/item-metadata256/interactors256 blocks (total2048). The content block is either a unit-normalized hash of the exact retained token window or the frozen384-dimensional MiniLM embedding followed by640zeros. Subject and other fields unchanged between branches. No benchmark ID feature. Same pretrained tokenizer/window as R7; canonical content identical for both. This is a representation comparison, not an architecture search.

Same original MLP128/64, Brier, AdamW, response cap400/BM, seed2026, batch512, lr0.001,wd0.0001,max30/patience5, benchmark-disjoint internal validation. This deliberately retains earlier sampling to isolate representation; it does not assert that400rows/BM is optimal. Export torch/NumPy agreement required. No pretrained encoder fine-tuning.

Outer seed0,max100pairs/BM,64evalitems. Primary method residual2; prior-only B0 diagnoses transfer. Compare both new branches, original MLP, frozen/selected mean baselines. Gate unchanged. If semantic prior is still unreliable, do not assume larger MLP fixes it; preserve local semantic method and investigate evidence availability/base-rate estimation.

R7 verification is separately frozen before new seed scores: same selected ridge per fold, seeds0/1/2, no parameter changes; bootstrap benchmarks after averaging seeds. No new-blind-benchmark claim.
