# Literature evidence ledger

<!-- current-v2-r29 -->
## R27–R29 文献用途

- Sheen & Mei, An elementary approach for minimax estimation of Bernoulli proportion in the restricted parameter space，https://arxiv.org/html/2009.11413v2 。AnySearch全文检索，已读intro Eq1、Section2；raw/v2_binomial_minimax.md。使用其中回顾的非受限iid参数minimax公式；不转移其受限参数新定理，不将参数风险与预测Brier混同。R27是实测负结果。
- Koenker & Gu, Empirical Bayes For the Reluctant Frequentist，https://arxiv.org/html/2404.03422v1 。AnySearch全文，已读Sections1–2；raw/v2_empirical_bayes.txt。共同采样模型、交换性和mixing distribution relevance构成适用条件。R28对称化是本地干预，不是该文在PAIEC的结果。页面arXiv标签与文内version日期不同，保留原文，不猜测版本历史。
- R29复用已核实ALPaCA的末层先验概念；没有引用其连续Gaussian回归表现来证明本任务有效。真实选择结果为关闭协方差。
<!-- end-current-v2-r29 -->

Retrieved with AnySearch on 2026-10-22. Raw search/extract outputs are in `literature/raw/`. The file `targeted_search.json` contains CLI Markdown, not machine-readable JSON. No API credentials are recorded.

## L1: Weinberger et al., Feature Hashing for Large Scale Multitask Learning (2009)

Primary paper: https://arxiv.org/abs/0902.2206 ; full text read at https://ar5iv.labs.arxiv.org/html/0902.2206 (sections2-3, raw `feature_hashing_full.md`).

- Source-supported: signed hashing estimates inner products without the bias of unsigned hashing; variance depends on dimension and feature mass. Independently hashed task subspaces can have bounded interference under assumptions. Application is personalized spam classification.
- PAIEC inference: the current signed hash is a legitimate baseline, and token namespaces already distinguish fields. Disjoint blocks remove cross-field collisions but spend fewer coordinates on each field, potentially increasing within-field collisions. No theorem here promises better OOD Brier.
- Decision: R1 block-hash is a falsifiable intervention, not an established improvement. Keep total2048 dimensions and all token weights unchanged.

## L2: Polo et al., tinyBenchmarks: evaluating LLMs with fewer examples (2024)

Primary full text: https://arxiv.org/html/2402.14992v2 ; read sections3,4.1-4.4,5 (`tinybenchmarks_full.md`).

- Source-supported: item discrimination/difficulty parameters are fitted from historical model-by-item response matrices; a new model's ability is estimated from a small anchor set while item parameters stay fixed. gp-IRT combines model estimates and observed outcomes using a bias-versus-sampling-variance heuristic.
- Limitation for PAIEC: item parameter training assumes prior responses on those benchmark items. This is new-model evaluation on a known item pool, not unseen-benchmark cold start. Its accuracy claims are not evidence that an ID-based IRT model is valid here.
- Transferable mechanism: separate uncertain global predictions from low-sample local evidence; estimate reliability on training-only held-out groups. A fixed prior mix already failed here, so a direct implementation of the same shrinkage formula is not a new experiment. Any follow-up must provide a different information source or conditional reliability evidence.

## L3: Official PAIEC task and rules

https://aims.stanford.edu/competition ; AnySearch extraction `official_rules.md`.

Confirmed six budgets, distinct-item acquisition/evaluation isolation, response-level scoring and pair-equal ALC. No inference-time web search. Prepackaged models require disclosure. Final code/report deadline October30 2026 23:59 AoE. Current implementation uses a bounded public proxy (100pairs/BM and64evaluation items/pair), which must remain explicit.

## Leads not accepted as method evidence

- Pliakos et al. (2019), DOI10.1016/j.compedu.2019.04.009: retrieved bibliographic/abstract lead only; cold-start educational IRT, original mechanism not yet inspected.
- RIDE arXiv2511.04120 and Using Vision + Language Models to Predict Item Difficulty arXiv2603.04670: search-level leads only. Human difficulty and known-task synthetic rewriting do not directly establish PAIEC transfer. No algorithm decision or quantitative claim based on them.

## L4: Bertinetto et al., Meta-learning with differentiable closed-form solvers (ICLR2019)

Primary text: https://ar5iv.labs.arxiv.org/html/1805.08136 ; read sections3.1-3.3 (`ridge_full.md`). AnySearch academic.preprint located the original after academic.search returned a replication first.

- Source-supported: a ridge-regression base learner adapts within an episode; the dual solution solves an n-by-n system instead of a feature-dimension-sized system. Outer episodic training learns feature representations and solver hyperparameters. A scale and bias calibrate regression scores for classification.
- Transferable mechanism: with at most31 labels, a regularized local slope is computationally cheap and can use item-level evidence that an intercept-only residual cannot use.
- Critical difference: an initial PAIEC diagnostic using frozen hash features and training-only selection is **not a reproduction of learned R2D2 representations**, nor is its image-classification evidence proof of probability calibration. A useful first test would add a centered local ridge correction to a fixed smoothed prevalence estimate, then verify Brier and negative transfer.
- Entry condition: use after the field/capacity and structural-content diagnostics clarify that a global prior remains unreliable. Include zero slope as a training-selectable fallback; never select ridge strength using outer scores.

## L5: Chen et al., A Closer Look at Few-shot Classification (ICLR2019)

Primary text verified directly: https://ar5iv.labs.arxiv.org/html/1904.04232 ; read sections3.1-3.2 and cross-domain motivation. AnySearch initial title search ranked a later similarly named paper first; that later paper was not substituted for this source.

- Source-supported: train a feature extractor on base classes, freeze it and fit a new classifier using novel-class support examples. The study emphasizes matched implementation and explicit cross-domain evaluation.
- Transfer to PAIEC: tests whether a failed pretrained probability head and its hidden representation can be separated. R4 compares a locally refitted head on frozen MLP activations against the same head on raw inputs.
- Boundary: their classification protocol does not establish Brier calibration or benchmark-cold-start reliability here. A learned episodic metric (L4) is a subsequent hypothesis if the supervised frozen representation is unhelpful.

## L6: Harrison, Sharma, Pavone, Meta-Learning Priors for Efficient Online Bayesian Regression

Primary full text: https://ar5iv.labs.arxiv.org/html/1807.08912 ; read sections2-4, including Gaussian observation assumptions, conjugate last-layer regression, and the equivalent kernel interpretation. Raw `bayesian_local_full.md`.

- Mechanism: a learned feature map and a prior over last-layer weights provide fast conditional updates. The paper studies continuous regression with Gaussian observation noise.
- R6 takes only the idea of distinguishing constant uncertainty from item-dependent covariance. Its clipped binary probability is an approximate regression rule, not an exact posterior probability or a reproduced ALPaCA model. The matched smoothing control was essential: actual R6 B1/B3 worsened and the covariance did not outperform its shrinkage-only counterpart.

## L7: Official all-MiniLM-L6-v2 model documentation

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 ; full model card extracted with AnySearch (`minilm_model_card.md`). Hugging Face API pinned revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Weights downloaded to D:/paiec_pretrained/all-MiniLM-L6-v2.

- Source-supported:384-dimensional sentence embeddings, attention-mask mean pooling and L2 normalization; contrastive training on sentence pairs. Default inference truncation256wordpieces; original training sequence limit128.
- R7 uses a fixed128-token window for CPU cost, with identical tokens supplied to its lexical control. It tests semantic similarity as a source of local information, without assuming similarity implies difficulty. Non-English and non-text benchmarks remain in evaluation.
- Permitted pretrained model use is distinct from competition-specific training data. No new competition labels or extra fine-tuning dataset are introduced. Hosted dependency availability and portable packaging remain separate gates.
# v2 evidence additions — 2026-10-22

- **Ruan et al., Observational Scaling Laws and the Predictability of Language Model Performance, arXiv2405.10938v2.** AnySearch search/fullHTML, methods3.2/3.4 inspected. Main analysis uses77base models,8standardized benchmarks and<1%missing values; PCA capabilities plus task-specific regression. Search abstract from another version says~100, v2abstract~80: retain actual version differences. https://arxiv.org/html/2405.10938v2 . **Conceptual/structural inspiration:** relative capabilities deserve an overlap audit, but our sparse heterogeneous table does not justify importing their PCA law. R15/R16 use only allowed public data, no external scores. Fixed scalar ability and semantic filtering failed.
- **Official measurement-db metadata/README at pinned5407445 revision:** AnySearch README retrieved; local61benchmark tables reveal54higher/4lower/3unordered label directions. This is a measurement-contract fact, not proof of what hosted ingestion does. R16 preserves all target labels and tests source-only filtering. https://huggingface.co/datasets/aims-foundations/measurement-db/blob/main/README.md .

- **User reference mapping:** High-level A08/A09 and MN-P32 suggest distinguishing representations, access and use. These are explanatory prompts, not a theorem about PAIEC. Operator source: `常见的组合思路.md`, sections3 and7; selected audit -> exact diagnostic -> correspondence control, before differentiated treatment.
- **Chan et al., Transformers generalize differently from information stored in context vs weights (2022), arXiv2210.05675.** AnySearch search plus full HTML extraction; methods/results sections2/3 inspected. Controlled two-feature partial-exposure tasks distinguish rule versus exemplar generalization; pretraining structure changes the context result. Source: https://arxiv.org/abs/2210.05675 . **Conceptual inspiration only:** information source and learning conditions must be isolated. Does not prove that local ridge wins over MLP or that context is always superior. R7/R8 have additional confounds, so R10 uses matched fixed kernels instead. Raw `literature/raw/v2_context_weights_*`.
- **Wu et al., Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment (2019), arXiv1903.01689.** AnySearch search/full HTML; preliminaries and propositions3.1/3.2 inspected. Exact latent distribution matching can conflict with differing label proportions; target-label conditional terms require assumptions. Source: https://arxiv.org/abs/1903.01689 . **Boundary evidence:** a good representation or marginal alignment is not sufficient for correctness transfer. Its deterministic-label domain-adaptation setting differs from stochastic PAIEC responses and does not establish empirical impossibility here. Raw `literature/raw/v2_shift_full.md`.
- **Bob Carpenter, Hierarchical Partial Pooling for Repeated Binary Trials (2016), official Stan case study.** AnySearch full HTML; pooling and beta hierarchy sections inspected. Source: https://mc-stan.org/learn-stan/case-studies/pool-binary-trials.html . **Transferable with modification:** pooling presumes a population model and can have poorly identified heterogeneity with few groups. R11 subsequently implemented the hierarchy; R12 strength-matched controls did not justify additional integration or heterogeneity complexity. The earlier deferred status is superseded.

- **Complex-to-Real Sketches for Tensor Products with Applications to the Polynomial Kernel, arXiv2202.02031.** AnySearch academic search and full HTML extraction, introduction and section2 equations1–2 inspected. https://arxiv.org/html/2202.02031 ; raw `v2_polynomial_kernel.md`. Source supports the tensor-product identity for homogeneous integer-degree kernels. R21 uses an exact degree4 Gram matrix with current support, not the paper's randomized sketch or its claimed application gains. The mathematical identity was additionally checked against explicit tensor features. Actual PAIEC geometry benefit remains uncertain; do not cite the paper as performance evidence.

