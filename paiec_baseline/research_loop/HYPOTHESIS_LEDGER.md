# Hypotheses

<!-- current-v2-r29 -->
## 当前假设更新（R22–R29）

1. 小训练覆盖不是已证实的唯一根因：固定计算扩大池后仍不胜均值。
2. metadata确有部分局部信息，但统一正则混淆了新增信号；分块惩罚恢复小增益。
3. 小预算的先验均值方向偏差解释E3部分损失；对称化和同矩Beta均未解决。
4. 用源任务斜率的二阶统计迁移“有用方向”是合理但未获支持的假设：内层选择全关。
5. pair覆盖已审计并实际扩大：原代理改变收益幅度，但未解释全部未达标问题。64items及重复响应噪声分离尚未执行。
<!-- end-current-v2-r29 -->

| ID | Mechanism | Evidence so far | Discriminating action | Status |
|---|---|---|---|---|
| H-bias | Nontransferable pair prevalence dominates prior error | Bias squared0.099386, negligible/negative centered item gain | Evaluate robust priors; historical global/EB failed | Supported phenomenon, cause unresolved |
| H-sample | Tiny sampled pairs cause poor prior | Median1-3 rows/pair on large benchmarks | Prior pair-balanced trial did not reliably help | Not sufficient; scale not isolated |
| H-block | Shared hash coordinates cause cross-field interference | Blocks reduce old model deficit, still lose all baseline folds | R1 completed | Not sufficient |
| H-capacity | Nonlinear head learns unstable correlations | Linear pooled score improves old MLP, remains worse than mean | R1 completed | Not sufficient |
| H-parse | Structural noise/truncation obscures question content | Noise confirmed, chosen parser gives mixed fold effects | R2 completed | No stable performance support |
| H-local | Within-pair evidence can learn item-relative difficulty | R3 three-seed delta-0.000863, mostly15/31 | Fixed local ridge | Small supported effect, gate unmet |
| H-hidden | Supervised hidden features retain useful local information | Two folds select off, tiny pooled change | R4 completed | No practical support |
| H-episodic | Metric learning through adaptation improves over fixed metric | R5 difference-0.0000155, CI crosses0 | R5 completed | No evidence for expansion |
| H-covariance | Uncentered kernel can improve n=1 via similarity | R6 worsens B1/B3, loses matched smoothing | R6 completed | Chosen lexical covariance rejected |
| H-semantic | Semantic text similarity strengthens local difficulty relations | Current lexical gain small; model card provides semantic, not difficulty, training | R7 matched-token-window semantic versus hash | Active |
