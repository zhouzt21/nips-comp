# R62–R64 单一只读结构复核

2026-10-22。使用既有 `low_budget_structural_reviewer`，理由是多层来源选择、迁移归档权重和variant计数的复现合同复杂，需要独立核对。不是双审，也不是隐藏泛化或提交验收。

## 第一轮范围与结论

核对本轮合同、refit/sampler/scorer/template代码、R54/R56历史选择及六个实际refit配置、R43归档权重。**无阻断问题，必须修项为零。** 当时R63/R64尚未完成，复核结果不覆盖其尚未产生的最终数字。

- R54/R56历史选择validation及新refit monitoring均与相应目标Benchmark分离；训练实现hash保持，epoch固定1/2/1。
- 两份归档R43权重hash一致，并分别等于原full refit epoch0/0/20；训练归一化仅fit来源拟合。当前`local_scale.py`对归档仅改变迁移后的import路径。
- interactor variants保留counts/successes；核对汇总item计数；六预算权重和pair×seed平均不变。
- B1 gate只读首支持标签；R63 epoch0是训练来源选出的整折固定策略。
- R64模板只用当前支持输入，固定.5平均；来源池重叠限制明确。

## 两项解释边界

1. R64置换仅在clipping前保持item等权残差均值，不保持response加权均值，也不保证clipping后均值。不得称为纯item对应关系的因果识别。主机制依据另有response加权精确损失差分解。
2. 原R43 full refit使用原shuffled names，R55 full refit使用half0+half1；R43/R55的差值不能只归因于重采样。R55 query与episode控制组保持相同顺序。

## 性能补丁限定复核

仅新增缓存代码变化触发第二次同一reviewer的限定复核。结论：无阻断。逐坐标原顺序相加与每步float32写回保持；FunctionType globals为独立副本；未修改原模型。

Reviewer建议将`np.array_equal`的数值一致扩展为dtype/shape/raw bytes一致。主agent已落实在`research_loop/audit_historical_cache_overlap.py`：7个原慢速完整chunk、57,344个query variants、2,867,200个数组值逐字节一致，见 `paiec_baseline/runs/research_loop/R62/cache_overlap_audit.json`。这项实际结果由主agent运行，不冒称由reviewer运行。
