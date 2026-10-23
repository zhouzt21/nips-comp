# R59–R61 验证与证据边界

## 已完成的直接核对

`audit_low_budget_composition.py`基于真实预测与成功/失败计数独立累加平方误差，核对四项分解、作者评分核的常数/二分类归一化、所有门控输入及非B1预测不变。实际机器结果见 `paiec_baseline/runs/research_loop/R61/validation.json`；这不是新增训练或新的效果评价总体。

- 824次来源池pair记录、52,274个response groups（均含来源池重叠）。
- V5冻结回放最大误差约2.02e-8，小于评分前1e-7；独立算术误差≤9.03e-16。
- R59已选epoch来源标签重用，诊断性预算重组、oracle、相关性和bootstrap均不称独立验证。
- R60候选唯一，无拟合/阈值扫描；R61先保存预测再加载评价计数进行评分，门控只用首个支持标签的平方误差。
- 来源门槛失败，因此外层`condition_not_met`；不自动提交或替换fallback。
- 72个实际读取的MLP validation/semantic source缓存与V5已有inventory哈希逐项一致，清单保存R59/used_input_cache_inventory.json；本轮没有重建或变更这些输入。
- 已打开检查两张PNG：分解和配对区间图文字/坐标/图例完整，无裁切；对应SVG同时保存。

## 单一只读结构复核

触发理由：本轮有重复response/interactor variant对齐、冻结checkpoint的选择偏差、query标签oracle与合法首标签门控的隔离、不同论文Brier常数约定及条件进入外层等复杂科研合同。确定性算术检查不能独立判断这些证据边界的表述是否一致，因此按AGENTS.md使用一个只读structural_reviewer；不启用双审或Quality Judge。

复核范围：执行合同、R59回放脚本、R60/R61候选脚本、原始指标、主报告和来源映射。复核状态：完成，无阻断问题、无必须修项。reviewer为本会话独立只读代理`/root/low_budget_structural_reviewer`，没有写文件、训练或跑全套测试；不是双审。

复核确认：variant对齐保留重复response贡献；低预算权重为原.1/.2；首标签门控和B1残差不读取未来/query标签；R60冻结先于R61评分；外层condition_not_met及来源选择偏差表述一致；作者Brier核的(raw+1)/2转换正确。

唯一P3可选意见为“题内variation”容易误读为同一道题内变化，已在报告及生成脚本改为“pair内变化项”，正文明确包含各query item/interactor variant差异。此为文字修正，未改分数或程序。

## 其他可复现性说明

当前工作树没有`dist/smoothed_mean_strength2.zip`；在`D:/nips-comp/paiec-dataset/paiec_git_backups/`的`layout_20261022_120427`和`nips-comp_20261022_113534`两份归档内均核对到原包，SHA256均为`E55EE0B267F3B345685EF072CFA511C3C017D9082256573793961EA20FEADCB9`。本轮未创建、移动或替换提交包；“fallback保持”指原方案和已归档包保持。

初次文献PDF解析发现迁移Python没有pypdf/fitz，随后使用已有TeX Live pdftotext；没有变更环境。AnySearch参数调用异常与修正已在来源表披露，未计作任何效果实验。所有真实评分已完成，无资源中断fit，因为本轮新增fit为0。
