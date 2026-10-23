# plan_2 执行状态

更新：2026-10-22 18:58 +08。五候选任务已完成，当前无后台训练。授权原文 `doc/plan_2.md`；合同 `TRAINING_MECHANISM_V5_SPEC.md`；澄清 `TRAINING_MECHANISM_V5_ERRATA.md`；原文/作者代码表 `literature/training_mechanism_v5.md`。主报告：[结果与机制认识](../reports/training_mechanism_v5_findings.md)。

| 轮次 | 当前状态 | 已有证据 |
|---|---|---|
| R54 / 完整轨迹 | 已完成18fits，558个epoch checkpoint | 普通MLP三个source pools最佳epoch1/2/1；全部双方向满足过拟合且无稳定恢复的停止条件。R43 query最佳epoch0/0/20，无预定义迟发恢复 |
| R55 / 支持集重采样 | 已完成12fits×20epochs及四次预算内置换 | query选择epoch0/0/17，全部进入条件0/3通过。第三池相对固定query改善1.15e-5 ALC，但置换净差分+7.68e-4不支持新增对应信息 |
| R56 / Benchmark风险 | 已完成6fits×30epochs | λ=.5相对pair风险ΔALC为+0.0000110/−0.0002789/−0.0007302；完整进入条件0/3通过 |
| R57 / 平均 | 已完成R43两模型族×三平均方法×三个池评价 | 全部平均均差于最佳单一checkpoint，0/3通过；MLP无稳定窗口未平均 |
| R58 / 初始化 | 触发审计完成，未执行新fit | 0/3 source pools符合条件，故不执行输出尺度/LR/decay扫描；不是已训练负结果 |

数据输出根：`paiec_baseline/runs/research_loop/training_mechanism_v5/`。`core_provenance.json`/`source_snapshot`、`cache_inventory.json`、`completion_audit.json`提供依赖、输入与完成状态记录；`implementation_audit.json`记录数值对照。以上均为来源内探索性代理，不是5242pairs全外层或隐藏测试成绩。R25及提交fallback未替换。ADeLe继续等待用户资源。

合计36fits、996个epoch checkpoint，另24个EMA/SWA权重。新增19138条本地MiniLM向量；预处理607.5秒，全部fit记录303.5秒（不含抽样、置换、研究整理）。R55首次准备缺cache失败留痕，未产生fit；重试完成。`result_audit.json`核对18组干预与控制的初始化、fit数据、验证pair和更新步数；stage来源hash在复用前与结束后核对。此任务的执行完成，不表示原0.005 Research Gate已达成。
