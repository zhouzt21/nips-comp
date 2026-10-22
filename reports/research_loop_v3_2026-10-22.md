# PAIEC Loop v3 简明实验记录

最新结果与进入条件见 [证据和方法决策](research_loop_v3_findings.md)。

## Loop v3 当前状态

Authority: `doc/loop_2.md`，及四份相同目录的研究参考材料；取代旧loop_1的控制流程，保持所有已冻结实验纪律。R1–R31历史结果保留。

R32-R50 completed/audited. Current six-field decisions are in ITERATION_V3.md. No job is implied to be running by these files. Gate remains unmet; fallback unchanged. R48 uses extra labels, R49 failed source admission, pixel runtime contract unverified.

当前六字段研究判断与下一动作以 ITERATION_V3.md 为准。以下R32段落是已完成的测量审计摘要，不是仍待执行的任务。

全5242pair中5224受64-item限制；完整2179077queryitems/seed，原335247。R31收益77%来自预算15/31；B0/B1相同是当前算法的构造事实，不能直接推论输入不含冷启动信息。重复响应的经验within-item方差在两个预测器Brier差异中精确抵消，不能直接用该方差解释全部未达标差距。

R32先按benchmark×评价池大小×重复响应分层，从计数预先选择pair；每个被选pair完整评价pool，三个原种子、同一acquisition labels。分层外推与实际全pair评分分开。code/protocol/selection/log均在runs/research_loop/R32及research_loop/R32*。缺失语义向量只用原模型离线补D盘缓存。

研究Gate尚未通过；fallback不变；未正式提交。恢复时先检查最新轮次的进程、日志与summary，不要重复启动已完成的R32。日志存在不代表仍在后台运行，单benchmark结果不代表全轮完成。R33–35的幅度估计、R36–38的条件共享、R39的似然求解及当前输入分组研究的详细结论均见ITERATION_V3.md；历史v2段落不覆盖v3控制流程。

R32c已完成：主导benchmark全部4278pairs，三个原种子，全部evaluation items/responses。R25 full ALC=0.19066693872042262；smoothed2=0.1947777824570729。改善从capped约0.004122变为full约0.004111。所有旧capped逐pair/seed/budget回放通过。仅此benchmark不能替代36benchmark汇总；重复响应经验方差不是不可约噪声下界。

R32分层测量结论（不是全5242pair完整评分）：

{
  "hybrid_cap_effect": 1.8690490164230804e-05,
  "approx_design_ci95": [
    -0.00016079132519827036,
    0.000198172305526732
  ],
  "cap_corrected_gain_estimate": 0.0035447463018261067,
  "remaining_gain_to_gate_estimate": 0.0014552536981738934
}

完整六字段记录见 [ITERATION_V3](../research_loop/ITERATION_V3.md)；命令见 [复现入口](research_loop_v3_reproduction.md)。
