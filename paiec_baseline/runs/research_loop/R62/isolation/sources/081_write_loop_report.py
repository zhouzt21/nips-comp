"""Build a checkpoint report/figure directly from completed research outputs."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path("runs/research_loop"); report=Path("../reports/research_loop_2026-10-22.md")
    r1=json.loads((root/"R1/summary.json").read_text())
    r2=json.loads((root/"R2/summary.json").read_text())
    r3=json.loads((root/"R3/summary.json").read_text())
    verify=json.loads((root/"R3_verify/summary.json").read_text())
    r4=json.loads((root/"R4/summary.json").read_text())
    rows=[("经验均值",r3,"empirical_mean"),("平滑均值 strength2",r3,"smoothed_mean_2"),
          ("历史训练内选择平滑均值",r3,"training_selected_smoothed"),
          ("本轮训练 episode 选择平滑均值",r3,"inner_selected_smoothed"),
          ("原 MLP + residual2",r3,"original:residual_2"),
          ("R1 分块哈希 MLP",r1["block:residual_2"],"block:residual_2"),
          ("R1 线性模型",r1["linear:residual_2"],"linear:residual_2"),
          ("R2 JSON 文本提取 MLP",r2["structured:residual_2"],"structured:residual_2"),
          ("R3 局部 ridge",r3,"local_ridge"),("R4 冻结 MLP 隐藏层 + 局部 ridge",r4,"hidden")]
    if (root/"R5/summary.json").exists():
        r5=json.loads((root/"R5/summary.json").read_text());rows.append(("R5 episodic metric",r5,"episodic_metric"))
    lines=["# PAIEC 持续研究检查点：2026-10-22", "",
           "执行依据：项目根目录 loop.md。以下都是实际公开数据实验，Research Gate 尚未达到；没有 Codabench 提交。",
           "", "## 证据范围与固定条件", "",
           "数据 revision `54074453da7b9911be0f48e4c560be547bc84d21`。沿用原训练池的三折 benchmark holdout，共737个抽样 pair、36个可评分 benchmark。每 benchmark 最多100pairs，每 pair 最多64evaluation items；重复 response 按记录保留。下表同为 item split seed0。原开发集/本地测试没有用于新一轮选择。公开 outer folds 已反复分析，不能称为新盲测。",
           "", "门槛保持不变：对冻结平滑均值和训练内选择基线，pair-equal ΔALC≤−0.005、三折均改善、benchmark-equal改善、benchmark聚类bootstrap95%上界<0，并排查>0.03的单benchmark退化。统计区间未修正整个连续研究过程的适应性选择。",
           "", "## 同条件结果（seed0）", "",
           "| 方法 | B0 | B1 | B3 | B7 | B15 | B31 | Pair-equal ALC | Benchmark-equal ALC |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for title,data,key in rows:
        nums=[*data["pair_equal_brier"][key],data["pair_equal_alc"][key],data["benchmark_equal_alc"][key]]
        lines.append("| "+title+" | "+" | ".join(f"{x:.6f}" for x in nums)+" |")
    lines += ["", "训练内选择的简单方法并不保证外层更强：历史近似风险选择为2/2/4，本轮实际 training-episode 选择为4/2/4，后者外层更差。因此始终保留 strength2 作为必须战胜的对照。", "",
              "## 多种子复核：R3", "",
              f"固定 ridge1，seeds0/1/2：局部 ridge ALC **{verify['pair_equal_alc']['local_ridge']:.6f}**，同条件平滑均值 **{verify['pair_equal_alc']['smoothed_mean_2']:.6f}**。",
              f"ΔALC={verify['comparisons']['smoothed_mean_2']['pair_equal_delta']:.6f}；三折差值={verify['comparisons']['smoothed_mean_2']['fold_deltas']}；benchmark聚类95%CI={verify['comparisons']['smoothed_mean_2']['cluster_bootstrap_ci95']}。先在各benchmark内平均随机种子，再bootstrap；未把2211次轨迹当成2211个独立pair。", "",
              "## 机制判断", "",
              "1. **分块哈希只能减轻旧 MLP 的问题。** 三折均改善旧模型，但三折均劣于平滑均值。B0/B1仍贡献多数差距，不能把共享坐标碰撞认定为唯一根因。",
              "2. **降低模型容量和解析 JSON 均未解决先验失效。** 线性模型及结构化提取的总分仍较差；后者在不同折的方向不同。输入审计发现的结构噪声是真实存在的，但尚无证据证明当前清理方式能够稳定改善目标。",
              "3. **局部 item 信息有小幅价值。** R3保持B0/B1与平滑均值完全相同，在15/31预算改善最明确。它通过当前标签学习斜率，而旧residual只修正截距。这是当前较可信的新信号，但幅度不足，第三折仍略差。",
              "4. **旧 MLP 隐藏层也没有显示实用优势。** R4内层在两折关闭其斜率；结果接近原平滑均值，未恢复强item区分能力。内层仅37/33/33pairs，选择不确定性也需保留。",
              "5. **后续优先检验直接适应目标。** R5仅学习正则化的对角特征权重，通过本地ridge求解器优化六预算损失。对照固定相同特征权重，以区分学习本身的增益；不能将与旧400rows/BM模型的差异全部归因于目标函数。", "",
              "## 文献如何改变实验", "",
              "- [Feature Hashing](https://arxiv.org/abs/0902.2206)解释了signed hashing和碰撞方差；没有保证分块优于共享空间，因此将分块视为待否定假设。",
              "- [tinyBenchmarks](https://arxiv.org/html/2402.14992v2)的item参数依赖已知benchmark响应矩阵，不能直接移植到匿名新benchmark；采用其区分偏差与采样方差的思路，未照搬item查表。",
              "- [Differentiable closed-form solvers](https://arxiv.org/abs/1805.08136)支持用小规模对偶ridge研究少标签适应；R3仅测试局部求解器，R5才测试通过求解器学习表示。",
              "- [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232)支持将冻结表示与新局部头拆开比较；其图像分类结果不等同于PAIEC的Brier证据。", "",
              "## 实际产物与复现", "",
              "- `research_loop/` 保存每轮事前协议、状态、假设、文献证据和恢复说明。",
              "- `runs/research_loop/R1,R2/` 保存真实权重、训练轨迹、命令、耗时、代码快照、逐benchmark/预算CSV、bootstrap汇总。",
              "- `runs/research_loop/R3/` 保存训练内参数选择与配对结果；`R3_verify/` 为冻结参数的三种子复核；`R4/` 为冻结隐藏层实验。",
              "- 核心入口：`python -m mlp.run.run_loop_round --round R1`（或R2）、`python -m mlp.run.run_local_round`、`python -m mlp.verify.verify_local_round`、`python -m mlp.run.run_hidden_round`、`python -m mlp.run.run_metric_round`。从工程根目录运行，数据入口固定D:/paiec_payload/data，缓存D:/paiec_research_cache。",
              "- 为避免混用不同代码版本，R1/R2 runner遇到已存在但代码不同的快照会拒绝继续；复现实验应使用该轮快照及已保存commands.jsonl，在独立副本运行。R3–R5的配置和额外代码快照见各轮目录。",
              "- 预测/训练一致性、不同预算证据隔离和双形式ridge测试在 `../tests/test_loop_models.py`、`../tests/test_local_ridge.py`、`../tests/test_episodic_metric.py`；合成测试只验证实现，不代替上表真实实验。", "",
              "## 提交决策与未完成项", "",
              "`dist/smoothed_mean_strength2.zip`继续作为冻结fallback。所有新方法仍为研究候选；未覆盖fallback，也没有把全公开重训权重归属到某个未见benchmark分数。独立新benchmark/隐藏测试证据尚缺。继续按照最新RESEARCH_STATE执行下一轮，不能把本检查点当作最终研究成功。", ""]
    report.write_text("\n".join(lines),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
    x=np.arange(6);labels=[0,1,3,7,15,31]
    for title,data,key in [rows[5],rows[6],rows[7],rows[8]]:
        delta=np.array(data["pair_equal_brier"][key])-np.array(data["pair_equal_brier"]["smoothed_mean_2"])
        english={"block:residual_2":"Block hash MLP","linear:residual_2":"Linear","structured:residual_2":"Structured MLP","local_ridge":"Local ridge"}[key]
        axes[0].plot(x,delta,marker="o",label=english)
    delta=np.array(verify["pair_equal_brier"]["local_ridge"])-np.array(verify["pair_equal_brier"]["smoothed_mean_2"])
    axes[1].plot(x,delta,marker="o",color="tab:green",label="Local ridge, 3 seeds")
    for ax in axes:
        ax.axhline(0,color="black",lw=.8);ax.set_xticks(x,labels);ax.set_xlabel("Label budget")
        ax.set_ylabel("Brier difference vs smoothed mean 2");ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_title("Matched public replay (seed 0)");axes[1].set_title("Frozen-parameter trajectory check")
    fig.savefig(root/"budget_deltas.png",dpi=180);fig.savefig(root/"budget_deltas.pdf");plt.close(fig)
    print(report)


if __name__=="__main__":main()
