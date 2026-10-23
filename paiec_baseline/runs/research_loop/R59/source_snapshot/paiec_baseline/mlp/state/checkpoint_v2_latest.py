"""Current evidence header; historical sections remain available below."""
import json
from pathlib import Path


def prepend_latest(history):
    root=Path('runs/research_loop')
    def read(r):return json.loads((root/r/'summary.json').read_text(encoding='utf-8'))
    r25=read('R25_verify');r20=read('R20_verify')
    lines=['# PAIEC v2 当前实验检查点：已执行至 R31','',
      '日期：2026-10-22。遵循`doc/loop_1.md` 与四份参考文件。以下为真实运行结果，不是新增盲测；未进行 Codabench 提交。后面的 R10–R25 章节保留历史推理链，当前结论以本节为准。','',
      '## 当前模型决策','',
      '- **继续保留 smoothed_mean_strength2.zip 为正式提交备选。** 新方法尚未通过冻结的研究门槛；不能用最好点估计替换成功判据。',
      f"- 独立 pair、固定三种子：R25 分块 metadata + 局部语义 {r25['pair_equal_alc']['separate']:.6f}，语义 R7 {r25['pair_equal_alc']['semantic']:.6f}，平滑均值 {r25['pair_equal_alc']['smoothed_mean_2']:.6f}。",
      f"- 条件共享 cohort4、固定三种子：R20 共同 item 斜率 {r20['pair_equal_alc']['pooled']:.6f}，强 peer 均值 {r20['pair_equal_alc']['peer']:.6f}。不可与独立 pair 分数直接混排。",
      '- R25 相对均值改善约0.00146；R20相对强peer改善约0.00212，均低于0.005。共享证据规模、跨subject池对齐方式尚无 hosted 验证。',
      '- 本地36个外层benchmark已经反复参与研究；重新划分或增加seed不能制造独立泛化证据。HF实查revision未变化，见runs/research_loop/release_check.json。','',
      '## 独立 pair：三种子逐预算比较','',
      '| 方法 | B0 | B1 | B3 | B7 | B15 | B31 | ALC |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,r,key in [('经验均值',read('R27'),'empirical_mean'),('平滑均值2',r25,'smoothed_mean_2'),('训练内选择平滑',r25,'training_selected_smoothed'),('R7语义',r25,'semantic'),('R25分块metadata',r25,'separate'),('R27预算相关均值',read('R27'),'budget_mean'),('R28对称经验先验',read('R28'),'symmetric')]:
        lines.append('| '+label+' | '+' | '.join(f'{x:.6f}' for x in [*r['pair_equal_brier'][key],r['pair_equal_alc'][key]])+' |')
    lines+=['','### R25 不确定性与证据范围','',f"对固定均值：`{r25['comparisons']['smoothed_mean_2']}`",'',
       f"对R7：`{r25['comparisons']['semantic']}`",'',
       '先在benchmark内平均种子，再按benchmark聚类bootstrap；不是把2211个pair-seed当成独立任务。区间没有校正整轮自适应研究的选择偏差。相对R7的第三折略退化，保留该事实。','',
       '## R26–R29：新增否证与停止的具体路线','']
    descriptions={
      'R26':('active','random','在线特征方差获取','固定R25预测器，仅改变标签获取。改善主要伴随局部均值变化；一个折略退化，区间跨零，不增加该策略复杂度。仅为前64候选流的条件模拟。'),
      'R27':('budget_mean','smoothed_mean_2','预算相关平滑','sqrt(n)强度的参数估计minimax性质不等于PAIEC平均Brier最优；真实结果退化，不扫描新指数。'),
      'R28':('symmetric','smoothed_mean_2','先验方向偏置与形状分离','对称化改善旧E3，但仍输给固定平滑；同方差Beta对照也失败。停止泛用的源prevalence形状拟合。'),
      'R29':('learned','semantic','任务斜率二阶协方差','两折内层benchmark验证，三外折均选rho=0；独立调参的打乱标签控制也全关。没有采用新协方差模块，故不把它当新最佳模型。')}
    for rid,(key,base,title,desc) in descriptions.items():
        d=read(rid);a=d['pair_equal_alc'];c=d['comparisons'][base]
        lines += [f'### {rid}：{title}','',f"ALC {a[key]:.6f}；对照 {a[base]:.6f}；Δ {c['pair_equal_delta']:.6f}；fold Δ `{c['fold_deltas']}`；聚类95%CI `{c['cluster_bootstrap_ci95']}`。",'',desc,'']
    if (root/'R31/summary.json').exists():
        d=read('R31');a=d['pair_equal_alc'];c=d['comparisons']['smoothed_mean_2']
        lines+=['## R30–R31：评估覆盖改变了收益幅度，未改变晋级结论','',
          'R30先审计到36个外层benchmark共有5242合格pair，原100pairs/BM上限只覆盖737个，约14.1%。基于完整计数的重加权外推不是实际全量评估。随后R31冻结模型、实际扩展四个受上限影响的benchmark，其余32个此前已覆盖全部pair，直接复用；三种子均完成。','',
          '| 方法 | 全部合格pair ALC |','|---|---:|']
        for k,v in a.items():lines.append(f'| {k} | {v:.6f} |')
        lines+=['',f"R25相对固定平滑Δ={c['pair_equal_delta']:.6f}；fold Δ `{c['fold_deltas']}`；benchmark聚类95%CI `{c['cluster_bootstrap_ci95']}`。仍小于0.005；不能只与较弱的训练选择平滑比较后宣布通过。",'',
          'alignment_faking占4278/5242≈81.6%的pair。benchmark等权R25为0.212920，平滑0.214869。不同聚合方式均有改善，但该公开集合组成不代表隐藏分布。R25相对R7在完整pair权重下仅改善约0.0000545，metadata收益没有扩大。','',
          '仍保留每pair最多64个evaluation items及重复response权重；因此是完整合格pair覆盖，不是所有evaluation responses的全量评分。旧100-pair子集逐预算回放误差≤1e-7。新增离线语义向量仅4个；R31实际耗时约68秒；逐pair分数CSV、运行日志与哈希均已保存。','',
          '![真实实验：原代理与完整合格pair覆盖](../runs/research_loop/v2_coverage_evidence.png)','']
    lines+=['## 复现与下一步边界','',
      '所有协议、权重/先验表、分数、代码快照、失败恢复记录均保存在 research_loop/ 与 runs/research_loop/。R22保存三份实际训练权重；R29保存三折真实/打乱协方差及内层选择记录，最终选择均为关闭。',
      '', '从paiec_baseline执行：','', '```powershell',"$env:OPENBLAS_NUM_THREADS='1'","$env:OMP_NUM_THREADS='4'",'python -m mlp.run.run_budget_mean_round','python -m mlp.run.run_symmetric_prior_round','python -m mlp.run.run_covariance_round','python -m mlp.state.checkpoint_v2','```','',
      '已有语义缓存是R29复现依赖；不存在时明确报错，不临时联网替换表示。R31使用隔离离线编码环境，详见reports/research_loop_v2_reproduction.md。完整高层更新见research_loop/HIGH_LEVEL_STATE.md。当前研究门槛未通过；独立泛化与语义候选提交验收也未完成，不能宣称研究目标已全部达成。','',
      '最终完整tools回归实际通过63项（7.34秒），日志runs/research_loop/v2_final_regression_tests.log。R30/R31之后新增的覆盖审计另外通过了真实数据的计数与历史子集回放断言，不冒充额外单元测试。','',
      '## 历史实验与推理记录','']
    return lines+history[2:]
