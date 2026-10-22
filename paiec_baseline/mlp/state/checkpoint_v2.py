"""Refresh v2 checkpoint from saved scores; never invent an unrun result."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('runs/research_loop')
STATE=Path('../research_loop')


def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))


def main():
    refs=['loop_1.md','Research Down–Up Loop.md','High-level 认识维度与相互关系.md',
          'Model-Nature_High-Level_Recognition_Paper_List_v1_1.md','常见的组合思路.md']
    (STATE/'V2_SOURCE_MANIFEST.json').write_text(json.dumps({p:{'path':str((Path('../doc')/p).resolve()),
        'sha256':hashlib.sha256((Path('../doc')/p).read_bytes()).hexdigest()} for p in refs},ensure_ascii=False,indent=2),encoding='utf-8')
    registry=json.loads((STATE/'EXPERIMENT_REGISTRY.json').read_text())
    registry['authority']='../doc/loop_1.md'
    updates=[('R7','completed_small_gain_gate_unmet','semantic'),('R8','completed_no_promotion',None),
             ('R9','completed_conditional_evidence_only',None),('R10','completed_mechanism_diagnosis',None)]
    for rid in ['R11','R12','R13','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24','R25','R26','R27','R28','R29','R30','R31']:
        if (ROOT/rid).exists():
            complete=(ROOT/rid/'summary.json').exists() or (ROOT/rid/'provenance.json').exists()
            updates.append((rid,'completed_diagnostic_no_promotion' if complete else 'running',None))
    for name,status,key in updates:
        item=next((r for r in registry['rounds'] if r['id']==name),None)
        if item is None:item={'id':name};registry['rounds'].append(item)
        item.update(status=status,protocol=f'../research_loop/{name}_protocol.md',outputs=f'runs/research_loop/{name}')
        if key:item['alc']=read(name+'/summary.json')['pair_equal_alc'][key]
    (STATE/'EXPERIMENT_REGISTRY.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    r7=read('R7_verify/summary.json');r10=read('R10/semantic/summary.json')
    lines=['# PAIEC v2：认识更新与已执行实验','',
      '依据：`doc/loop_1.md` 及四份参考材料。日期：2026-10-22。所有分数来自真实公开数据；没有正式提交。',
      '', '## 当前判断','',
      '研究对象从“增强 MLP”细化为：区分可迁移的先验均值、当前标签支持的 item 难度关系，以及其他 subject 标签能提供的群体信息。R7 与 R8 不是严格的学习方式因果对照；R10 才隔离了局部标签和 item 的对应关系。',
      '', '## 统一 seed0 对照','',
      '三折、737 pairs、36 benchmarks；每 benchmark 上限100 pairs，每 pair 上限64 evaluation items。沿用已反复查看的公开外层折，绝非新盲测。训练与当前外层 benchmark 保持隔离。',
      '', '| 方法 | B0 | B1 | B3 | B7 | B15 | B31 | ALC |','|---|---:|---:|---:|---:|---:|---:|---:|']
    r3=read('R3/summary.json')
    entries=[('经验均值',r3,'empirical_mean'),('平滑均值2',r3,'smoothed_mean_2'),
             ('训练内选择平滑均值',r3,'training_selected_smoothed'),('原MLP residual2',r3,'original:residual_2')]
    for name,key in [('R1','block:residual_2'),('R1','linear:residual_2'),('R2','structured:residual_2'),
                     ('R3','local_ridge'),('R4','hidden'),('R5','episodic_metric'),('R6','covariance'),
                     ('R7','semantic'),('R8','semantic:residual_2'),('R8','retained:residual_2')]:
        data=read(name+'/summary.json')
        if key in data and isinstance(data[key],dict):data=data[key]
        if key not in data.get('pair_equal_alc',{}):
            if name=='R6':key=next(iter(data['pair_equal_alc']))
            else:raise KeyError((name,key))
        entries.append((name+' '+key,data,key))
    for title,d,k in entries:
        lines.append('| '+title+' | '+' | '.join(f'{v:.6f}' for v in [*d['pair_equal_brier'][k],d['pair_equal_alc'][k]])+' |')
    c=r7['comparisons']['smoothed_mean_2']
    lines+=['','## R7 三种子复核','',
      f"语义局部方法 ALC **{r7['pair_equal_alc']['semantic']:.6f}**；平滑均值 **{r7['pair_equal_alc']['smoothed_mean_2']:.6f}**；Δ={c['pair_equal_delta']:.6f}；三折差值 `{c['fold_deltas']}`；benchmark 聚类95%CI `{c['cluster_bootstrap_ci95']}`。",'',
      '固定训练内参数后复核 seeds0/1/2。先按 benchmark 平均 seed，再进行聚类 bootstrap；CI没有修正整个多轮研究的适应性选择。',
      '', '## R10 / V2-01：从分数上钻到信息使用机制','',
      '事前协议：`research_loop/R10_protocol.md`。算子顺序：审计分数含义 → 精确误差分解 → 对应关系破坏对照 → 更新研究对象。',
      '', '| 诊断量 | 语义表示 | 同窗口文本哈希 |','|---|---:|---:|']
    t=read('R10/text_hash/summary.json')
    for k in ['actual','baseline','mean_only','query_permutation_expectation','support_permuted']:
        lines.append(f"| {k} | {r10['pair_equal_alc'][k]:.6f} | {t['pair_equal_alc'][k]:.6f} |")
    for k in ['mean_gain','item_gain']:
        lines.append(f"| {k}（正值改善） | {r10['components'][k]:.6f} | {t['components'][k]:.6f} |")
    lines+=['', '精确恒等式：`B(base)-B(p) = [(base-q)^2-(mean(p)-q)^2] + [2cov(p,y)-var(p)]`，q为评价响应均值。第一项描述pair均值修正，第二项描述中心化item区分。', '',
      f"语义方法item项占总净收益约 {100*r10['components']['item_gain']/(r10['components']['mean_gain']+r10['components']['item_gain']):.1f}%。与mean-only相比三折均改善；打乱当前预算内support标签与item的对应后，ALC变为{r10['pair_equal_alc']['support_permuted']:.6f}。这支持局部item信息利用，不能升级为对语义因果机制或隐藏benchmark普遍性的证明。",'',
      'mean-only与query permutation是事后数学诊断，不能作为合规单输入预测器提交。重复response保留评分权重；16次置换不作为16个独立测试集。',
      '', '## 共享证据：独立条件场景 R9','',
      '| 每个同benchmark cohort的pair数 | shared prior | 同证据平滑均值2 | ΔALC |','|---:|---:|---:|---:|']
    for size in [1,2,4,6]:
        d=read(f'R9/cohort{size}/summary.json');a=d['pair_equal_alc'];lines.append(f"| {size} | {a['shared_prior']:.6f} | {a['smoothed_mean_2']:.6f} | {a['shared_prior']-a['smoothed_mean_2']:.6f} |")
    lines+=['','官网允许labeled跨采样pair；真实hosted共享分布未验证。R9是假设的同benchmark cohort场景，不能替换隔离pair主门槛；其收益集中在fold0，fold1退化。可见subject配置相同的不同源ID按相同输入匹配，因而同证据baseline也可能变化。',
      '', '## 文献、参考材料与证据边界','',
      '四份用户资料用于控制循环、认识关系、论文入口和算子选择；没有将其中所有论文摘要当成已验证事实。',
      '- [Context vs weights](https://arxiv.org/abs/2210.05675)：已读方法与结果，用于提醒信息源及训练条件需要受控；其Transformer结论仅作类比。',
      '- [Asymmetrically relaxed domain alignment](https://arxiv.org/abs/1903.01689)：已读理论条件，强调分布和标签关系假设；没有据此宣称PAIEC不可辨识。',
      '- [Stan partial pooling](https://mc-stan.org/learn-stan/case-studies/pool-binary-trials.html)：已读二项层次模型，提示小样本群体异质性的估计边界；当前仅是后续条件机制来源。',
      '', '## Gate 与工程','',
      '冻结门槛仍为对两类强简单基线绝对ALC改善0.005、三折方向一致、benchmark等权改善、聚类CI上界<0且无未解释严重退化。R7只达到小幅改善，研究成功门槛未达到。独立泛化证据不足，语义候选尚未完成正式打包验收。fallback仍为 `dist/smoothed_mean_strength2.zip`。',
      '', '实际入口（从paiec_baseline运行）：', '', '```powershell',
      "$env:OPENBLAS_NUM_THREADS='1'", 'python -m mlp.audit.diagnose_local_signal', 'python -m mlp.state.checkpoint_v2', '```', '',
      'R10复用D盘R7缓存，缺失时明确报错，不临时重新选择模型。`runs/research_loop/R10/`含逐pair/seed/预算CSV、各折JSON、bootstrap、代码快照和哈希。状态及后续选择见 `research_loop/HIGH_LEVEL_STATE.md`、`OPERATOR_LEDGER.md` 和 `RESUME.md`。']
    if (ROOT/'R12/summary.json').exists():
        d=read('R12/summary.json')
        lines+=['','## R11–R12：分层解释未通过更强控制','',
                '固定参数、cohort4、三个轨迹种子的同证据比较：','',
                '| 方法 | ALC |','|---|---:|']
        for k,v in d['pair_equal_alc'].items():lines.append(f'| {k} | {v:.6f} |')
        lines+=['','训练内选择的peer prior强度为4/4/4。匹配强度后，固定分层模型不优于简单plug-in，差值CI跨零；自适应异质性相对固定模型也未显示可信额外收益。于是将复杂度降回简单peer估计，不能把R11的收益归因于Bayesian integration。',
                '', '来源：`runs/research_loop/R11_verify/`、`R12/`。R11的四种共享规模和数值/标签范围测试均已执行。']
    if (ROOT/'R13/cohort6/summary.json').exists():
        lines+=['','## R13：两项信息的四组对照','',
                '| cohort | base | local | peer | combined | interaction |','|---:|---:|---:|---:|---:|---:|']
        for size in [1,4,6]:
            d=read(f'R13/cohort{size}/summary.json');a=d['pair_equal_alc']
            lines.append(f"| {size} | {a['base']:.6f} | {a['local']:.6f} | {a['peer']:.6f} | {a['combined']:.6f} | {d['interaction_alc']:.6f} |")
        lines+=['','这是当前标签中不同信息来源的组合检验；共享规模不能事后择优代替预注册主场景。cohort1验证隔离pair行为，cohort4对照更强peer基线。']
    if (ROOT/'R14/summary.json').exists():
        d=read('R14/summary.json');a=d['pair_equal_alc']
        lines+=['','## R14：共享组成稳健性','',
          f"三种cohort分组×三种item划分，组合ALC{a['combined']:.6f}，简单peer{a['peer']:.6f}，base{a['base']:.6f}。相对base下降{a['base']-a['combined']:.6f}，相对强简单peer只下降{a['peer']-a['combined']:.6f}，故未通过晋级门槛。不能用条件场景替代原隔离pair门槛。",'',
          '![真实实验机制分解与共享组成敏感性](../runs/research_loop/v2_mechanism_evidence.png)']
    if (ROOT/'R16/summary.json').exists():
        d=read('R16/summary.json');a=d['pair_equal_alc']
        lines+=['','## R15–R16：否定固定能力轴，并审计标签含义','',
          '完整subject配置跨benchmark匹配为零；放宽到名称/提供商/reasoning后，实际可用的多benchmark能力支持只覆盖188/737目标pair。16个可比较benchmark仅6个协方差为正。',
          f"能力修正ALC{a['original_ability']:.6f}，仅higher方向训练修正{a['filtered_ability']:.6f}，均未超过简单peer{a['peer']:.6f}。停止扩大该标量能力表。",'',
          '61个公开二元benchmark含4个lower、3个unordered方向，1不统一代表回答正确。历史分数按原始outcome计算；没有翻转标签或改变外层测试。详见 [标签语义审计](label_semantics_audit_2026-10-22.md)。语义过滤也不能充分解释旧MLP失效：higher子集B0仍为0.274379。']
    if (ROOT/'R17/summary.json').exists():
        d=read('R17/summary.json');a=d['pair_equal_alc']
        lines+=['','## R17：现有预测的收缩修复边界','',
          f"允许读取评价标签的pair级oracle，在R7与smoothed2的凸组合族中最优ALC为{a['oracle_pair']:.6f}，相对base{a['base']:.6f}的最大改善为{a['base']-a['oracle_pair']:.6f}。仍低于0.005，因此这类系数衰减无法使冻结的R7预测达标。",'',
          '这是明确使用评价标签的oracle诊断，不是可提交方法，不用于选择参数，也不是所有预测器的信息论下界。它约束的是当前预测与base之间的凸组合；不排除新表示、新信息或新的预测方向。']
    if (ROOT/'R18/summary.json').exists():
        d=read('R18/summary.json');a=d['pair_equal_alc']
        lines+=['','## R18：输入截断修复没有改善当前方法','',
            f"3600个抽样item中1783超过126个内容token，但同一冻结编码器扩展至256token后，语义ALC{a['semantic256']:.6f}，原128窗口{a['semantic128']:.6f}；同窗口词法对照也从{a['text128']:.6f}变为{a['text256']:.6f}。两种表示均未改善，停止这一窗口扩展路线。",'',
            '新旧128实现前向最大差为0；实际模型输入未超过256。审计发现信息被省略，不能据此断言被省略内容一定对当前标签有用。没有扩大模型或额外下载数据。']
    if (ROOT/'R19_verify/summary.json').exists():
        d=read('R19_verify/summary.json');a=d['pair_equal_alc']
        lines+=['','## R19：当前其他subject的同item标签含有小幅额外信号','',
            f"条件cohort4、三种子：同item残差迁移ALC{a['matched']:.6f}，共享均值{a['peer']:.6f}，打乱item对应关系{a['permuted']:.6f}，R13语义组合{a['combined']:.6f}。相对共享均值仅改善{a['peer']-a['matched']:.6f}，不达到0.005。",'',
            '匹配仅使用可见item内容、features与interactors；源item ID只用于重建固定划分。预算31的匹配覆盖约12.9%，低预算更低。没有读取当前目标的评价标签。不同subject的池是否在hosted端对齐仍未核实，因此这里只能支持条件共享场景结论。']
    if (ROOT/'R20/summary.json').exists():
        d=read('R20/summary.json');a=d['pair_equal_alc']
        lines+=['','## R20：扩展当前局部斜率的证据来源','',
            '| 方法 | ALC |','|---|---:|']
        for k,v in a.items():lines.append(f'| {k} | {v:.6f} |')
        lines+=['','先在每个可见subject内中心化特征和标签，再用当前cohort共同拟合item斜率；共享均值与编码器固定。独立选择的own-only对照避免将正则强度变化归因于证据共享。具体判断见最新研究状态，条件场景边界保持不变。']
    if (ROOT/'R20_verify/summary.json').exists():
        d=read('R20_verify/summary.json');a=d['pair_equal_alc']
        lines+=['','### R20 三种子复核：相同 cohort4 / seed193','',
            '| 方法 | 三种子 ALC |','|---|---:|']
        for k,v in a.items():lines.append(f'| {k} | {v:.6f} |')
        c=d['comparisons']['peer']
        lines+=['',f"共同item斜率相对peer改善{a['peer']-a['pooled']:.6f}，三折差值`{c['fold_deltas']}`，benchmark聚类95%CI`{c['cluster_bootstrap_ci95']}`。这项收益仍小于0.005。相对smoothed2超过0.005不能替代强简单peer对照，也不能替代原独立pair门槛。",'',
                '![相同条件下当前item证据的收益](../runs/research_loop/v2_current_evidence.png)']
    if (ROOT/'R21/summary.json').exists():
        d=read('R21/summary.json');a=d['pair_equal_alc'];c=d['comparisons']['pooled1']
        lines+=['','## R21：固定非线性核的额外收益不稳定','',
            f"seed0：四次核{a['pooled4']:.6f}，线性核{a['pooled1']:.6f}；差值CI`{c['cluster_bootstrap_ci95']}`跨零，fold0退化。未满足预注册的全折改善条件，停止核阶数扩展，也未把它升级为最佳已验证版本。",'',
            '核公式来自[多项式核与张量积的一手论文](https://arxiv.org/html/2202.02031)，只用于数学构造；没有借用该论文的应用性能结论。二次实现验证包括power1与原线性解一致、power4与显式张量特征一致。']
    if (ROOT/'R22/summary.json').exists():
        d=read('R22/summary.json');a=d['pair_equal_alc']
        lines+=['','## R22：扩大训练池仍未修复跨benchmark先验','',
            f"每benchmark训练池上限400→2000，验证样本保持400，每验证周期训练暴露量及更新步数固定为旧模型。扩大后prior Brier{a['expanded:prior']:.6f}，residual ALC{a['expanded:residual_2']:.6f}；旧semantic residual{a['semantic:residual_2']:.6f}，smoothed2 {a['smoothed_mean_2']:.6f}。两折改善、一折退化；不继续扩大当前训练池。",'',
            '| 折 | 可训练响应池 | 实际执行中见过的不同响应 | pair数 | 最佳验证周期 |','|---:|---:|---:|---:|---:|']
        for f in range(3):
            m=read(f'R22/fold{f}/weights.json');lines.append(f"| {f} | {m['train_rows']} | {m['unique_training_rows_exposed']} | {m['train_pairs']} | {m['best_epoch']} |")
        lines+=['','暴露计数涵盖早停检查后的全部已执行周期，不等于最佳权重见过的精确样本数。此次扩大了样本池与pair覆盖，并固定每周期计算量；没有证明所有更大数据或更长训练都无效。三组实际权重、训练日志、独立进程评估及前向导出误差均保存在R22。']
    if (ROOT/'R23/summary.json').exists():
        d=read('R23/summary.json');a=d['pair_equal_alc']
        lines+=['','## R23：低预算软subject约束没有稳定超过现有方法','',
            f"联合Gaussian条件均值{a['joint']:.6f}，相同Gaussian模型不含item时{a['gaussian_mean']:.6f}，原硬中心化{a['hard_centered']:.6f}。相对硬中心化的两折退化、差值区间跨零，停止该路线。预算1有局部改善，不能据此宣称整体可靠收益。",'',
            '这是Bernoulli结果的Gaussian近似并经过裁剪，不是精确二元后验。所有条件化仅使用当前预算标签；不混入任何评价结果。']
    if (ROOT/'R24/summary.json').exists():
        a=read('R24/summary.json')['pair_equal_alc'];c=read('R25/attribution_summary.json')['components']
        lines+=['','## R24–R25：metadata信号与统一正则的混淆','',
            '审计3600items：原metadata 2900非空；去除ID、来源路径等字段后，21个预声明通用字段在1900items可用，18个benchmark组内有变化。原格式是分号分隔key=value，已增加相应解析。', '',
            f"统一正则组合ALC{a['combined']:.6f}，语义{a['semantic']:.6f}。归因分解：新增字段分量{c['added_field_delta']:.6f}，正则变化分量{c['regularization_delta']:.6f}。新增字段收益本身仍异质，不能说metadata普遍有效。"]
    if (ROOT/'R25/summary.json').exists():
        a=read('R25/summary.json')['pair_equal_alc']
        lines+=['',f"固定原语义正则、单独选择metadata正则后的seed0 ALC{a['separate']:.6f}，原语义{a['semantic']:.6f}；一折仍略退化。空/恒定metadata可精确回退原预测。"]
    if (ROOT/'R25_verify/summary.json').exists():
        d=read('R25_verify/summary.json');a=d['pair_equal_alc']
        lines+=['','### R25 固定参数三种子验证','',
            '| 方法 | ALC |','|---|---:|']
        for k,v in a.items():lines.append(f'| {k} | {v:.6f} |')
        lines+=['',f"相对原语义：`{d['comparisons']['semantic']}`。仍按冻结0.005及全折等门槛判断；不按点估计重新定义成功。"]
    lines+=['','## 当前工程检查','',
            '新增可选训练控制参数后，完整tools回归实际通过59项（日志runs/research_loop/v2_regression_tests.log）。此数对应当次运行，不包含以后未运行的测试。fallback ZIP SHA256仍为E55EE0B267F3B345685EF072CFA511C3C017D9082256573793961EA20FEADCB9。没有正式提交，也没有将未过门槛的新模型替换fallback。']
    from mlp.state.checkpoint_v2_latest import prepend_latest
    lines=prepend_latest(lines)
    Path('../reports/research_loop_v2_2026-10-22.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Updated v2 manifest, registry and report')


if __name__=='__main__':main()
