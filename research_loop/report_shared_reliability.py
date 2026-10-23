"""Generate plan_9 tables from frozen results; interpretations reviewed by root."""
from shared_reliability_common import *

def num(v,d=9):return '不可估' if v is None else f'{v:.{d}f}'
def ci(v):return '['+', '.join(num(x) for x in v)+']'
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',*['|'+'|'.join(map(str,r))+'|' for r in rows]])
def save(name,text):(REPO/'reports'/name).write_text(text.strip()+'\n',encoding='utf-8')

def main():
    root=ROOT/'R77';summary=read(root/'full/summary.json');comp=read(root/'comparisons.json');decision=read(root/'decision.json');avg=comp['cohort_average']
    metrics=summary['results'];bench=summary['by_benchmark'];matrix=pd.read_csv(root/'benchmark_cohort_delta.csv');geometry=pd.read_csv(root/'prediction_geometry.csv')
    ctable=table(['cohort','pooled ALC','partial8 ALC','Δpartial−pool','BM-equal Δ','95% BM-cluster CI','三个fold Δ'],[
        [c,num(metrics[f'pool_{c}']['alc_mean']),num(metrics[f'partial_{c}']['alc_mean']),num(comp['cohorts'][str(c)]['delta']),
         num(comp['cohorts'][str(c)]['benchmark_equal_delta']),ci(comp['cohorts'][str(c)]['cluster_ci95']),'/'.join(num(v) for v in comp['cohorts'][str(c)]['fold_delta'])] for c in COHORTS])
    stable=decision['status']=='partial_sharing_composition_robust_in_frozen_scenarios'
    g=geometry.groupby('cohort').mean(numeric_only=True)
    shrinkrows=[[c,num(metrics[f'shrink_{c}']['alc_mean']),num(comp['cohorts'][str(c)]['partial_vs_shrink']['delta']),ci(comp['cohorts'][str(c)]['partial_vs_shrink']['cluster_ci95']),
                 num(g.loc[c,'partial_displacement_cost']/g.loc[c,'pool_displacement_cost'],6)] for c in COHORTS]
    scenrows=[[s,num(metrics[f'pool_{s}']['alc_mean']),num(metrics[f'partial_{s}']['alc_mean']),num(v['delta']),ci(v['cluster_ci95'])] for s,v in comp['scenarios'].items()]
    budgetdelta=table(['cohort/场景','B3 ΔBrier','B7 ΔBrier','B15 ΔBrier','B31 ΔBrier'],
        [[label,*[num(z) for z in v['per_budget_delta'][2:]]] for label,v in [*comp['cohorts'].items(),*comp['scenarios'].items()]])
    daverows=[[c,num(bench['dave/formatted_tables']['alc_mean'][f'pool_{c}']-bench['dave/formatted_tables']['alc_mean']['R25']),
                  num(bench['dave/formatted_tables']['alc_mean'][f'partial_{c}']-bench['dave/formatted_tables']['alc_mean']['R25']),
                  num(comp['cohorts'][str(c)]['by_benchmark']['dave/formatted_tables'])] for c in COHORTS]
    hetero=read(root/'heterogeneity_association.json')
    htable=table(['cohort','诊断BM数','H与同四人组B31 Δ的Spearman'],[[c,v['benchmark_rows'],num(v['spearman_H_vs_query_increment'],6)] for c,v in hetero['by_cohort'].items()])
    text=f'''# R77：Partial8 跨 peer 组成稳健性与机制解释

2026-10-23。原始计划[plan_9](../doc/plan_9.md)，[执行合同](../research_loop/SHARED_RELIABILITY_SPEC.md)，[复现入口](shared_reliability_reproduction.md)。状态：实际完成全部预设组成与压力场景；`conditional_simulation`。没有选择最优cohort作为部署组合。

## 1. 主要结论

五个cohort中 **{decision['cohorts_improved']}/5** 的partial−pooled增量为负。先在每pair/seed内平均五个cohort风险，再以BM为cluster计算，平均ΔALC **{num(avg['delta'])}**，95% CI **{ci(avg['cluster_ci95'])}**；BM-equal Δ **{num(avg['benchmark_equal_delta'])}**。预设判定：`{decision['status']}`。

这{'加强' if stable else '限制'}了部分共享的组成稳健性证据，仍不表示任意peer、未见Benchmark或真实Hosted都可靠。五次cohort重测没有变成五倍独立样本，不能选其中最低分当新的无偏成绩。

## 2. 同一组成内的结构增量

{ctable}

7501为R76主cohort，7502为已冻结第二组成，7701/7702/7703为评分前冻结的新组成。7502的partial补评先于扩展场景执行；其pooled概率与原R76_alt3复用并核对。新三个cohort同时计算pooled/partial。所有方法使用同query、own/peer acquisition、原response counts、相同特征与罚项8。

## 3. 缺失、数量不足与不均衡

以下仅主7501，不代表第二或其他cohort也做了完整压力曲线。

{table(['场景','pooled ALC','partial ALC','配对Δ','95% CI'],scenrows)}

0peer和强制歧义为精确R25，不额外估计subject偏移。缺失和uneven同时改变标签数量、组成及估计权重，不能单凭差值分离每种原因。逐预算风险、有效peer与实际标签数见全表和原label_costs.csv；共享不是总预算等于own预算的公平比较。

{budgetdelta}

表中均为partial减同信息pooled，B0/B1恒为0。1peer的平均改善幅度约.000219，高于完整3peer的约.000181；uneven仍改善约.000207。因而partial的边际收益并非随peer数量或总标签数单调增加，也未在少peer场景消失。这与联合估计幅度/方差变化相容，但压力场景未独立控制观测数和组成，不能唯一归因于Subject特异方向。

## 4. 幅度收缩还是Subject特异方向？

每个outer fold的简单幅度alpha只由合法R75来源预测几何拟合并固定：投影partial修正到pooled修正，限制在[0,1]。不看outer outcome；alpha随预算变化的既定值见R77/source_shrinkage.json。该对照接近partial平均幅度，但不保证逐请求或outer上的平方成本完全匹配。

{table(['cohort','来源幅度对照ALC','partial−幅度对照','95% CI','partial/pool位移平方成本比'],shrinkrows)}

跨cohort平均partial−简单幅度对照 **{num(avg['partial_vs_shrink']['delta'])}**，CI **{ci(avg['partial_vs_shrink']['cluster_ci95'])}**。本次CI跨零，BM-equal差值为 **{num(avg['partial_vs_shrink']['benchmark_equal_delta'])}**，去alignment后差值为 **{num(avg['partial_vs_shrink']['without_alignment_delta'])}**。因此不能用更好的总体点估计排除幅度正则解释，更不能断言已经准确识别Subject特异方向；优势仍主要集中于alignment。精确误差项/位移成本在prediction_geometry.csv；平方位移成本不是中心化预测方差。

辅助H3只取每BM×cohort可见配置排序首个完整四人组，seed0/B31，用R75两半重复性及32次fitted null。该附加诊断在第二cohort补评后、扩展结果完成前登记，不是事前主要确认性检验；不参与选择或门控。

{htable}

先按BM平均cohort后的H–query增量Spearman为 **{num(hetero['benchmark_average']['spearman_H_vs_query_increment'],6)}**。负相关才与“异质性越大，partial越有益”方向一致；小样本fitted H受噪声和模型假设影响，即使有关联也不证明真实能力结构已可辨识。该B31统计不作为B3时可见输入。

## 5. 集中性、翻转与失败边界

cohort平均增量中alignment贡献 **{num(avg['alignment_contribution'])}**，其余BM贡献 **{num(avg['others_contribution'])}**；去alignment再归一Δ **{num(avg['without_alignment_delta'])}**。同BM在五cohort中的partial−pooled方向翻转有 **{int(matrix.sign_flip.sum())}/{len(matrix)}** 个；完整矩阵见R77/benchmark_cohort_delta.csv和图。

{table(['cohort','dave pooled−R25','dave partial−R25','dave partial−pool'],daverows)}

相对pooled未新增>=.005严重回归与相对R25没有严重回归是两个不同命题。partial在五套组成中均略微缓和dave，但没有消除它的退化；三个组成仍超过+.005。不能据dave名称反向写门控；完整Research Gate和Hosted条件没有因稳健性研究自动通过。出现partial−pool方向翻转的BM为bertaqa、cmmlu、exploitgym，幅度较小，详见矩阵。

## 6. 指标、原文、执行与认识

全部5242pairs×3item seeds×6预算，36BM；pair内按response计数，pair/seed等权，ALC权重.1/.2/.2/.2/.2/.1，64query item cap。主要新配对CI采用10000次BM cluster、seed20261023。准确率是p>=.5预测response成功，不是subject原正确率；详见[完整表](shared_reliability_tables.md)。

部分共享结构依据KDD2004原文及MALSAR作者代码，分组选择依据DomainBed原文/代码；[精确版本映射](../research_loop/literature/shared_reliability/source_evidence.md)。本轮复用原R76真实中心化、核、clip，不宣称复刻文献原实验数字。

研究认识：公共的局部关系可以存在，而对每个Subject的使用收益仍取决于组成、样本量和估计误差。R77直接回答结构增量是否跨本次组成保持；能否在调用时识别有害共享，需要R78的合法支持预测诊断，不能由平均ALC代替。

导出图：[cohort增量](shared_reliability_figures/cohort_increment.png)、[BM×cohort矩阵](shared_reliability_figures/benchmark_cohort_matrix.png)。原概率与独立replay分别在R77/alternative及R77/full，资源见execution.json。下一步按R78来源条件决定，不继续扫描partial强度。
'''
    save('shared_cohort_robustness_findings.md',text)
    tables=['# plan_9 数值、准确率与诊断表','', 'R77是完整公开条件模拟；R78是来源诊断，二者不横排。Accuracy为p>=.5的response分类准确率。主要评分是Brier ALC。','', '## R77 五cohort配对结果','',ctable,'','## R77 压力场景','',table(['场景','pooled ALC','partial ALC','Δ','CI'],scenrows),'','## R77 各预算配对增量','',budgetdelta,'','## R77 全方法六预算','']
    rows=[]
    for m,v in metrics.items():
        for j,n in enumerate(BUDGETS):rows.append([m,n,num(v['brier_mean'][j]),f"{v['accuracy_at_0_5'][j]*100:.6f}%"])
    tables.append(table(['方法','预算','Brier','准确率'],rows))
    for c,v in comp['cohorts'].items():
        tables+=['',f'## cohort {c} alignment与其余','',table(['比较','ΔALC','alignment贡献','其余贡献','去alignment重归一Δ','最坏BM','最坏Δ'],
             [[name,num(x['delta']),num(x['alignment_contribution']),num(x['others_contribution']),num(x['without_alignment_delta']),x['worst_benchmark'],num(x['worst_delta'])] for name,x in [('partial−pool',v),('partial−R25',v['partial_vs_own'])]])]
    if (ROOT/'R78/decision.json').exists():
        r78=read(ROOT/'R78/decision.json');assoc=read(ROOT/'R78/associations.json');validation=read(ROOT/'R78/replay_validation.json');a=assoc['deduplicated']['main']
        atable=table(['指标','BM平均Spearman','95% CI','方向balanced acc','可靠−不可靠query风险差','BM内组差','mask符号一致率'],
          [[m,num(v['benchmark_equal_spearman'],6),ci(v['spearman_cluster_ci95']),num(v['direction_balanced_accuracy'],6),num(v['pair_group_gap']),num(v['within_benchmark_group_gap']),num(v['mask_repeat_sign_agreement'],6)] for m,v in a.items()])
        st=table(['变量','未通过的预设项'],[[v['metric'],', '.join(k for k,p in v['tests'].items() if not p) or '全部通过'] for v in r78['tests']])
        budgets=pd.read_csv(ROOT/'R78/budget_metrics.csv');slices=pd.read_csv(ROOT/'R78/mechanism_slices.csv')
        scenetable=table(['场景','partial BM rho','95% CI','方向BA','可靠−不可靠query差'],
            [[s,num(v['partial']['benchmark_equal_spearman'],6),ci(v['partial']['spearman_cluster_ci95']),num(v['partial']['direction_balanced_accuracy'],6),num(v['partial']['pair_group_gap'])] for s,v in assoc['deduplicated'].items()])
        slicetable=table(['预算','切片','支持partial−own','query partial−own','z>=0请求比例'],
            [[r.budget,r.slice,num(r.support_partial_delta),num(r.query_partial_delta),f'{r.unreliable_request_fraction:.2%}'] for r in slices[slices.scenario=='main'].itertuples()])
        report=f'''# R78：支持标签上的共享可靠性是否预示query收益？

2026-10-23。实际来源诊断完成；状态 **`{r78['status']}`**。本报告与[R77](shared_cohort_robustness_findings.md)分别回答可共享结构和可观测可靠性。

## 1. 主要结果与停止决定

{atable}

{st}

R79诊断入口：**{r78['R79_diagnostic_entry']}**。{'通过后仍须嵌套来源策略比较，不能直接进入完整外层。' if r78['R79_diagnostic_entry'] else '没有指标满足全部事前进入条件；按合同停止门控开发，不创建R79实验目录，不训练更复杂gate，也不降低阈值。'}

这里“可靠”固定指z<0；query风险差目标为full-support partial−own。rho正值表示较坏support相对风险伴随较坏query风险，负值则反向。方向识别排除query差为零的中性pair；balanced accuracy为有利/有害两类召回均值，不能与PAIEC response准确率混淆。

**结果的关键限制：** partial的可靠组query Δ为{num(a['partial']['reliable_query_delta'])}，不可靠组仍为{num(a['partial']['unreliable_query_delta'])}，两组平均都从共享获益。组间排序较好不等于z=0能标定query收益的正负边界。partial组间差的BM-cluster CI为{ci(a['partial']['group_gap_cluster_ci95'])}，跨零；本轮预设入口只要求该组差的点估计方向，不能把通过入口写成所有检验显著。

主场景单独B15/B31的rho分别为{num(assoc['per_budget']['main']['15']['partial']['benchmark_equal_spearman'],6)}/{num(assoc['per_budget']['main']['31']['partial']['benchmark_equal_spearman'],6)}，各自CI均跨零。预设跨预算、seed汇总得到的正面信号，强于单次预算下实际可使用的证据。只有24/35个BM可估rho，来源至多8pairs/BM，不等同完整5242pairs分布。

## 2. 实际协议与隔离

沿用R75每BM至多8个唯一可见配置的来源名单；三来源池有重叠。总诊断记录{validation['diagnostic_records']}，对应{validation['source_trajectories']}次来源轨迹暴露；固定每BM首个合法来源池去重后{validation['unique_reporting_trajectories']}条轨迹。每轨迹比较B15/B31及主7501、第二7502、missing、uneven、peer-label permutation五场景，不是新增独立Benchmark。

每预算做两个固定随机二分重复、每重复互补两mask，共4次fit；每条own支持标签恰好被留出两次。fit-own均值、中心化和斜率都只用未遮挡部分；peer只用当前预算合法前缀。query outcome只在预测完成后评分。改变heldout-own标签不改变fit预测的对抗断言逐实例执行。

保存{validation['support_prediction_rows']}条support留出预测记录、{validation['query_variant_rows']}条query variant记录。独立重计分：query Brier误差{validation['max_query_brier_error']:.3g}、accuracy误差{validation['max_query_accuracy_error']:.3g}、support z误差{validation['max_support_z_error']:.3g}。mask、身份、peer前缀及原概率均已落盘。

## 3. 均值一致与斜率一致

- partial/pooled：各Subject均值各自保留，检验局部斜率预测heldout own标签的相对Brier。
- mean：仅比较peer等权平滑均值与fit-own平滑均值，两者都没有局部方向。
- combined：partial的已裁剪预测移除own截距、加peer均值再clip；固定构造，没有增加拟合参数。它是均值借用控制，不宣称精确恢复未裁剪的潜在斜率。

四类变量均事前固定，没有按dave或外层误差设计阈值；预设z<0与z>=0分组。先平均同pair预算和item seeds，再在每BM内算Spearman。少于3pairs或常数统计的BM标不可估，不填0。CI以可估BM为cluster；它并不代表那些无法估计BM的表现。

主场景mean-only支持风险相对fit-own均值在B15/B31分别恶化{num(float(budgets[(budgets.scenario=='main')&(budgets.budget==15)].z_mean.iloc[0]))}/{num(float(budgets[(budgets.scenario=='main')&(budgets.budget==31)].z_mean.iloc[0]))}；平均成功率借用本身未显示益处。斜率指标方向BA约0.629/0.647，均值约0.527；combined虽rho约0.247，方向BA约0.569。固定控制更支持局部对应关系提供排序信息，但没有对四指标差异作独立显著性检验，不能声称已唯一识别因果机制。

## 4. 反例与机制边界

完整的各预算、missing/uneven/第二组成/置换和每BM统计见[全表](shared_reliability_tables.md)、R78/associations.json与benchmark_associations.csv。置换控制破坏的是当前peer对应关系；若某可靠性变量只能识别大幅破坏，不足以证明它能区分真实共享中较小的有害误差。

{scenetable}

主场景的斜率关联在peer-label permutation后接近零、CI跨零，支持真实item对应关系参与了该信号；这并未消除小样本噪声或证明跨来源泛化。

{slicetable}

dave的z与query风险均偏坏，当前信息能捕捉部分错误共享；alignment之外仍有组间排序信号。但dave只含8个来源pair，不能据该切片推出完整dave已修复。源数据中的风险排序不提供按BM名称关闭共享的授权。

R78比R43/R61/R71新增了“peer局部方向对暂时隐藏own标签的预测”这一可观测量，直接接触共享机制。但最多31个标签再做holdout会减少own拟合信息，support风险与full-support query风险并非同一个估计问题。相关弱或跨BM不稳可以由有限样本噪声、fit规模变化或真正结构异质性共同造成；失败不能证明所有可靠性信号不存在。

R77结构增益成立也不推出z能可靠决定每次是否共享。本轮不从事后错误切片制造专门gate，不把均值接近自动解释为斜率一致。{'已进入严格嵌套R79来源策略验证，结果见[门控决策](shared_reliability_policy_findings.md)；未通过其进入条件，停止完整外层。' if (ROOT/'R79/decision.json').exists() else '来源入口和实际策略执行状态以总决策为准。'}

## 5. 来源、资源与复现

原文与作者代码为KDD2004/MALSAR的共享结构以及DomainBed的模型选择纪律；R78变量和mask是本项目预设机制实验，不是假称文献已验证的PAIEC方法。[来源定位](../research_loop/literature/shared_reliability/source_evidence.md)，[执行命令](shared_reliability_reproduction.md)，[严格OOF补充](../research_loop/SHARED_RELIABILITY_OOF_ADDENDUM.md)。

CPU运行墙钟耗时/真实进程峰值见R78/execution.json与analysis_execution.json；无新网络、LLM标注、外部付费API或Hosted工作。公开数据反复使用及来源池重叠使本轮仍是探索性诊断。图：[支持风险与query关联](shared_reliability_figures/support_query_association.png)。
'''
        save('support_sharing_reliability_findings.md',report)
        tables+=['','## R78 主场景可靠性诊断','',atable,'','## R78 各场景、各预算','']
        ar=[]
        for scene,v in assoc['per_budget'].items():
            for n,mm in v.items():
                for m,x in mm.items():ar.append([scene,n,m,num(x['benchmark_equal_spearman'],6),ci(x['spearman_cluster_ci95']),num(x['direction_balanced_accuracy'],6),num(x['pair_group_gap']),x['reliable_pairs'],x['unreliable_pairs']])
        tables.append(table(['场景','预算','指标','BM rho','CI','方向BA','分组query Δ差','可靠pair','不可靠pair'],ar))
        rr=[]
        for r in budgets.itertuples():
            for m in ['own','pooled','partial','meanbase','mean','combined']:
                rr.append([r.scenario,r.budget,m,num(getattr(r,'support_brier_'+m)),num(getattr(r,'query_brier_'+m)),f"{getattr(r,'query_accuracy_'+m)*100:.6f}%",num(r.own_labels,2),num(r.peer_labels,3),num(r.active_peers,3)])
        tables+=['','## R78 支持与query风险、response准确率及标签数','',
            '来源去重191pairs/35BM，pair/seed等权。support是遮挡后fit，query是full-current-support；两个集合风险不作同分布等同。','',
            table(['场景','预算','模型','支持Brier','query Brier','query准确率','own标签','peer标签均数','有效peer均数'],rr),
            '', '## R78 主要反例切片','',slicetable]
    if (ROOT/'R79/decision.json').exists():
        tables+=['','## R79 条件来源验证','',report_policy()]
    resources=[]
    for label,path in [('R77第二组成','R77/alternative/execution.json'),('R77五组成及压力','R77/full/execution.json'),('R77回放分析','R77/analysis_execution.json'),('R77异质性辅助','R77/heterogeneity_association.json'),('R78来源预测','R78/execution.json'),('R78回放分析','R78/analysis_execution.json'),('R79来源策略','R79/execution.json')]:
        if (ROOT/path).exists():
            e=read(ROOT/path);resources.append([label,num(e['seconds'],3),num(e['peak_working_set_mib'],2)])
    cost=pd.read_csv(ROOT/'R77/label_cost_summary.csv')
    tables+=['','## 实测资源','',table(['阶段','CPU运行墙钟秒','进程lifetime峰值MiB'],resources),
        '', '各阶段耗时不包括全部检索、实现、报告时间；不是用户投入时间或CPU core-seconds。R79复用R78预测，未重训网络。','',
        '## R77 标签成本汇总','',table(list(cost.columns),cost.round(6).values.tolist()),
        '', '标签成本对应每次请求可用的当前输入；不是跨场景新标注的唯一标签数。最大own31+peer93=124，不是总预算31。']
    save('shared_reliability_tables.md','\n'.join(tables))
    print('R77/R78 reports and combined tables generated')

def report_policy():
    root=ROOT/'R79';c=read(root/'source_comparisons.json');d=read(root/'decision.json');e=read(root/'source_error_decomposition.json');p=read(root/'nested_source_policies.json')
    rows=[]
    for f,v in c['source_pools'].items():
        for m,x in v.items():rows.append([f,m,num(x['delta_ALC']),num(x['benchmark_equal_delta']),ci(x['cluster_ci95'])])
    ct=table(['来源pool','gate减去对照','ΔALC','BM-equal Δ','95% BM-cluster CI'],rows)
    agg=table(['去重汇总对照','ΔALC','BM-equal Δ','去alignment Δ','CI'],
        [[m,num(x['delta_ALC']),num(x['benchmark_equal_delta']),num(x['without_alignment_delta']),ci(x['cluster_ci95'])] for m,x in c['deduplicated'].items()])
    pt=table(['来源pool','validation半组','fit BM数','validation BM数','fit选择指标','B15匹配mean强度','B31匹配mean强度'],
        [[f,h,len(v['fit_benchmarks']),len(v['validation_benchmarks']),v['metric'] or '无：保持partial',num(v['coefficients']['15']['mean'],6),num(v['coefficients']['31']['mean'],6)] for f,hh in p.items() for h,v in hh.items()])
    budget=pd.read_csv(root/'budget_metrics.csv');bt=table(['预算','模型','query Brier','response准确率'],
        [[r.budget,m,num(getattr(r,'brier_'+m)),f'{getattr(r,"accuracy_"+m)*100:.6f}%'] for r in budget.itertuples() for m in ['own','pooled','partial','gate','matched_mean','matched_cost']])
    report=f'''# R79：二值共享回退的来源验证与停止决定

2026-10-23。R78诊断入口已通过，实际完成R79嵌套来源验证；状态 **`{d['status']}`**。完整5242-pair外层 **condition_not_met，未运行**。不把来源局部Δ写成新的完整ALC，不选择事后阈值或更有利cohort。

## 1. 来源验证结果

{ct}

{agg}

三来源池中{d['source_pools_improved_vs_both']}/3同时优于partial与matched-mean；综合pair、BM及去alignment均未改善。CI跨零，结论是没有满足预设独立收益条件，不是证明所有可靠性方法必然有害。停止本配置；不执行完整外层、不扩阈值网格。

## 2. 真正嵌套的选择与覆盖

每pool按固定hash `7901|Benchmark` 分两半。仅fit半组按partial/pooled/mean/combined既定顺序和R78局部准则选指标；只用fit半组确定B15/B31 matched平均开启率及平方位移对照。validation的query结果不参与这些选择。无fit指标通过时，gate和对照都保持原partial。

{pt}

只有pool0的一条内层lane采用了pooled指标，其他五条未达到小fit样本下的准入条件。去重后实际有可用门控的预算请求占{e['eligible_policy_fraction']:.2%}，所以大量零差值代表原partial回退，不能解释成已经充分检验了每个Benchmark的门控。hash半组不均衡/样本少限制了检验能力，未为获得结果改分组或降低门槛。全pool拟合得到的指标只存作审计，未应用到任何外层query。

## 3. 错误关闭收益与避免损失

- 名义关闭率{e['closure_fraction']:.2%}；确有peer的请求中关闭率{e['closure_fraction_when_peer_present']:.2%}。
- 关闭的有益共享{e['wrong_closed_requests']}次、有害共享{e['harmful_closed_requests']}次、净效果中性{e['neutral_closed_requests']}次；统计单位是pair×seed×预算，不是独立Benchmark。
- 错误关闭损失ALC {num(e['lost_benefit_ALC'])}；避免有害共享挽回{num(e['avoided_harm_ALC'])}；净退化{num(e['net_gate_minus_partial_ALC'])}。用query outcome事后分解，只用于解释，不能成为运行时特征。
- dave来源切片gate−partial为{num(c['deduplicated']['partial']['by_benchmark']['dave/formatted_tables'])}，但algotune、beavertails、cmmlu、financebench的损失抵消了收益。这不是完整dave外层验证，不能宣称严重回归已解决。

## 4. 与简单收缩的公平比较

matched-mean为fit半组相同平均共享开启率，matched-cost为fit半组相同修正平方成本；二者都不是validation/outer事后最优常数，在validation上成本不必完全一致。gate对这两个对照没有独立增量。所有策略共享相同当前标签、预测和response计数，只改变B15/B31；其他预算未改变，所以`.2*ΔB15+.1*ΔB31`是六预算ALC的精确差值贡献。没有汇报伪造的完整绝对ALC。

{bt}

## 5. 理解与下一步

R78显示的风险排序与“何时关闭共享”是两个需要分别验证的主张。被判不可靠组平均仍略微受益，且单预算相关CI跨零；基于半样本fit预测的z=0，没有可靠对齐full-support query的收益零点。R79新增了真实peer方向的支持留出证据，仍未跨过可操作决策的门槛。

保留固定Partial8作为conditional_simulation研究候选；停止本轮二值配置，不开展B3/B7扩展、阈值扫描或门控网络。重开需要新的来源外辨识/校准证据与足够group样本；不是依已见dave结果造例外。

原概率、mask来源和完整命令见[复现说明](shared_reliability_reproduction.md)。R79/source为实际执行的来源阶段，source_contract记录所有源码/来源输入hash；source_replay_validation.json的Brier/accuracy最大误差为0。没有新增外层模型或Hosted操作。
'''
    save('shared_reliability_policy_findings.md',report)
    return ct+'\n\n'+agg+'\n\n'+pt+'\n\n'+bt
if __name__=='__main__':main()
