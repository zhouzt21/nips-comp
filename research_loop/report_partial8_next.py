"""Reproducible plan10 tables and figures; interpretations tied to observed results."""
from partial8_next_common import *
def fmt(x,d=9):return '不可估' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
def ci(x):return '['+', '.join(fmt(v) for v in x)+']'
def table(h,r):
    def cell(x):return str(x).replace('|','\\|')
    return '\n'.join(['|'+'|'.join(map(cell,h))+'|','|'+'|'.join(['---']*len(h))+'|',*['|'+'|'.join(map(cell,v))+'|' for v in r]])
def save(name,s):(REPO/'reports'/name).write_text(s.strip()+'\n',encoding='utf-8')

def report80():
    root=ROOT/'R80';summary=read(root/'summary.json');s=summary['deduplicated'];decision=read(root/'decision.json');v=read(root/'replay_validation.json');d=pd.read_csv(root/'deduplicated_metrics.csv')
    rows=[];errors=[];intervals=[]
    for k in ['half','k5','loio']:
        for n in ['15','31']:
            a=s[k][n];rows.append([k,n,fmt(a['request_BA'],6),fmt(a['BA_improvement'],6),ci(a['BA_improvement_ci95']),fmt(a['benchmark_mean_pair_spearman'],6),fmt(a['repeat_sign_agreement'],6)])
            errors.append([k,n,fmt(a['bias']),fmt(a['mae']),fmt(a['rmse']),fmt(a['mean_abs_fit_size_gap']),fmt(a['mean_abs_validation_gap'])])
            intervals.append([k,n,fmt(a['mean_interval_width'],6),f"{a['unknown_fraction']:.2%}",a['harmful_pairs'],a['harmful_BM'],fmt(a['harmful_query_mean']),ci(a['harmful_query_ci95']),f"{a['nominal_interval_contains_query']:.2%}"])
    t=table(['估计器','预算','请求级符号BA','相对half ΔBA','配对ΔBA 95% BM CI','BM平均pair rho','重复mask符号一致率'],rows)
    pools=table(['来源pool','估计器','预算','符号BA','相对half ΔBA','配对BM CI'],
        [[f,k,n,fmt(v['request_BA'],6),fmt(v['BA_improvement'],6),ci(v['BA_improvement_ci95'])] for f,dd in summary['source_pools'].items() for k,nn in dd.items() for n,v in nn.items()])
    et=table(['估计器','预算','z−query均值偏差','MAE','RMSE','平均|拟合量失配|','平均|支持到query剩余差|'],errors)
    it=table(['估计器','预算','条件区间平均宽度','跨零请求','有害组pairs','有害组BM','有害组真实query Δ','BM CI','区间包含经验query风险比例'],intervals)
    br=pd.read_csv(root/'refit_bootstrap_summary.csv');first=br.groupby('benchmark').selection_fold.min().to_dict();br=br[br.apply(lambda r:r.selection_fold==first[r.benchmark],axis=1)]
    bt=table(['预算','BM数','OOB z的BM平均std','BM平均符号错误率','符号正比例介于.1/.9的BM数'],
        [[n,len(g),fmt(g.z_std.mean(),6),f'{g.sign_error_fraction.mean():.2%}',int(((g.positive_fraction>.1)&(g.positive_fraction<.9)).sum())] for n,g in br.groupby('budget')])
    slicerows=[]
    for k,dd in d.groupby('estimator'):
        for n,gg in dd.groupby('budget'):
            for label,ss in [('alignment',gg[gg.benchmark.str.startswith('alignment_faking/')]),('without_alignment',gg[~gg.benchmark.str.startswith('alignment_faking/')]),('dave',gg[gg.benchmark.str.startswith('dave/')])]:
                slicerows.append([k,n,label,len(ss),fmt(ss.query_delta.mean()),fmt(ss.estimation_error.abs().mean()),f'{((ss.lo<=0)&(ss.hi>=0)).mean():.2%}'])
    slices=table(['估计器','预算','切片','请求数','full query partial−own','风险差MAE','条件区间跨零比例'],slicerows)
    report=f'''# R80：Full-support共享收益符号的可辨识性

2026-10-23，实际来源审计完成。决定 **`{decision['status']}`**；R82/R83未触发。合同[PARTIAL8_NEXT_SPEC](../research_loop/PARTIAL8_NEXT_SPEC.md)，[原文与代码](../research_loop/literature/partial8_next/source_evidence.md)。

## 1. 回答：更多复用标签是否足够？

LOIO显著减少了与full fit的样本量差异，风险MAE也下降，但收益符号没有稳定改善。B15从half的.556232降至.537646；B31从.574356升至.581774，提升CI跨零。k5提高mask重复稳定性，B15符号BA却下降，配对CI全负。**估计器更稳定、风险数值更接近，不等于收益零点判别更可靠。**

{t}

BA是预测“共享有害/有益”的两类召回均值，不是response分类准确率。这里按实际pair×seed×预算请求计数；R78报告还包含先平均seed/预算再计算的pair级统计，数值不能直接混用。rho仍先按pair平均seed，再在可估BM内算；24/35个BM可估，其他不是零相关。

## 2. 样本量失配直接分解

对每个mask使用同query评价对应fit模型，按heldout个数加权风险，得到Δcvfit；完整预算预测给出Δfull。恒等式：

`z−Δfull = (z−Δcvfit) + (Δcvfit−Δfull)`。

后项是测得的拟合样本量变化；前项混合支持抽样误差、query经验风险噪声、支持/query分布差异和训练依赖，不能直接全部命名为标签噪声。不是先平均query概率再评分。

{et}

LOIO在B15/B31将平均绝对fit-size差降到约.000137/.000080，但剩余支持→query差仍约.010294/.008386。因此“半支持→全支持失配是主要或唯一原因”缺少支持；它确实存在，却不是本轮中被消除后即可恢复符号识别的解释。两个绝对项不可相加解释成总误差占比。

固定特征的support/query均值距离与绝对误差的BM内平均rho，LOIO为{fmt(s['loio']['15']['feature_gap_error_BM_spearman'],6)}/{fmt(s['loio']['31']['feature_gap_error_BM_spearman'],6)}。此简单描述量未出现预期稳定正关联，不能据此排除更复杂分布失配或证明真实异质性。

## 3. 有害、收益小、无法判断必须分开

每请求用OOF逐item风险差做1000次item bootstrap，固定预测；lo>0标名义有害、hi<0标名义受益、否则无法判断。这些不是经校准的95%CV风险置信区间：训练集重叠、用同一批有限标签以及query噪声都破坏简单独立解释。区间实际包含经验query差的比例约80%–84%，也不能当真实期望风险覆盖率。

{it}

名义有害组的真实query均值仍非正，CI均跨零；即便只选择区间不跨零的请求，也未取得来源有害组确认。LOIO在B15/B31仍有约89.9%/83.4%的请求无法判断。不能把这些请求当共享有害，更不能重复R79的强制关闭。

## 4. 支持标签重采样与预算边界

每来源BM首个合法可见配置、seed0，B15/B31各32次item bootstrap重新拟合，peer当前前缀固定，原item整体out-of-bag评分；实际fit/holdout及概率落盘。该有限经验分布审计不产生新的标签。

{bt}

从15到31标签，数值误差和区间宽度有所下降，但未达到预设符号门槛。当前无法估计可靠决策需要的准确标签数；没有预算外标签证据，不外推“增加到某个n必然足够”。LOIO没有mask随机性，其repeat列不可估，不能解释为风险估计方差为零。

{slices}

alignment/dave是来源中的固定至多8pairs切片，不能当完整外层修复；R80没有改变query预测，任何风险估计器的query基线分数相同。表中dave只用于反例诊断，没有按BM名字拟合规则。

## 5. 实际范围、隔离与决定

三个合法来源池合计{v['source_trajectories']}轨迹暴露；按BM首个合法来源池去重{v['dedup_trajectories']}轨迹，即191pairs、35BM、3item seeds；主cohort7501、B15/B31。每mask在同holdout上比较own/pooled/partial，同item整体留出，peer只用当前预算前缀。隐藏own y翻转不改变fit预测；原R78 full query与half z数值身份核对通过。模型、数据与query完全固定。

保存概率重计分/分解最大误差{v['max_error']:.3g}。新估计器未达到BA两预算>=.60、稳定配对改善与来源有害组确认，因此停止R82/R83；失败不依赖单一人为界值，因为B15没有改善、B31提升区间跨零、有害组也未确认。未追加B3/B7、风险校准器或阈值搜索。

原始概率、mask、计数、风险校准分箱、逐BM/预算以及正确率见R80目录和[总表](partial8_decision_tables.md)；所有来源结果仍为公开数据探索，不是独立盲测。理论依据仅约束解释：JMLR2004 §2.3/Theorem6/§8.3不保证LOIO在PAIEC有效，官方分组代码只提供隔离逻辑参考。
'''
    save('partial8_identifiability_findings.md',report)
    q=d[d.estimator=='half'].groupby('budget').mean(numeric_only=True)
    qt=table(['预算','模型','query Brier','response准确率'],[[n,k,fmt(r['query_brier_'+k]),f"{r['query_accuracy_'+k]:.6%}"] for n,r in q.iterrows() for k in ['own','pooled','partial']])
    return ['## R80 符号与排序','',t,'','## R80 各合法来源池','',pools,'','## R80 风险误差分解','',et,'','## R80 条件不确定性','',it,'','## R80 OOB重拟合','',bt,'','## R80 alignment/rest/dave切片','',slices,'','## R80 来源query分数与正确率','',qt]

def report81():
    root=ROOT/'R81';c=read(root/'comparisons.json');g=pd.read_csv(root/'geometry_by_cohort_budget.csv');h=read(root/'heterogeneity/summary.json');s=read(root/'summary.json');audit=read(root/'replay_validation.json');cost=read(root/'geometry_summary.json')
    comparisons=table(['对照','partial−对照平均ΔALC','BM-equal Δ','95% BM cluster CI','去alignment Δ'],
        [[k,fmt(v['delta']),fmt(v['benchmark_equal_delta']),ci(v['cluster_ci95']),fmt(v['without_alignment_delta'])] for k,v in c['cohort_average'].items()])
    per=table(['cohort','对照','partial−对照Δ','CI','三foldΔ'],[[co,k,fmt(v['delta']),ci(v['cluster_ci95']),'/'.join(fmt(x) for x in v['fold_delta'])] for co,d in c['cohorts'].items() for k,v in d.items()])
    geom=table(['cohort','预算','partial/pool平方成本','符号翻转率','同向缩小率','同向放大率','raw翻转率','partial clip率','point边界不匹配率'],
        [[r.cohort,r.budget,fmt(r.partial_cost/r.pool_cost if r.pool_cost>0 else None,6),f'{r.direction_flip:.6%}',f'{r.same_direction_shrink:.6%}',f'{r.same_direction_enlarge:.6%}',f'{r.raw_direction_flip:.6%}',f'{r.partial_clip_fraction:.6%}',f'{r.point_boundary_mismatch:.6%}'] for r in g[g.budget>=3].itertuples()])
    real=table(['peer场景','预算','δ留出支持风险增量','BM CI','重复δ平均cosine','BM CI','cosine可估BM'],
        [[k,n,fmt(v['offset_risk']['mean']),ci(v['offset_risk']['ci95']),fmt(v['repeat_cosine']['mean'],6),ci(v['repeat_cosine']['ci95']),v['repeat_cosine']['benchmarks']] for k,d in h['real'].items() for n,v in d.items()])
    synth=table(['已知世界','预算','对照','partial expected Brier Δ','BM CI'],
        [[k,n,b,fmt(v['mean']),ci(v['ci95'])] for k,d in h['synthetic'].items() for n,dd in d.items() for b,v in dd.items()])
    budgets=table(['对照','B0','B1','B3','B7','B15','B31'],[[k,*[fmt(x) for x in v['per_budget_delta']]] for k,v in c['cohort_average'].items()])
    failures=table(['cohort','对照','最差Benchmark','partial−对照 ΔALC','>=.005回归BM数'],
        [[co,k,v['worst_benchmark'],fmt(v['worst_delta']),len(v['regression_ge_005'])] for co,dd in c['cohorts'].items() for k,v in dd.items()])
    baseline=table(['cohort','基线','partial−基线 ΔALC','BM-equal Δ','CI','三fold Δ','最差BM及Δ'],
        [[co,k,fmt(v['delta']),fmt(v['benchmark_equal_delta']),ci(v['cluster_ci95']),'/'.join(fmt(x) for x in v['fold_delta']),v['worst_benchmark']+' '+fmt(v['worst_delta'])] for co,dd in c['baseline_comparisons'].items() for k,v in dd.items()])
    peer=table(['cohort','预算','平均可见peers','平均提取peer前缀标签','用于共享的平均额外标签','每peer上限','own标签'],
        [[int(r.cohort),int(r.budget),fmt(r.active_peers,3),fmt(r.peer_labels,3),fmt(r.peer_labels if r.budget>=3 else 0.,3),int(r.budget),int(r.budget)] for r in g.itertuples()])
    decomp=table(['cohort','预算','cost−partial 对齐项','cost−partial 平方项','point−partial 对齐项','point−partial 平方项'],
        [[int(r.cohort),int(r.budget),fmt(r.cost_alignment),fmt(r.cost_squared_change),fmt(r.point_alignment),fmt(r.point_squared_change)] for r in g[g.budget>=3].itertuples()])
    report=f'''# R81：Partial8方向、幅度与已知机制对照

2026-10-23。完整五cohort机制评价及真实/合成偏移检验均已完成。仍为`conditional_simulation`；exact-cost pooled是离线诊断，不是新提交方法。

## 0. 本轮学到了什么

**实际结果：** Partial8相对pooled的组成稳健性增量仍成立，但对逐点幅度控制的平均ΔALC只有−.000017538，CI[−.000020828,+.000005666]；对严格成本控制为−.000192802，CI[−.000229252,+.000056609]，两者BM-equal及去alignment结果均略退化。没有建立超出这些机制控制的稳定优势。

保留Partial逐点幅度、换回pooled正负方向的point控制，重现了相对pooled约90.25%的pair-equal点估计增益。主cohort的平方修正成本降至pooled的95.30%/94.49%/93.99%/92.80%（B3/7/15/31）；方向翻转约4.10%/2.39%/2.27%/2.48%，不是完全没有新方向。point边界幅度不匹配为0，因此本轮幅度控制的解释比有大量clip不匹配时更直接。该90.25%仅为本组控制的描述性分解，不是可识别的因果贡献比例。

**推断：** 证据更支持收缩和逐item幅度整形是重要机制；真实Subject方向存在弱重复性，但其独立预测效用尚未确认。不能断言“全部只是噪声”，也不能证明“纯标量收缩完全等价”。

## 1. 相同信息与成本下的增量

{comparisons}

{per}

五cohort先在pair/seed内平均风险，再按BM cluster计算。query集合、acquisition前缀、response counts相同，不选有利cohort、不新增独立BM数量。主基线R25 .195123926、pooled7501 .190728761、partial7501 .190547856均精确保留。

## 2. 两种方向控制的不同含义

- point：保留Partial每点绝对修正，换成pooled的正负号，再clip。该比较直接接近“符号改变是否有益”，但边界和零pooled方向可能导致幅度不再匹配。
- cost：整组query用一个非负倍数缩放pooled修正，使response加权平方范数与partial一致，再clip。未看successes，但用整组query/计数，不保证运行时可见，所以仅离线。
- source shrink：沿用R77合法来源投影alpha，不重拟合，不以outer选幅度。

cost共{cost['cost_total_requests']}个B3–B31请求×cohort，其中{cost['cost_unreachable_requests']}个边界下无法达到目标；可达项最大平方成本误差{cost['max_reachable_cost_error']:.3g}。控制总范数仍不能唯一分离向量方向、逐item相对幅度、clip与真正Subject能力差异。即便partial优于cost，也只说明全局幅度不是充分解释，不能自动推出主要收益来自δ的真实能力信息。

{geom}

方向率按原response权重、pair/seed等权；数值零1e−12同时作用两向量。canonical汇总来自独立分析；原runner几何CSV的same/flip只对pool施加容差，分析已按合同重新计算两端容差并保留原CSV，概率/风险不变。raw定义围绕未clip的own预测，clip定义围绕合法own概率，二者不能混作同一方向。

## 3. 真实支持下的Subject偏移

复用R77固定主cohort首个完整四人组，25BM、seed0、B15/B31。两个half重复，每次互补mask，δ为Partial核中own-block/8产生的偏移；比较有/无δ的heldout风险，两者共享同一个Partial估计出的共同成分。该共同成分不等于另行拟合的fully pooled，避免把两个参数变化混成δ效应。

{real}

重复cosine在共同feature anchor上去均值计算，避免仅由截距差产生方向一致。置换仅破坏peer当前label对应关系，own标签不变；因此某些δ的own信息即使置换后仍可保留，不能要求所有关系机械归零。

实际δ重复cosine为.0747/.1292，CI均为正，23/25BM可估；两个退化几何BM不可估，未当作0。保留δ的heldout风险Δ约−.000102/−.000083，CI均跨零。peer置换后cosine更高，B31保留δ还有−.000458的风险改善（CI全负），符合own信息修正错误共享成分的解释；这不是恢复真实Subject能力方向的专属证据。

## 4. 合成共同斜率与真实差异

沿用相同25BM真实几何、4个Subject、预算和query计数，各世界每预算32次Bernoulli支持重采样，使用配对uniform随机数。共同latent斜率anchor SD固定.15；heterogeneous额外加入Subject不同的Gaussian方向，整体anchor SD .15。截距来自当前own平滑均值，真概率clip[.01,.99]，参数/clip率均保存。

{synth}

这里评分是已知真概率下的expected Brier，不是实际PAIEC成绩。共同latent世界的概率clip会引入非线性，因此不能称严格同一个概率线性斜率null；它仍可检验“没有新增Subject latent方向时是否也会出现Partial增量”。真实几何/拟合截距和有限预算限制合成外推。

共同latent世界中，partial−pooled为−.000036649/+.000021611，CI均跨零，且partial显著差于cost/point；没有稳定复现真实数据中的partial−pool收益。异质世界中，partial−pooled为−.000504238/−.000986951，CI全负，说明固定Partial能缓解人为设置的异质性损失；但对cost/point的CI仍跨零，B31还显著差于own +.001280056。已知存在真实差异，也不足以推出Partial的收益来自可靠恢复δ方向，或固定共享强度足够保守。

真概率clip比例在common B15/B31平均5.16%/6.75%，heterogeneous平均10.88%/12.09%；各BM最大值可达43.42%。这些数值及完整参数均保存，合成世界不能被描述为无clip、无限样本的理论线性模型。

## 5. 复核与证据边界

全部5242pairs×3seeds×6预算、36BM、五cohort。保存概率重计分Brier最大误差{audit['max_brier_error']:.3g}，raw clip身份误差{audit['max_raw_clip_error']:.3g}，风险分解误差{audit['max_risk_decomposition_error']:.3g}；accuracy/各fold/预算/失败BM见[全表](partial8_decision_tables.md)。`ctl−partial = 2*(partial−y)*(ctl−partial)+(ctl−partial)^2`按response聚合独立核对。

四个机制问题的最终判断及下一步见[总决策](../research_loop/PARTIAL8_NEXT_DECISION.md)。沿用KDD2004/MALSAR原始结构；本轮方向/成本与合成实验是项目自身的判别设计，不是论文原数值复现。所有数据仍被多轮自适应使用，严重负迁移和Hosted限制不会因机制对照自动解除。

三种强基线、逐预算、最差Benchmark、peer数量/实际额外标签与风险恒等分解均见[全表](partial8_decision_tables.md)。各cohort中每个有效peer最多提供当前预算n条标签；B1虽然代码提取了可见peer前缀，预测仍退回own，没有用于共享修正。B31平均约2.961个peer、91.794条额外标签，最多3peer/93条，加own31最多124条。共享方法的额外信息成本必须与No-Peer分开解释，不能称相同总标签预算优势。
'''
    save('partial8_mechanism_findings.md',report)
    metrics=table(['方法','预算','Brier','response准确率','ALC'],[[m,int(n),fmt(v['brier_mean'][j]),f"{v['accuracy_at_0_5'][j]:.6%}",fmt(v['alc_mean'])] for m,v in s['results'].items() for j,n in enumerate(BUDGETS)])
    return ['## R81 核心对照','',comparisons,'','## R81 各cohort/fold','',per,'','## R81 强基线与失败切片','',baseline,'',failures,'','## R81 平均风险逐预算差','',budgets,'','## R81 实际peer标签成本','',peer,'','## R81 方向和clip','',geom,'','## R81 风险差恒等分解','',decomp,'','## R81 真实偏移','',real,'','## R81 合成机制','',synth,'','## R81 完整分数/正确率','',metrics]

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter,MaxNLocator
    out=REPO/'reports/partial8_next_figures';out.mkdir(exist_ok=True);s=read(ROOT/'R80/summary.json')['deduplicated']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,k in enumerate(['k5','loio']):
        vals=np.array([s[k][n]['BA_improvement'] for n in ['15','31']]);bounds=np.array([s[k][n]['BA_improvement_ci95'] for n in ['15','31']]);x=np.arange(2)+(i-.5)*.1
        axes[0].errorbar(x,vals,yerr=[vals-bounds[:,0],bounds[:,1]-vals],fmt='o',label=k)
    axes[0].axhline(0,c='gray',ls='--');axes[0].set_xticks([0,1],[15,31]);axes[0].set_xlabel('Own label budget');axes[0].set_ylabel('Sign BA difference vs half (BM-cluster 95% CI)');axes[0].legend()
    for i,k in enumerate(['half','k5','loio']):axes[1].plot([15,31],[s[k][n]['mae'] for n in ['15','31']],'o-',label=k)
    axes[1].set_xticks([15,31]);axes[1].set_xlabel('Own label budget');axes[1].set_ylabel('Mean absolute error of query risk difference');axes[1].legend();fig.savefig(out/'risk_sign_and_error.png',dpi=180,bbox_inches='tight');plt.close(fig)
    if (ROOT/'R81/comparisons.json').exists():
        c=read(ROOT/'R81/comparisons.json')['cohort_average'];labels=list(c);vals=np.array([c[k]['delta'] for k in labels]);bounds=np.array([c[k]['cluster_ci95'] for k in labels])
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.errorbar(vals,np.arange(len(vals)),xerr=[vals-bounds[:,0],bounds[:,1]-vals],fmt='o');ax.set_yticks(range(len(labels)),labels);ax.axvline(0,c='gray',ls='--')
        ax.xaxis.set_major_locator(MaxNLocator(6));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x*1e4:.2f}'));ax.set_xlabel('Partial8 minus control ALC (×10⁻⁴; BM-cluster 95% CI)');fig.savefig(out/'cost_direction_controls.png',dpi=180,bbox_inches='tight');plt.close(fig)

def main():
    blocks=['# plan_10 实验指标、正确率和机制表','','R80为来源诊断；R81为完整公开机制评价及合成对照，不能当同一数据集横排。BA为收益符号识别，response accuracy为p>=.5分类准确率，二者不同。','']+report80()
    if (ROOT/'R81/heterogeneity/summary.json').exists():blocks+=['']+report81()
    res=[]
    for label,path in [('R80预测','R80/execution.json'),('R80分析','R80/analysis_execution.json'),('R81完整','R81/execution.json'),('R81合成/偏移','R81/heterogeneity/execution.json'),('R81完整分析','R81/analysis_full_execution.json'),('R81机制分析','R81/analysis_heterogeneity_execution.json')]:
        if (ROOT/path).exists():v=read(ROOT/path);res.append([label,fmt(v['seconds'],3),fmt(v['peak_working_set_mib'],2)])
    blocks+=['','## 实际资源','',table(['阶段','CPU运行墙钟秒','进程峰值MiB'],res),'','不是CPU core-seconds或人类投入时间；检索、实现、报告时间未混入实验耗时。']
    save('partial8_decision_tables.md','\n'.join(blocks));figures();print('Reports/figures generated from saved results',flush=True)
if __name__=='__main__':main()
