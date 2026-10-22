"""Synchronize completed R22-R29 state without deleting prior evidence."""
from pathlib import Path

STATE=Path('research_loop')


def prepend(name,body):
    path=STATE/name;old=path.read_text(encoding='utf-8');heading,rest=old.split('\n',1)
    marker='<!-- current-v2-r29 -->';end='<!-- end-current-v2-r29 -->'
    if marker in rest:rest=rest.split(marker,1)[0]+rest.split(end,1)[1]
    path.write_text(heading+'\n\n'+marker+'\n'+body.strip()+'\n'+end+'\n\n'+rest.lstrip(),encoding='utf-8')


def main():
    current='''## 当前状态：R10–R31 均已完成

2026-10-22。下方原有章节是历史检查点；其中“R22 running”等文字已被本节替代。当前无研究训练进程。主报告 reports/research_loop_v2_2026-10-22.md；最新推理链 ITERATION_V2_13_20.md 与 ITERATION_V2_21_22.md。

- 独立pair三种子：R25=0.205952959，R7=0.206243741，smoothed2=0.207408367；相对均值改善0.001455407，三折改善，但远低于冻结0.005。R25相对R7有一折略退化。
- 条件cohort4三种子：R20=0.202834515，强peer=0.204959086；增量0.002124571，仍低于强对照门槛。hosted共享范围未验证，不能替代独立pair证据。
- R22真实训练扩大池未修复OOD prior；R23软subject效应无稳定增益；R24/R25分离了新增字段与正则改变；R26在线获取CI跨零；R27预算强度退化；R28对称先验仍输固定均值；R29二阶源任务协方差三折内层全部关闭。
- 原fallback保持：dist/smoothed_mean_strength2.zip，SHA256 E55EE0B267F3B345685EF072CFA511C3C017D9082256573793961EA20FEADCB9。没有晋级新模型、提交Codabench或公开发布。
- HF当前revision仍为54074453da7b9911be0f48e4c560be547bc84d21，实时检查见runs/research_loop/release_check.json。文件列表达100000项可能截断，不是仓库总文件数。没有发现新的独立benchmark发布。
- R30发现原100pairs/BM代理只覆盖737/5242合格pair。R31冻结模型，实际扩大四个被截断benchmark、复用其他完整benchmark；三种子全部合格pair R25=0.195123926，smoothed2=0.198687363，改善0.003563437。原子集逐预算回放一致。该覆盖敏感性仍保留64evaluation items上限，不是全response评分；alignment_faking占81.6%，也不是新增独立泛化证据。未替换原晋级门槛。
- 最终63项tools回归通过；R30/R31另有真实数据回放断言。没有新增正式提交包。

研究目标尚未全部完成。局部机制证据成立不等于0.005研究门槛、独立泛化与提交门槛通过。不能把已有公开集合重分折称为新盲测。
'''
    for name in ['HIGH_LEVEL_STATE.md','RESEARCH_STATE.md','RESUME.md']:prepend(name,current)
    prepend('HIGH_LEVEL_STATE.md','''## 下一层认识与尚未解决的问题

已测试的证据支持“当前少量标签下的item对应关系”比“跨任务固定正确率映射”更可靠，但不是普遍的架构定理。扩大样本覆盖、文本窗口、先验分布形状、固定非线性核、源任务二阶协方差均未带来足够增益。它们排除的是具体实现/假设，不能推论所有特征学习无效。

R30/R31现已完成pair覆盖审计与真实扩展。采样上限显著改变公开集合权重，但完整pair改善0.00356仍不达到门槛。下一开放问题是64-item评价样本及同一item重复response噪声与可被item输入解释的差异；尚未执行，不得当作已发现的限制。只有诊断证明缺口足以影响目标，才进入新特征或学习机制。不可继续原先已失败的rank/rho、固定mix、窗口或数据量网格。

独立泛化验收需要未参与设计的数据；目前没有已核实可用的新发布。该外部证据缺口不意味着所有本地研究都无法继续，也不能据此声称目标已经完成。
'''+current)
    prepend('RESUME.md',current+'''
## 可复现入口

从paiec_baseline：python -m mlp.checkpoint_v2 刷新报告；R27/R28/R29分别运行 mlp.run_budget_mean_round、mlp.run_symmetric_prior_round、mlp.run_covariance_round。所有已完成结果已存在，恢复时先读JSON和协议，不默认重跑。

D:/paiec_research_cache语义缓存仍有效；semantic_features.py没有修改。大文件继续放D。后续若做覆盖审计，先读取corpus_manifest及现有pair_rates统计，避免重新下载。保留旧ZIP、权重、负结果和所有研究日志。任何持续后台运行声明都必须对应实际进程/调度器。
''')
    prepend('RECOGNITION_GRAPH.md','''## R22–R29 新增认识关系

| 关系 | 类型/证据强度 | 实验与边界 |
|---|---|---|
| 扩大源训练池 → 修复OOD prior | M / 当前干预不支持 | R22一折退化且仍输均值；不外推所有规模 |
| n=1硬中心化 → 丢失局部斜率信息 | M / 代数事实 | R23软约束只局部改善，不证明新solver有效 |
| metadata → 独立局部信息 | M / 小幅支持 | R25三种子与打乱控制；一折相对语义退化 |
| 更好特征覆盖 → 更好均值估计 | E / 不保证 | R26方差获取CI跨零，收益主要伴随mean项 |
| iid参数minimax → PAIEC最低平均Brier | E / 不成立的推论 | R27实测退化，损失对象/采样假设不同 |
| 奇矩偏置 → 全部先验负迁移 | M / 不充分 | R28对称化改善E3仍输均值 |
| 源任务二阶方向 → 可迁移语义子空间 | M / 本估计器不支持 | R29内层三折均rho=0，无模型增益 |

R30/R31更新：pair上限确实改变权重，实际全合格pair验证改善0.00356，仍未过门槛。下一开放节点为evaluation item上限及重复response噪声；尚未执行，不当作已验证机制。
''')
    prepend('OPERATOR_LEDGER.md','''## R22–R29 操作及退出依据

| Round | 操作链 | 结果/停止条件 |
|---|---|---|
| R22 | 隔离覆盖与每周期计算 → 真实重训 → OOD检验 | 新MLP仍输均值；停止该池扩张 |
| R23 | 分解subject/item → 软约束 → mean-only控制 | 两折退化；停止该近似 |
| R24/R25 | 审计字段 → 信息/正则归因 → 分块惩罚 | 小幅增益三seed复核；不晋级 |
| R26 | 固定预测器 → 流式获取 → mean/item分解 | CI跨零；不开发正式labeling.py |
| R27 | 核对理论目标 → 解析风险检查 → 预算比较 | minimax性质不迁移为平均Brier优势 |
| R28 | 找回E3负证据 → 先验对称化 → 同矩控制 | 仍输固定均值；停止泛用shape拟合 |
| R29 | 分离方向与子空间 → beta二阶统计 → 内层分组选择 | 真实/打乱均全关；不扫rank/rho |
| R30/R31 | 审计覆盖 → 区分外推/实测 → 冻结模型扩大实际pair | 全5242pair改善0.00356仍未达门槛；不称新盲测 |
''')
    prepend('HYPOTHESIS_LEDGER.md','''## 当前假设更新（R22–R29）

1. 小训练覆盖不是已证实的唯一根因：固定计算扩大池后仍不胜均值。
2. metadata确有部分局部信息，但统一正则混淆了新增信号；分块惩罚恢复小增益。
3. 小预算的先验均值方向偏差解释E3部分损失；对称化和同矩Beta均未解决。
4. 用源任务斜率的二阶统计迁移“有用方向”是合理但未获支持的假设：内层选择全关。
5. pair覆盖已审计并实际扩大：原代理改变收益幅度，但未解释全部未达标问题。64items及重复响应噪声分离尚未执行。
''')
    prepend('DECISION_LOG.md','''## 当前模型决策（R29完成后）

不晋级新方法。R25是独立pair三种子最佳点估计；原代理改善0.00146，真实扩展至5242合格pair后改善0.00356，均不足冻结0.005；R20仅在条件共享场景相对强peer改善0.00212。保留原smoothed2 ZIP。负结果全保留；不能用已复用benchmark的新seed或扩大pair数宣称盲测。下一待论证诊断是evaluation item上限与重复响应噪声。
''')
    prepend('LITERATURE_EVIDENCE_LEDGER.md','''## R27–R29 文献用途

- Sheen & Mei, An elementary approach for minimax estimation of Bernoulli proportion in the restricted parameter space，https://arxiv.org/html/2009.11413v2 。AnySearch全文检索，已读intro Eq1、Section2；raw/v2_binomial_minimax.md。使用其中回顾的非受限iid参数minimax公式；不转移其受限参数新定理，不将参数风险与预测Brier混同。R27是实测负结果。
- Koenker & Gu, Empirical Bayes For the Reluctant Frequentist，https://arxiv.org/html/2404.03422v1 。AnySearch全文，已读Sections1–2；raw/v2_empirical_bayes.txt。共同采样模型、交换性和mixing distribution relevance构成适用条件。R28对称化是本地干预，不是该文在PAIEC的结果。页面arXiv标签与文内version日期不同，保留原文，不猜测版本历史。
- R29复用已核实ALPaCA的末层先验概念；没有引用其连续Gaussian回归表现来证明本任务有效。真实选择结果为关闭协方差。
''')
    path=STATE/'literature/raw/v2_empirical_bayes.txt';raw=path.read_bytes()
    if raw.startswith(b'\xff\xfe'):path.write_text(raw.decode('utf-16'),encoding='utf-8')
    print('Updated current state headers; historical records preserved')


if __name__=='__main__':main()
