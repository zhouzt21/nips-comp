"""Generate standalone tables/figures and inventory from completed plan_2 artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mlp.evaluate import BUDGETS
from mlp.training_mechanism_data import write_json
from mlp.training_mechanism_guard import sha
from mlp.run.run_training_mechanism_round import ROOT, info

REPORT = Path('../reports')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def collect():
    records = []
    for fold in range(3):
        source = info(fold)
        for name in ('R25','smoothed_mean_2','training_selected_smoothed','constant_0.5'):
            records.append({'round':'baseline','fold':fold,'method':name,'epoch':None,
                'pairs':source['training_pairs'], 'metrics':source['baseline_summary']['results'][name],
                'benchmark_equal_alc':source['baseline_summary']['benchmark_equal_alc'][name],
                'by_benchmark': {b:{'pairs':v['pairs'],'brier':v['brier_mean'][name],'alc':v['alc_mean'][name]}
                                 for b,v in source['baseline_summary']['by_benchmark'].items()}})
        def add(round_,method,epoch,summary):
            records.append({'round':round_,'fold':fold,'method':method,'epoch':epoch,'pairs':summary['pairs'],
                'metrics':summary['results']['model'],'benchmark_equal_alc':summary['benchmark_equal_alc']['model'],
                'by_benchmark':{b:{'pairs':v['pairs'],'brier':v['brier_mean']['model'],'alc':v['alc_mean']['model']}
                                for b,v in summary['by_benchmark'].items()}})
        for family in ('mlp','query','episode'):
            analysis = read(ROOT/'R54'/f'fold{fold}_{family}_analysis.json')
            for rule in ('patience_epoch','best_same_steps','best20','best30','fixed20','fixed30'):
                add('R54',f'{family}_{rule}',analysis['selection'][rule],analysis['scores'][rule])
        path = ROOT/'R55'/f'fold{fold}_selection.json'
        if path.exists():
            selected = read(path)
            for family,summary in selected['models'].items():
                add('R55',f'resampled_{family}',selected['epochs'][family],summary)
        path = ROOT/'R56'/f'fold{fold}_selection.json'
        if path.exists():
            selected = read(path)
            add('R56','benchmark_mixture',selected['epoch'],selected['summary'])
        for family in ('mlp','query','episode'):
            path = ROOT/'R57'/f'fold{fold}_{family}_scores.json'
            if path.exists():
                selected = read(path)
                for method,summary in selected['summaries'].items():
                    add('R57',f'{family}_{method}',selected['window'],summary)
        path = ROOT/'R58'/f'fold{fold}_scores.json'
        if path.exists():
            selected = read(path)
            add('R58','output_scale_0.1',selected['analysis']['selection']['best30'],selected['scores']['summary'])
    return records


def tables(records):
    write_json(REPORT/'training_mechanism_v5_metrics.json',records)
    columns = ['round','fold','method','epoch','pairs','budget','brier','accuracy_at_0_5','alc','benchmark_equal_alc']
    with (REPORT/'training_mechanism_v5_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for record in records:
            for j,budget in enumerate(BUDGETS):
                writer.writerow({**{k:record[k] for k in columns[:5]},'budget':budget,
                    'brier':record['metrics']['brier_mean'][j],'accuracy_at_0_5':record['metrics']['accuracy_at_0_5'][j],
                    'alc':record['metrics']['alc_mean'],'benchmark_equal_alc':record['benchmark_equal_alc']})
    with (REPORT/'training_mechanism_v5_benchmark_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['round','fold','method','benchmark','pairs','budget','brier','alc'])
        for record in records:
            for benchmark,values in record['by_benchmark'].items():
                for j,budget in enumerate(BUDGETS):
                    writer.writerow([record['round'],record['fold'],record['method'],benchmark,values['pairs'],budget,values['brier'][j],values['alc']])


def figures(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    folder = REPORT/'training_mechanism_v5_figures'
    folder.mkdir(exist_ok=True)
    fig,axes = plt.subplots(2,3,figsize=(13,7))
    for fold in range(3):
        for row,family in enumerate(('mlp','query')):
            ax = axes[row,fold]
            for inner,color in ((0,'#1f77b4'),(1,'#d95f02')):
                hist = read(ROOT/'R54'/f'fold{fold}_{family}_inner{inner}'/'history.json')
                epochs = [h['epoch'] for h in hist]
                val = [h['validation']['results']['model']['brier_mean'][0] if family=='mlp'
                       else h['validation']['results']['model']['alc_mean'] for h in hist]
                fit = np.asarray([h['train_metric'] for h in hist])
                val = np.asarray(val)
                if family != 'mlp':
                    fit, val = fit-fit[0], val-val[0]
                    ax.axhline(0,color='gray',lw=.6)
                ax.plot(epochs,fit,color=color,ls='--',label=f'fit {inner}')
                ax.plot(epochs,val,color=color,label=f'held-out sources {inner}')
            ax.set_title(f'Source pool {fold}: '+('MLP B0 Brier' if family=='mlp' else 'R43 query ALC change from epoch 0'))
            ax.set_xlabel('epoch'); ax.set_ylabel('Brier / ALC (lower is better)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(folder/f'trajectories.{ext}',dpi=160)
    plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(13,6))
    for fold in range(3):
        for inner,color in ((0,'#1f77b4'),(1,'#d95f02')):
            hist = read(ROOT/'R54'/f'fold{fold}_mlp_inner{inner}'/'history.json')
            axes[0,fold].plot([h['epoch'] for h in hist],[h['diagnostic']['taylor_relative_error'] for h in hist],color=color,label=f'fit {inner}')
            axes[1,fold].plot([h['epoch'] for h in hist],[h['diagnostic']['hidden_cka'] for h in hist],color=color)
        axes[0,fold].axhline(.1,color='gray',ls=':');axes[1,fold].axhline(.9,color='gray',ls=':')
        axes[0,fold].set_title(f'Source pool {fold}: fixed fit probe')
        axes[0,fold].set_ylabel('Taylor relative error');axes[1,fold].set_ylabel('Hidden linear CKA')
        axes[1,fold].set_xlabel('epoch')
    axes[0,0].legend();fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(folder/f'representation_diagnostics.{ext}',dpi=160)
    plt.close(fig)
    if (ROOT/'R55'/'summary.json').exists():
        fig,axes = plt.subplots(1,3,figsize=(13,3.6))
        for fold in range(3):
            selected = [r for r in records if r['fold']==fold]
            base = next(r for r in selected if r['method']=='R25')['metrics']['brier_mean']
            for name,label in [('query_best20','fixed query'),('resampled_query','resampled query'),('resampled_episode','resampled episode')]:
                r = next(r for r in selected if r['method']==name)
                axes[fold].plot(range(6),np.asarray(r['metrics']['brier_mean'])-base,marker='o',label=label)
            axes[fold].axhline(0,color='black',lw=.8)
            axes[fold].set_xticks(range(6),BUDGETS);axes[fold].set_xlabel('support budget')
            axes[fold].set_ylabel('Brier minus R25');axes[fold].set_title(f'Source pool {fold}')
        axes[0].legend(fontsize=8);fig.tight_layout()
        for ext in ('png','svg'):
            fig.savefig(folder/f'resampling_by_budget.{ext}',dpi=160)
        plt.close(fig)


def inventory():
    artifacts = [p for p in ROOT.rglob('*') if p.is_file() and p.name != 'execution_inventory.json']
    artifacts += [p for p in Path('mlp').rglob('*training_mechanism*.py') if p.is_file()]
    artifacts += [Path('../research_loop/TRAINING_MECHANISM_V5_SPEC.md'),Path('../research_loop/TRAINING_MECHANISM_V5_ERRATA.md')]
    artifacts += list(REPORT.glob('training_mechanism_v5*.*'))
    artifacts += list((REPORT/'training_mechanism_v5_figures').glob('*'))
    write_json(ROOT/'execution_inventory.json',{'files':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in artifacts},
               'epoch_checkpoints':len(list(ROOT.glob('R*/fold*/epoch*.npz'))),
               'fits':len(list(ROOT.glob('R*/fold*/complete.json')))})


def findings(records):
    if not all((ROOT/r/'summary.json').exists() for r in ('R54','R55','R56','R57','R58')):
        return
    trajectory = read(ROOT/'R54'/'summary.json')
    resampling = read(ROOT/'R55'/'summary.json')['folds']
    risk = read(ROOT/'R56'/'summary.json')['folds']
    averaging = read(ROOT/'R57'/'summary.json')
    initialization = read(ROOT/'R58'/'summary.json')
    completions = read(ROOT/'completion_audit.json')
    cache = read(ROOT/'semantic_cache_preparation.json')
    def find(fold, method):
        return next(r for r in records if r['fold']==fold and r['method']==method)
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])
    mlp_stop = sum(a['stop_delayed_search'] for a in trajectory['mlp'])
    lines = ['# plan_2 五候选：实验结果与机制认识', '',
             '记录日期：2026-10-22。用户原文 `doc/plan_2.md` 保持原样。下述均为实际执行记录；条件未满足的实验单列，不算作已训练负结果。', '',
             '## 1. 执行结果', '',
             table(['候选','执行量 / 条件','来源内结果'],[
                 ['1 / R54 完整轨迹','18 fits × 30 epochs',f'MLP {mlp_stop}/3来源池双方向满足过拟合停止条件；无预定义延迟恢复'],
                 ['2 / R55 支持集重采样','12 fits × 20 epochs；4次预算内标签置换',f"通过来源进入条件 {sum(f['gate']['passes'] for f in resampling)}/3"],
                 ['3 / R56 Benchmark风险','6 fits × 30 epochs；λ=.5',f"通过来源进入条件 {sum(f['gate']['passes'] for f in risk)}/3"],
                 ['4 / R57 平均','R43参数EMA/SWA和最终概率平均','R43全部方法0/3通过；MLP无稳定窗口，未执行平均'],
                 ['5 / R58 初始化',f"触发来源池 {initialization['qualifying_source_pools']}/3，要求≥2",f"{initialization['status']}；新增fit {initialization['fits_executed']}"]]), '',
             '当前这组来源试验没有提供满足全体来源条件的新方案；没有执行新的5242-pair外层评分、组合或替换提交fallback。原R25的5242pairs×3seeds代理ALC 0.195123926及冻结均值0.198687363是历史总体结果，不能与下表不同来源池的数字直接比较。', '',
             '## 2. 比较协议和主要分数', '',
             '三个source pools分别316/260/248 pairs，沿用R25来源池及R43 Benchmark互斥二分，各拟合两个方向。池之间有重叠，不是三次独立盲测。每Benchmark最多25pairs、固定seed73、最多64query items，重复responses保留；预算0/1/3/7/15/31，ALC权重.1/.2/.2/.2/.2/.1。', '',
             '普通MLP用历史2048→128→64→1、每Benchmark最多400行、AdamW/Brier/residual-strength2；R43保持16→16→8→1、原16统计、g约束与0.01(g−1)²。MLP初始化seed2026，R43为4307；这是单初始化pilot，未完成新方法的多训练seed稳健性验收。来源选择使用B0（MLP）或ALC（R43），outer不参与。', '',
             table(['来源内 pair-equal ALC ↓','池0','池1','池2'],[
                 [method]+[f"{find(f,method)['metrics']['alc_mean']:.9f}" for f in range(3)] for method in
                 ('R25','smoothed_mean_2','training_selected_smoothed','mlp_best30','query_best20','resampled_query','resampled_episode','benchmark_mixture')]), '',
             '`best30`表示在epochs1–30中按来源B0选择MLP checkpoint，`best20`表示在epochs0–20中按来源ALC选择R43 checkpoint，并非固定取epoch30/20。完整patience、同停止步数最佳、固定20/30等规则在指标表分别列出。', '',
             '六预算Brier及固定p≥0.5的成功预测正确率全部在 [metrics.csv](training_mechanism_v5_metrics.csv)；逐Benchmark结果在 [benchmark_metrics.csv](training_mechanism_v5_benchmark_metrics.csv)。这里的正确率是预测器是否判对subject响应成败，不是subject本身的答题正确率；ALC也不是accuracy。', '',
             '## 3. 候选1：训练确实继续，但普通MLP泛化持续变差', '',
             table(['来源池','最佳来源epoch','最佳B0 Brier','epoch30 B0 Brier','最佳B0预测正确率','最佳B31预测正确率'],[
                 [f,trajectory['mlp'][f]['selection']['best30'],f"{find(f,'mlp_best30')['metrics']['brier_mean'][0]:.6f}",
                  f"{find(f,'mlp_fixed30')['metrics']['brier_mean'][0]:.6f}",
                  f"{find(f,'mlp_best30')['metrics']['accuracy_at_0_5'][0]:.2%}",
                  f"{find(f,'mlp_best30')['metrics']['accuracy_at_0_5'][5]:.2%}"] for f in range(3)]), '',
             '每个池的两块互斥来源验证组均符合冻结停止条件：epochs21–30训练误差低于epochs1–5，来源验证比早停最佳恶化，且没有连续3epochs恢复≥.001。因此停止本配置的延迟泛化/长Grokking搜索；这不证明任意更长训练或其他任务都不能grok。', '',
             'R43 query的best≤20与best≤30均为0/0/20，episode-only均为0/0/9；没有晚于原20epochs的新最优选择。**R43历史是固定20epochs+合并曲线argmin，新增patience5只是反事实对照**；不得解释为历史R43被早停截断。', '',
             '![训练及来源验证轨迹](training_mechanism_v5_figures/trajectories.png)', '',
             '## 4. 候选2：不同支持集是否让条件修正获得增量', '',
             '每epoch在相同合法pair的完整题池抽取新support/query，seed5501–5520；保持每pair query行数、四档预算、初始化、更新次数和fit参考标准化相同。固定组复用R54前20epochs。B0/B1按构造保持相同。', '',
             table(['来源池','重采样query所选epoch','相对R25 ΔALC','相对固定query ΔALC','相对重采样episode-only ΔALC','置换净差分'],[
                 [f['fold'],f['epochs']['query'],f"{f['gate']['delta']['R25']:+.9f}",f"{f['gate']['delta']['fixed_query']:+.9f}",
                  f"{f['gate']['delta']['resampled_episode']:+.9f}",f"{f['permutation']['difference_of_gains']:+.9f}"] for f in resampling]), '',
             '置换净差分是 `(model−R25)_真实 − mean(model−R25)_置换`，负值才支持模型增量依赖正确支持关系；每个预算内只置换已观察标签，重算d和全部统计，并同步评价置换R25。四次原始结果和逐pair分数均保留。选回epoch0时净差分为0，只说明本配置未获可采用的学习增量，不能据此否定R25已有局部对应信号。', '',
             '第三池相对固定query虽有约1.15e-5的局部改善，但置换净差分为正（约7.68e-4）：修正在错误对应关系下相对R25反而获益更多。一个与数据相容的解释是学到了幅度收缩/降噪，而非新增可迁移的题目对应信息；这仍是解释假设，不能从置换试验直接断言网络完全忽略标签。', '',
             '![重采样六预算比较](training_mechanism_v5_figures/resampling_by_budget.png)', '',
             '## 5. 候选3：风险权重改变了什么', '',
             table(['来源池','所选epoch','相对pair风险 ΔALC','相对R25 ΔALC','Benchmark-equal相对pair风险'],[
                 [f['fold'],f['epoch'],f"{f['gate']['delta']['pair_risk']:+.9f}",f"{f['gate']['delta']['R25']:+.9f}",
                  f"{f['gate']['benchmark_equal_delta_pair_risk']:+.9f}"] for f in risk]), '',
             '风险为pair和Benchmark两级平均的固定.5混合；不是GroupDRO的动态minimax。Benchmark风险质量保存在各fit的risk_audit.json；完整fit初始化梯度对比如下。没有额外抽样或搜索λ。', '',
             table(['来源池/方向','梯度cosine','混合/pair梯度norm'],[
                 [f'{f}/{i}',f"{read(ROOT/'R56'/f'fold{f}_mlp_inner{i}'/'risk_audit.json')['gradient']['cosine']:.6f}",
                  f"{read(ROOT/'R56'/f'fold{f}_mlp_inner{i}'/'risk_audit.json')['gradient']['norm_ratio']:.6f}"] for f in range(3) for i in range(2)]), '',
             '只有Benchmark-equal改善不能抵消真实pair-equal计分退化。本轮权重调整确实改变了有效梯度，但没有得到跨三个来源池一致的计分收益；是否能促成更可迁移的特征仍需额外证据。', '',
             '## 6. 候选4与5：条件进入与停止', '',
             'MLP三个来源池均无符合合同的3epoch稳定窗口，因此未对MLP强行平均。R43 query/episode均使用最早窗口[1,2,3]；对照包括best≤30、窗口内最佳单一权重，EMA=.9、SWA和三个checkpoint最终预算概率平均。没有新增训练更新。', '']
    average_rows = []
    for family,values in averaging.items():
        if values['status']=='evaluated':
            for method in ('ema','swa','probability_average'):
                average_rows.append([f'{family}/{method}']+[f"{v['gates'][method]['delta']['single_best']:+.9f}" for v in values['folds']]+
                                    [sum(v['gates'][method]['passes'] for v in values['folds'])])
    lines += [table(['平均方法','池0 ΔALC vs最佳单一','池1 ΔALC','池2 ΔALC','通过池数'],average_rows), '',
              '参数平均部署仍是单模型；概率平均需三个checkpoint的前向计算，此处未测Hosted端到端延迟。它是短窗口平均核心迁移，不是SWAD逐iteration密集loss-valley完整复现。', '',
              '初始化诊断：MLP六个方向epochs1–10的Taylor相对误差中位数范围0.808–0.941，hidden CKA中位数范围0.069–0.359，均不符合“长期接近初始化线性函数”的预声明条件；后期也无所需稳定来源增益。因此R58未触发，没有执行输出层尺度、学习率或weight-decay扫描。**未触发不等于已经证明改变初始化无效。**', '',
              '![初始化线性化与表示诊断](training_mechanism_v5_figures/representation_diagnostics.png)', '',
              '## 7. 对问题本身的认识', '',
              '1. **普通MLP早期已经改变表示，随后出现强烈的来源外退化。** 它很早就明显偏离初始化的一阶近似，训练拟合持续提高，来源外概率质量却显著变差。这与学习了来源特有相关性的解释一致，但本试验没有确定具体哪些特征造成退化。表示变化本身不能证明可迁移信息增加。', '',
              '2. **固定episode不是能够单独解释全部失败的原因。** 重采样是针对这一假设的单因素干预；应以相对R25、固定query及episode-only的共同结果解释，不能仅凭训练loss变好就宣称学会了题目对应关系。当前配置的进入条件未成立，继续加宽网络或扫描支持集策略缺少本轮证据。', '',
              '3. **稳定的来源窗口没有带来稳定的平均收益。** 留下全部checkpoint消除了“未观察后期”的缺口；早期验证平稳也不自动意味着平均后来源外更好。源内选择仍有自适应偏差。', '',
              '4. **pair目标与Benchmark目标必须分开报告。** 群组重权和较公平的来源覆盖是可检验干预，不是成绩提升的充分条件。最终判据仍是同一pair-equal Brier ALC及强均值/R25对照。', '',
              '5. **负结果的范围是已运行配置。** 本轮没有证明所有条件模型、所有初始化或所有表示学习都失败。已有证据更支持保留R25局部证据基线；新的计算投入需要新增输入结构、可靠冷启动信号或独立数据依据，不能把文献中的机制直接当成本任务的收益承诺。', '',
              '## 8. 文件、来源和复现入口', '',
              table(['文件','用途'],[
                  ['`doc/plan_2.md`','用户五候选完整原文'],
                  ['`research_loop/TRAINING_MECHANISM_V5_SPEC.md` / `...ERRATA.md`','冻结参数、进入/停止条件及历史名称澄清'],
                  ['`research_loop/literature/training_mechanism_v5.md` / `...sources.json`','AnySearch记录、原文定位、作者仓库commit、PDF SHA256、代码公开缺口'],
                  ['`reports/training_mechanism_v5_metrics.csv/json`','全部选择规则、六预算Brier、ALC和分类正确率'],
                  ['`reports/training_mechanism_v5_benchmark_metrics.csv`','逐Benchmark增减'],
                  ['`paiec_baseline/runs/research_loop/training_mechanism_v5/R54–R58/`','真实fit、epoch权重、逐pair指标、选择及置换、触发审计'],
                  ['同run根的`environment.json` / `core_provenance.json` / `stage_contracts`','环境、实际源文件与分组版本、阶段复用约束'],
                  ['同run根的`implementation_audit.json` / `result_audit.json` / `execution_inventory.json`','数值核对、匹配对照审计、产物SHA256'],
                  ['`research_loop/training_mechanism_v5_validation.md`','单个只读结构复核、修正与剩余证据边界']]), '',
              '命令在 `paiec_baseline` 运行：', '', '```powershell',
              'powershell -NoProfile -ExecutionPolicy Bypass -File D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1 -m mlp.run.run_training_mechanism_round trajectory',
              '# 其后按顺序把 trajectory 替换为 resample、benchmark_risk、averages、initialization',
              'powershell -NoProfile -ExecutionPolicy Bypass -File D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1 -m mlp.run.audit_training_mechanism_results',
              'powershell -NoProfile -ExecutionPolicy Bypass -File D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1 -m mlp.run.report_training_mechanism',
              '```', '',
              '缺失向量准备入口为 `mlp.run.prepare_training_mechanism_cache`，只用已迁移本地MiniLM。已完成结果会检查源码与输入hash后复用；源码变化必须创建新实验版本，不能混用旧选择。NPZ权重与外部缓存是本地产物，Git忽略不等于已随仓库分发。', '',
              '文献严格区分：Kumar/DomainBed/CNP/GroupDRO/SWAD/Kaddour/Woodworth提供机制和方法依据；2026谱理论仍为预印本，代码声明待公开，且齐次网络/耦合L2条件不覆盖本任务的bias+sigmoid+AdamW。详见来源表，不将本轮说成七篇论文原始成绩的完整复现。', '',
              '## 9. 计算、异常与证据边界', '',
              f"实际完成 {len(completions)} 个fit，保存 {sum(x['checkpoint_count'] for x in completions)} 个epoch checkpoint；fit记录耗时合计 {sum(x['seconds'] for x in completions):.1f} 秒，最长单fit {max(x['seconds'] for x in completions):.1f} 秒（均<1800秒）。此数不含抽样、特征准备、置换和文献整理。CPU为Ryzen7 4800H，Torch2.9.1+cpu，4threads，训练串行。", '',
              f"重采样新增本地MiniLM向量 {cache['encoded_total']} 条，预处理记录 {cache['seconds']:.1f} 秒。首次R55准备因缺缓存退出，尚未拟合；失败留在execution_attempts。补算保留原token截取规则；PowerShell曾把Tokenizer截断前长序列提示包装为NativeCommandError，Python仍写出完整准备记录，后续全部新episode能读取向量。", '',
              '完整轨迹只覆盖30epochs；训练初始化各一个seed。四个置换seed不等于四个训练seed。来源池重复使用、跨池重叠、checkpoint选择均限制结论强度；没有独立未见Benchmark盲测、隐藏测试或新正式提交。原0.005 Research Gate仍未通过。ADeLe继续按用户决定等待资源。', '']
    (REPORT/'training_mechanism_v5_findings.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    records = collect()
    tables(records)
    figures(records)
    findings(records)
    inventory()
    print(json.dumps({'records':len(records),'output':str(REPORT/'training_mechanism_v5_metrics.csv')}))


if __name__ == '__main__':
    main()
