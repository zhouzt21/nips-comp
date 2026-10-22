"""Small v3 state checkpoint, preserving historical reports and original artifacts."""
import json
from pathlib import Path


def main():
    state=Path('research_loop');root=Path('runs/research_loop');census=json.loads((root/'R32/census.json').read_text())
    complete=(root/'R32/summary.json').exists();status='R32 full-query contrast completed; interpretation pending' if complete else 'R32 full-query contrast running; census completed'
    if (root/'R32/measurement_conclusion.json').exists():status='R32 completed: query cap does not explain remaining gap at measured resolution. R33 conditional oracle diagnostic started; inspect its log/summary before restart.'
    if (root/'R33/summary.json').exists():status='R32 and R33 completed. Moderate-amplification oracle headroom exists; R34 support-only scale experiment is the current action. See ITERATION_V3.md and R34_run.log.'
    if (root/'R35/summary.json').exists():status='R32-R35 completed; R34 extra gain is small and its cluster interval crosses zero. Stop the scale grid. Next action is being chosen from existing information-source evidence; see ITERATION_V3.md.'
    if (root/'R36_run.log').exists():status='R32-R35 completed. R36 conditional cohort4 mean/slope factorial is active; inspect R36_run.log and summary before restarting. Isolated-pair gate remains unmet.'
    if (root/'R37_run.log').exists():status='R32-R36 completed. R37 conditional prevalence-partition experiment is active; inspect R37_run.log and summary before restarting. No candidate promoted.'
    if (root/'R39_run.log').exists():status='R32-R38 completed. R39 isolated-pair curvature-matched Bernoulli experiment is active; inspect R39_run.log and summary before restart. Partition/nearest-peer routes stopped; no promotion.'
    if (root/'R39/summary.json').exists():status='R32-R39 completed; Bernoulli versus R25 delta +0.000005675, cluster CI crosses zero. Stop local loss/link variants. R40 tests input-defined local item strata; inspect ITERATION_V3.md and its log/summary.'
    if (root/'R40/summary.json').exists():status='R32-R40 completed; inspect R40 summary and latest six-field decision in ITERATION_V3.md before further work. No automatic promotion or submission.'
    if (root/'R41/summary.json').exists():status='R32-R41 completed. R40 strata lose to R25 and smoothed2; no cluster grid. R41 empirical input-collision cost0.001076 is diagnostic only. R42 support-template source pilot is the current action; inspect log/summary. No candidate promoted.'
    if (root/'R42/summary.json').exists():status='R32-R42 completed. Inspect the R42 source-only admission condition and ITERATION_V3.md before any outer experiment. Source pilot scores are NOT OOD evidence. No candidate promoted.'
    if (root/'R43_run.log').exists():status='R32-R42 completed. R42 source pilot failed admission; no outer run. R43 trains query-conditioned versus episode-only reliability MLPs with benchmark-nested source selection. Inspect its process/log/summary; no promotion or submission.'
    if (root/'R43/summary.json').exists():status='R32-R43 completed through source/inner selection. Inspect R43 admit_outer before any outer run. Inner calibration scores are not OOD performance. Six-field decisions in ITERATION_V3.md remain authoritative.'
    if (root/'R44/summary.json').exists():status='R32-R44 completed. R42-R44 source entry conditions failed; no outer runs for those designs. R45 complementarity pilot is the current action; inspect its summary.'
    if (root/'R46_run.log').exists():status='R45 source pilot passed admission; R46 frozen equal-average outer evaluation started. Inspect actual log/summary; no promotion.'
    if (root/'R46/summary.json').exists():status='R32-R46 completed. R46 equal ensemble improves benchmark-equal but worsens pair-equal versus R25; Gate unmet. R47 checks multimedia observability before any pixel method. No candidate promoted.'
    if (root/'R48_run.log').exists():status='R32-R47 completed/audited. Pixel runtime contract unverified; public Codabench entry has TLS failure. R48 cross-panel oracle-optimism diagnostic is active; inspect actual log/summary before resuming. Extra evaluation labels are NOT budget-compliant scores; no promotion.'
    if (root/'R48/summary.json').exists():status='R32-R48 completed/audited. Read the R48 diagnostic and latest ITERATION_V3 decision; cross-panel scores use extra labels and cannot be promoted. Pixel runtime contract remains unverified. Original research gate unchanged.'
    if (root/'R49/summary.json').exists():status='R32-R49 completed/audited. R48 extra-panel signal persists but is not budget compliant; R49 support-role split loses all three source pools and is rejected. R50 audits omitted official interactor variation. Gate unchanged; no promotion.'
    if (root/'R50/summary.json').exists():status='R32-R50 completed/audited. Current six-field decisions are in ITERATION_V3.md. No job is implied to be running by these files. Gate remains unmet; fallback unchanged. R48 uses extra labels, R49 failed source admission, pixel runtime contract unverified.'
    text='''## Loop v3 当前状态

Authority: 根目录 loop_2.md，及四份相同目录的研究参考材料；取代旧loop_1的控制流程，保持所有已冻结实验纪律。R1–R31历史结果保留。

'''+status+'''

当前六字段研究判断与下一动作以 ITERATION_V3.md 为准。以下R32段落是已完成的测量审计摘要，不是仍待执行的任务。

全5242pair中5224受64-item限制；完整2179077queryitems/seed，原335247。R31收益77%来自预算15/31；B0/B1相同是当前算法的构造事实，不能直接推论输入不含冷启动信息。重复响应的经验within-item方差在两个预测器Brier差异中精确抵消，不能直接用该方差解释全部未达标差距。

R32先按benchmark×评价池大小×重复响应分层，从计数预先选择pair；每个被选pair完整评价pool，三个原种子、同一acquisition labels。分层外推与实际全pair评分分开。code/protocol/selection/log均在runs/research_loop/R32及research_loop/R32*。缺失语义向量只用原模型离线补D盘缓存。

研究Gate尚未通过；fallback不变；未正式提交。恢复时先检查最新轮次的进程、日志与summary，不要重复启动已完成的R32。日志存在不代表仍在后台运行，单benchmark结果不代表全轮完成。R33–35的幅度估计、R36–38的条件共享、R39的似然求解及当前输入分组研究的详细结论均见ITERATION_V3.md；历史v2段落不覆盖v3控制流程。
'''
    if complete and not (root/'R33/summary.json').exists():
        d=json.loads((root/'R32/summary.json').read_text());text+='\n实际结果：\n\n'+json.dumps({k:v for k,v in d.items() if k not in ['by_benchmark','estimated_full_brier','estimated_capped_brier']},ensure_ascii=False,indent=2)+'\n'
    if (root/'R32c/summary.json').exists():
        d=json.loads((root/'R32c/summary.json').read_text())
        text+='\nR32c已完成：主导benchmark全部4278pairs，三个原种子，全部evaluation items/responses。R25 full ALC='+str(d['full_alc']['R25'])+'；smoothed2='+str(d['full_alc']['smoothed_mean_2'])+'。改善从capped约0.004122变为full约0.004111。所有旧capped逐pair/seed/budget回放通过。仅此benchmark不能替代36benchmark汇总；重复响应经验方差不是不可约噪声下界。\n'
    if (root/'R32/measurement_conclusion.json').exists():
        d=json.loads((root/'R32/measurement_conclusion.json').read_text())
        text+='\nR32分层测量结论（不是全5242pair完整评分）：\n\n'+json.dumps({k:d[k] for k in ['hybrid_cap_effect','approx_design_ci95','cap_corrected_gain_estimate','remaining_gain_to_gate_estimate']},ensure_ascii=False,indent=2)+'\n'
    for name in ['HIGH_LEVEL_STATE.md','RESUME.md']:
        path=state/name;old=path.read_text(encoding='utf-8');heading,rest=old.split('\n',1);marker='<!-- current-v3 -->';end='<!-- end-current-v3 -->'
        if marker in rest:rest=rest.split(marker,1)[0]+rest.split(end,1)[1]
        path.write_text(heading+'\n\n'+marker+'\n'+text+'\n'+end+'\n\n'+rest.lstrip(),encoding='utf-8')
    registry=json.loads((state/'EXPERIMENT_REGISTRY.json').read_text());registry['authority']='../../loop_2.md'
    item=next((r for r in registry['rounds'] if r['id']=='R32'),None)
    if item is None:item={'id':'R32'};registry['rounds'].append(item)
    item.update(status='completed_measurement_diagnostic' if complete else 'running',protocol='research_loop/R32b_protocol.md',outputs='runs/research_loop/R32')
    if (root/'R33_run.log').exists():
        item=next((r for r in registry['rounds'] if r['id']=='R33'),None)
        if item is None:item={'id':'R33'};registry['rounds'].append(item)
        item.update(status='completed_oracle_diagnostic' if (root/'R33/summary.json').exists() else 'running',protocol='research_loop/R33_protocol.md',outputs='runs/research_loop/R33')
    if (root/'R34_run.log').exists():
        item=next((r for r in registry['rounds'] if r['id']=='R34'),None)
        if item is None:item={'id':'R34'};registry['rounds'].append(item)
        item.update(status='completed' if (root/'R34/summary.json').exists() else 'running',protocol='research_loop/R34_protocol.md',outputs='runs/research_loop/R34')
    if (root/'R35/summary.json').exists():
        item=next((r for r in registry['rounds'] if r['id']=='R35'),None)
        if item is None:item={'id':'R35'};registry['rounds'].append(item)
        item.update(status='completed_diagnostic_no_promotion',protocol='research_loop/R35_protocol.md',outputs='runs/research_loop/R35')
    if (root/'R36_run.log').exists():
        item=next((r for r in registry['rounds'] if r['id']=='R36'),None)
        if item is None:item={'id':'R36'};registry['rounds'].append(item)
        item.update(status='completed' if (root/'R36/summary.json').exists() else 'running',protocol='research_loop/R36_protocol.md',outputs='runs/research_loop/R36')
    if (root/'R37_run.log').exists():
        item=next((r for r in registry['rounds'] if r['id']=='R37'),None)
        if item is None:item={'id':'R37'};registry['rounds'].append(item)
        item.update(status='completed' if (root/'R37/summary.json').exists() else 'running',protocol='research_loop/R37_protocol.md',outputs='runs/research_loop/R37')
    for rid in ['R38','R39','R40','R41','R42','R43','R44','R45','R46','R47','R48','R49','R50']:
        if (root/(rid+'_run.log')).exists():
            item=next((r for r in registry['rounds'] if r['id']==rid),None)
            if item is None:item={'id':rid};registry['rounds'].append(item)
            item.update(status='completed' if (root/rid/'summary.json').exists() else 'running',protocol=f'research_loop/{rid}_protocol.md',outputs=f'runs/research_loop/{rid}')
    (state/'EXPERIMENT_REGISTRY.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    Path('reports/research_loop_v3_2026-10-22.md').write_text('# PAIEC Loop v3 简明实验记录\n\n最新结果与进入条件见 [证据和方法决策](research_loop_v3_findings.md)。\n\n'+text+'\n完整六字段记录见 [ITERATION_V3](../research_loop/ITERATION_V3.md)；命令见 [复现入口](research_loop_v3_reproduction.md)。\n',encoding='utf-8')
    print(status)


if __name__=='__main__':main()
