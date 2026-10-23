"""Read-only scientific-asset checks, then plan_8 status/manifest writeback.

Does not retrain, rerun scoring, rewrite experimental outputs, or access secrets.
The two global narrative documents are the only allowed R72-manifest updates.
"""
import ast
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'research_loop'
RUNS = REPO / 'paiec_baseline/runs/research_loop'
OUT = RUNS / 'dual_track_v8'
CARD = Path('D:/todo/tasks/graph/nips-comp.dual-track-demand-sharing-2026-10-23.md')
TRACE = Path('D:/todo/reports/work-traces/nips-comp/2026-10-23-dual-track-demand-sharing.md')
CACHE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        CACHE[key] = h.hexdigest()
    return CACHE[key]


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def manifest_entries(path):
    data = read(path)
    if isinstance(data, list):
        return data
    return data['entries'] if 'entries' in data else data['files']


def verify_entries(entries, allowed=()):
    changed, count, size = [], 0, 0
    for row in entries:
        path = Path(row['path']).resolve()
        actual = sha(path)
        if path in allowed:
            changed.append({'path': str(path), 'old_sha256': row['sha256'], 'current_sha256': actual,
                            'reason': 'plan_8 explicitly requires global state/ledger update'})
        else:
            assert actual == row['sha256'], f'Unexpected asset change: {path}'
            count += 1
            size += path.stat().st_size
    return {'unchanged_files': count, 'unchanged_bytes': size, 'authorized_global_updates': changed}


def verify_contract(path):
    data = read(path)
    for p, expected in data['sha256'].items():
        assert sha(p) == expected, f'Frozen source drift: {p}'
    return {'path': str(path), 'files': len(data['sha256'])}


def collected_files():
    paths = {Path(__file__), REPO / 'doc/plan_8.md', CARD, TRACE}
    for name in ['DUAL_TRACK_SPEC.md', 'HIGH_LEVEL_STATE.md', 'HISTORICAL_POSITIVE_SIGNAL_LEDGER.md',
                 'prepare_demand_annotation.py', 'audit_demand_annotations.py', 'validate_demand_provenance.py',
                 'plot_demand_annotation.py', 'demand_source_refresh.py', 'dual_track_cpu_features.py',
                 'verify_dual_track_cpu_features.py']:
        paths.add(SRC / name)
    for pattern in ['demand_pilot*', 'demand_student*', 'shared_slope*', 'SHARED_SLOPE*']:
        paths.update(p for p in SRC.glob(pattern) if p.is_file())
    for folder in [SRC / 'literature/dual_track_demand', *(RUNS / name for name in ['R73', 'R74', 'R75', 'R76', 'dual_track_v8'])]:
        paths.update(p for p in folder.rglob('*') if p.is_file())
    for pattern in ['demand_annotation*', 'demand_pilot*', 'shared_slope*', 'dual_track*']:
        for path in (REPO / 'reports').glob(pattern):
            paths.update([path] if path.is_file() else (p for p in path.rglob('*') if p.is_file()))
    return sorted((p.resolve() for p in paths if p.name != 'delivery_manifest.json' or p.parent != OUT), key=str)


def main():
    OUT.mkdir(exist_ok=True)
    result = {'scope': 'plan_8 R73-R76; completed conditional experiments, not research promotion or Hosted acceptance'}
    result['contracts'] = [verify_contract(RUNS / p) for p in [
        'R62/isolation/contract.json', 'R70/source_contract/contract.json',
        'R70/calibration_contract/contract.json', 'R70/full/contract/contract.json',
        'R71/full/contract/contract.json', 'R72/full/contract/contract.json',
        'R72/coverage_contract/contract.json', 'R74/contract_v2/contract.json',
        'R75/contract/contract.json', 'R76/full/contract/contract.json',
        'R76/smoke/contract/contract.json', 'R76/cpu_adapter_smoke/contract/contract.json']]
    assert result['contracts'][0]['files'] == 183
    print('Frozen sources verified', flush=True)

    allowed = {(SRC / n).resolve() for n in ['HIGH_LEVEL_STATE.md', 'HISTORICAL_POSITIVE_SIGNAL_LEDGER.md']}
    result['R72_delivery_preservation'] = verify_entries(manifest_entries(RUNS / 'R72/delivery_manifest.json'), allowed)
    assert len(result['R72_delivery_preservation']['authorized_global_updates']) == 2
    result['worker_manifests'] = {stage: verify_entries(manifest_entries(RUNS / stage / 'delivery_manifest.json'))
                                  for stage in ['R74', 'R76']}
    expected_fallback = 'e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9'
    result['fallback'] = []
    for name in ['layout_20261022_120427', 'nips-comp_20261022_113534']:
        p = Path('D:/nips-comp/paiec-dataset/paiec_git_backups') / name / 'workspace/paiec_baseline/dist/smoothed_mean_strength2.zip'
        assert sha(p) == expected_fallback
        result['fallback'].append({'path': str(p), 'sha256': expected_fallback})
    print('Historical scientific assets and worker manifests verified', flush=True)

    contract = read(RUNS / 'R73/annotation_contract.json')
    for relative, expected in contract['packet_hashes'].items():
        assert sha(RUNS / 'R73' / relative) == expected
    provenance = read(RUNS / 'R73/provenance_validation.json')
    for item in provenance['source_files_verified']:
        assert sha(item['path']) == item['sha256']
    final = RUNS / 'R73/final_annotations.jsonl'
    assert sha(final) == 'ab2c7d771ebd9b4010f7630209b8a876d4406467eb06bb6aac618832c781b8cf'
    records = [json.loads(s) for s in final.read_text(encoding='utf-8').splitlines()]
    assert len(records) == 512 and sum(v['grade'] is not None for v in records) == 436
    result['R73'] = {'main_records': 512, 'numeric_grades': 436, 'deliberate_null': 76,
                     'packet_hashes_verified': len(contract['packet_hashes']),
                     'source_hashes_verified': len(provenance['source_files_verified']), 'final_annotation_sha256': sha(final)}

    decision = read(RUNS / 'R74/decision.json')
    replay = read(RUNS / 'R74/probability_replay.json')
    assert decision['action'] == 'stop_at_pilot_keep_R25' and decision['candidate'] is None
    assert not decision['full_outer_authorized_by_source_contract'] and not (RUNS / 'R74/full/outer').exists()
    assert replay['max_brier_or_accuracy_replay_error'] == 0
    result['R74'] = {'decision': decision, 'probability_replay': replay}

    replay = read(RUNS / 'R76/full/replay_validation.json')
    summary = read(RUNS / 'R76/full/summary.json')
    assert (replay['pairs'], replay['trajectories'], replay['benchmarks'], replay['chunks'], replay['query_variant_rows']) == (5242, 15726, 36, 149, 1005741)
    assert replay['max_brier_error'] < 1e-12 and replay['max_accuracy_error'] == 0
    assert summary['unique_pairs'] == 5242 and summary['trajectories'] == 15726
    assert abs(summary['results']['R76_partial_selected']['alc_mean'] - .19054785628746854) < 1e-15
    assert abs(summary['results']['R25']['alc_mean'] - .19512392597599454) < 1e-15
    gate = read(RUNS / 'R76/full/gate.json')['R76_partial_selected']
    assert gate['four_numerical_conditions'] and gate['regression_review_required']
    assert gate['scope'] == 'conditional_simulation' and not gate['hosted_deployment']
    smoke = read(RUNS / 'R76/smoke/results.json')
    cpu_smoke = read(RUNS / 'R76/cpu_adapter_smoke/results.json')
    assert smoke['status'] == 'passed_synthetic_only' and len(smoke['cases']) == 16
    assert cpu_smoke['status'] == 'passed_new_text_synthetic_only'
    result['R76'] = {'replay': {k: v for k, v in replay.items() if k != 'checks'}, 'partial_gate': gate,
                     'protocol_smoke_cases': len(smoke['cases']), 'new_text_cpu_smoke': cpu_smoke['status'],
                     'cpu_resources': read(RUNS / 'R76/full/resources.json'),
                     'end_to_end_adapter_score_verified': False, 'hosted_verified': False}

    paths = collected_files()
    for p in paths:
        assert p.is_file(), p
        assert p.suffix != '.pyc' and '__pycache__' not in p.parts and not p.name.startswith('.env')
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    expected_outputs = {'delivery_validation.json', 'delivery_manifest.json'}
    broken = []
    for p in [REPO / 'reports/dual_track_demand_sharing_findings.md',
              *(REPO / 'reports' / name for name in ['demand_annotation_findings.md', 'demand_pilot_findings.md',
                                                     'demand_pilot_reproduction.md', 'shared_slope_findings.md',
                                                     'shared_slope_reproduction.md'])]:
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            resolved = (p.parent / target.split('#')[0]).resolve()
            if not resolved.exists() and not (resolved.parent == OUT and resolved.name in expected_outputs):
                broken.append({'report': str(p), 'target': target})
    assert not broken, broken
    result['documentation_links'] = 'passed'
    result['python_static_parse'] = 'passed; no experimental code altered'
    result['figures'] = {'visual_inspection': 'root viewed R73 annotation_quality, R74 source_comparisons/grade_mae and R76 budget_brier/cohort_delta_ci; readable',
                         'note': 'static exported scientific figures; no Hosted UI acceptance implied'}
    review = REPO / 'reports/dual_track_structural_review.md'
    assert '最终' in review.read_text(encoding='utf-8')
    result['review'] = {'path': str(review), 'sha256': sha(review), 'independent_read_only_reviewers': 1,
                        'confirmed_blocking_findings': 0, 'not_a_dual_review': True}
    result['status'] = 'passed'
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write(OUT / 'delivery_validation.json', result)

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M CST')
    text = CARD.read_text(encoding='utf-8')
    text = text.replace('status: doing', 'status: done', 1)
    text = re.sub(r'## 当前状态\n.*?\n## 下一步\n.*?\n## 边界',
        '## 当前状态\n\ndone：R73–R76完成。R74教师独立信息门槛失败，停止No-Peer外层并保留R25；R76固定partial8 ALC0.190547856，相对同cohort完全共享改善0.000180905，但dave仍严重退化，未晋级。全量概率重计分、CPU新输入smoke、独立只读结构复核及历史hash核对已完成。\n\n'
        '## 下一步\n\n本轮无待执行实验。No-Peer停止本轮扩张；Shared-Peer保留冻结候选，后续独立确认/支持标签风险诊断须新实验合同。Hosted继续暂停。\n\n## 边界', text, flags=re.S)
    marker = '## 完成证据'
    if marker not in text:
        text += ('\n' + marker + '\n\n- ' + stamp + '，Codex evidence，confidence high。\n'
                 '- 总报告：`D:/nips-comp/nips-comp/reports/dual_track_demand_sharing_findings.md`。\n'
                 '- 验证与hash：`D:/nips-comp/nips-comp/paiec_baseline/runs/research_loop/dual_track_v8/`。\n'
                 '- 15726轨迹/5242pairs/36BM/149chunks，R76 Brier回放最大误差8.88e-16，accuracy误差0。\n'
                 '- 旧183冻结资产、R72科学文件和两份fallback保留；两个全局状态文件按授权更新。\n'
                 '- 无Hosted恢复、正式提交或额外外部收费API；原生Luna计费未暴露，无人类投入时间记录。\n')
    CARD.write_text(text, encoding='utf-8')
    trace = TRACE.read_text(encoding='utf-8')
    marker = '<!-- plan8-final-delivery -->'
    if marker not in trace:
        trace += ('\n' + marker + '\n## ' + stamp + ' 完成\n\n'
                  '- 状态done：R73–R76按预设条件完成，非研究晋级。R74停止，R76仅conditional_simulation。\n'
                  '- R76 ALC0.190547856，同cohort增量−0.000180905，CI全负；dave仍退化+.006632866，不自动部署。\n'
                  '- 全量重计分、16项协议smoke、新文本CPU wrapper和124标签请求已完成；完整进程峰值3550.81MiB。\n'
                  '- 183旧冻结资产、R72科学文件、两份fallback hash核对；全局state/ledger有意更新。独立只读复核无阻断。\n'
                  '- 总报告：D:/nips-comp/nips-comp/reports/dual_track_demand_sharing_findings.md；数据R73/R74/R75/R76；交付验证dual_track_v8。\n'
                  '- 本段为AI执行记录；模型/批处理实测时长见原execution/resources，未推断用户time_spent_minutes。\n'
                  '- 下一步按总报告：No-Peer停止扩张，Shared候选等待独立确认/新合同；Hosted继续暂停。\n')
        TRACE.write_text(trace, encoding='utf-8')

    entries = []
    for p in collected_files():
        valid = not ('superseded' in p.name or (RUNS / 'R74/contract') in p.parents)
        entries.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p),
                        'current_evidence': valid,
                        **({'note': 'retained superseded audit or pre-fit failed contract; not final evidence'} if not valid else {})})
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'files': len(entries),
                'bytes': sum(e['bytes'] for e in entries), 'entries': entries,
                'excludes': 'this manifest only; external original models/data/papers referenced by verified contracts',
                'scope': result['scope']}
    write(OUT / 'delivery_manifest.json', manifest)
    print(json.dumps({'status': 'passed', 'new_manifest_files': len(entries), 'bytes': manifest['bytes'],
                      'old_R72_preserved': result['R72_delivery_preservation']['unchanged_files'],
                      'authorized_global_updates': 2, 'task_status': 'done'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
