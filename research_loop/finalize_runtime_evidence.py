"""R69: classify clauses rather than treating baseline code as a hosted runner."""
from historical_positive_common import *

root=ROOT/'R69';official='https://aims.stanford.edu/competition/'
clauses=[
('multi_pair_labeled','verified_official','labeled contains entries across sampled subject–benchmark pairs','official_interface.md:131'),
('visible_subject_fields','verified_official','Exactly eight plain-string subject fields, missing empty string','official_interface.md:95'),
('visible_item_fields','verified_official','item_content, item_features, interactors, anonymous stable benchmark_id; item_id absent','official_interface.md:108'),
('budget_zero','verified_official','labeled is empty at B0','official_interface.md:120'),
('budget_nested','verified_official','0,1,3,7,15,31 per pair; nested trajectory; same evaluation items','official_interface.md:120'),
('acquisition_separation','verified_official','50/50 items per pair; all repeated responses stay with item; evaluation outcomes never revealed','official_interface.md:118'),
('exact_input_prediction_reuse','verified_official','Identical subject/item input reuses one probability for repeats','official_interface.md:87'),
('visible_subject_match','verified_official','Official empirical mean matches complete subject dict and benchmark_id','official_baseline/empirical_mean/model.py:26'),
('actual_hosted_peer_count','unknown','No executable hosted runner or actual evidence sample obtained',None),
('hosted_peer_selection','unknown','No guaranteed same-benchmark peer cohort or selection distribution',None),
('own_plus_peer_per_call','inferred_not_guaranteed','Multi-pair schema permits it, but actual calls and timing not observed',None),
('target_query_peer_acquisition_overlap','unknown','Per-pair split is explicit; cross-subject item overlap rules are not',None),
('cross_call_state','unknown','Do not assume call order or persistent state, including across budgets',None),
('same_text_means_same_item','unsupported','Identical text can differ in features/interactors; internal item_id unavailable',None),
('peer_labels_without_visibility','unsupported','Never infer unseen labels from historical IDs or hidden evaluation outcomes',None)]
data={'date':now(),'official_repo_commit':read(root/'retrieval.json')['official_commit'],
      'actual_hosted_runner_observed':False,'clauses':[dict(id=i,status=s,statement=t,evidence=e) for i,s,t,e in clauses],
      'minimum_contract':{'stateless':True,'B0_empty':True,'matching':'exact canonical tuple of all visible item fields; exact subject dict equality',
       'exclude_same_visible_subject_as_peer':True,'unknown_or_ambiguous_match':'no same-item correction; R25 fallback',
       'peer_labels':'only supplied current-budget acquisition labels; never evaluation response of that peer',
       'simulation':'0/1/3 distinct visible peers, deterministic attribute-only assignment; conditional_simulation'},
      'bounded_retrieval':str(root/'retrieval.json')}
write_json(root/'runtime_contract.json',data);write_json(REPO/'research_loop/PAIEC_RUNTIME_EVIDENCE_CONTRACT.json',data)
lines=['# PAIEC runtime evidence contract — R69','',f'核查日期：{data["date"]}。官方repo `{data["official_repo_commit"]}`；原文副本与hash见R69/retrieval.json。','',
    '## 条款状态','', '|条款|状态|证据/边界|','|---|---|---|']
for i,s,t,e in clauses:lines.append(f'|{i}|{s}|{t}; {e or "无可核验runner条款"}|')
lines+=['','## 保守实验合同','',
'仅使用当前调用的允许标签；B0空；不依赖历史调用、缓存标签或匿名ID解码。同题匹配使用 benchmark_id/item_content/item_features/interactors 四字段完全一致；文本同而metadata不同不合并。完整subject字段相同不能借内部subject_id当作另一个peer。多对一的输入映射单独报告，无法从可见输入确认的匹配不赋予“同一内部item”语义。',
'','R72将运行0/1/3peer的条件模拟，peer由合法可见配置的确定性排序选择，未保证Hosted具有这种组成。一个peer的评价响应永不揭示；它的acquisition item可以在另一subject的query中出现，这是明确标注的跨subject重叠假设。自己的query响应不得进入任何拟合或校准。',
'','## 有界检索及证据缺口','',
'AnySearch检索与官方页面提取成功；官方GitHub完整tree和原baseline已取。AnySearch Codabench提取失败，公共API TLS EOF；未绕过认证或TLS。没有真实Hosted运行记录，故没有任何 verified_runtime 条款。未把unknown写成禁止共享。',
'','## 给组织者的问题草稿（未发送）','',
'1. predict的labeled是否同时含同一benchmark中所有抽中subject的当前预算标签？可否提供脱敏输入样例/runner版本？',
'2. 同benchmark subject数量/选取方式是什么？是否可能没有peer？各budget是否对全部pair同步截断？',
'3. acquisition/evaluation划分是每subject独立还是benchmark共享？某subject acquisition item是否允许出现在另一subject evaluation？',
'4. 对item_content/features/interactors完全相同但内部item_id不同的情况，应该如何理解？是否会提供稳定item标识？',
'5. predict调用顺序和进程状态是否隔离？是否保证每调用提供完整的当前允许证据？',
'',f'官方入口：{official}；repo：https://github.com/aims-foundations/paiec_baseline 。']
(REPO/'research_loop/PAIEC_RUNTIME_EVIDENCE_CONTRACT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
write_json(root/'status.json',{'status':'complete','runtime_guarantee':'not_observed','R72':'conditional_simulation_allowed_under_explicit_assumptions','utc':now()})
print(json.dumps({'clauses':len(clauses),'runtime_verified':0,'official_commit':data['official_repo_commit']}))
