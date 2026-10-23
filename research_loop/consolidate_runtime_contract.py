"""Final evidence index; preserve earlier frozen snapshots used by experiments."""
from historical_positive_common import *
root=ROOT/'R69';base=read(REPO/'research_loop/PAIEC_RUNTIME_EVIDENCE_CONTRACT.json');add=read(REPO/'research_loop/PAIEC_RUNTIME_EVIDENCE_CONTRACT_CLIENT_ADDENDUM.json')
text=(root/'official_interface.md').read_text(encoding='utf-8').splitlines()
patterns={'multi_pair_labeled':'contains `[[subject, item], label]`','visible_subject_fields':'exactly the following keys',
 'visible_item_fields':'item\\_content:','budget_zero':'at budget 0, `labeled` is empty','budget_nested':'Every submission is evaluated at label budgets',
 'acquisition_separation':'randomly split 50/50','exact_input_prediction_reuse':'Repeated responses with identical inputs'}
for clause in base['clauses']:
    if clause['id'] in patterns:
        hits=[i+1 for i,line in enumerate(text) if patterns[clause['id']] in line]
        assert hits,(clause,patterns[clause['id']]);clause['evidence']='R69/official_interface.md:'+','.join(map(str,hits))
base['clauses']+=add['clauses'];base['version']='final-consolidated-20261023';base['date']=now()
base['frozen_snapshots']=['PAIEC_RUNTIME_EVIDENCE_CONTRACT.json','PAIEC_RUNTIME_EVIDENCE_CONTRACT_CLIENT_ADDENDUM.json']
base['evidence_note']='Official text locators regenerated from saved full text; experiment snapshots preserved unchanged.'
write_json(REPO/'research_loop/PAIEC_RUNTIME_EVIDENCE_CONTRACT_FINAL.json',base);write_json(root/'runtime_contract_final.json',base)
print(json.dumps({'clauses':len(base['clauses']),'verified_runtime':sum(c['status']=='verified_runtime' for c in base['clauses']),'finished':now()}))
