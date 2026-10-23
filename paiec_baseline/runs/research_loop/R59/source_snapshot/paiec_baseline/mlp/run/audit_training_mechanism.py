"""Small deterministic contract checks before plan_2 training (no benchmark fitting)."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mlp.cohort_semantic import CachedSemantic
from mlp.training_mechanism_data import (raw_episodes, make_bank, reliability_fixed,
    row_weights, mlp_validation, write_json)
from mlp.training_mechanism_training import network, export, diagnostic, selections, evaluate
from mlp.evaluate import _load_directory, _score_group
from mlp.reproduction_episodes import DATA_ROOT


def main():
    checks = {}
    torch.set_num_threads(4)
    rng = np.random.default_rng(73)
    model = network('mlp')
    initial = export(model)
    x = rng.normal(size=(9, 2048)).astype(np.float64)
    params = tuple(torch.tensor(initial[k].T if k.startswith('W') else initial[k], dtype=torch.float64,
                                requires_grad=True) for k in ('W1','b1','W2','b2','W3','b3'))
    directions = tuple(torch.tensor(rng.normal(0, .002, size=p.shape)) for p in params)
    def f(*p):
        h = torch.relu(torch.tensor(x)@p[0].T+p[1])
        h = torch.relu(h@p[2].T+p[3])
        return h@p[4].T+p[5]
    f0, tangent = torch.autograd.functional.jvp(f, params, directions)
    actual = f(*(p+d for p,d in zip(params,directions))).detach().numpy()
    expected = float(np.sqrt(np.mean((actual-(f0+tangent).detach().numpy())**2)))
    current = {k: initial[k]+(d.numpy().T if k.startswith('W') else d.numpy())
               for k,d in zip(('W1','b1','W2','b2','W3','b3'),directions)}
    error = abs(diagnostic(x, initial, current)['taylor_rmse']-expected)
    assert error < 1e-12
    checks['taylor_vs_torch_jvp_error'] = error
    pair, mix, audit = row_weights(np.array(['a\0x','a\0x','a\0y','b\0z']))
    loss = np.array([1., 3., 4., 8.])
    assert abs(np.mean(pair*loss)-(2+4+8)/3) < 1e-6
    assert abs(np.mean((2*mix-pair)*loss)-((2+4)/2+8)/2) < 1e-6
    checks['risk_equations'] = audit
    assert selections([.25]+[.2]*30,'query')['stop_epoch'] == 6
    engine = CachedSemantic()
    name = 'abstract_reason/formatted_tables'
    fixed = reliability_fixed(name, engine, 4.)
    eps = list(raw_episodes(name, engine, 4.))
    bank = make_bank(eps)
    errors = {k: float(np.max(np.abs(fixed[k]-bank[k]))) for k in ('z','base','delta','fraction','weight')}
    assert max(errors.values()) == 0, errors
    checks['exact_historical_R43_replay'] = errors
    sampled = make_bank(list(raw_episodes(name, engine, 4., seeds=(5501,))))
    assert len(sampled['z']) == len(fixed['z'])
    assert [e['subject_id'] for e in sampled['meta']] == [e['subject_id'] for e in fixed['meta']]
    assert any(a['identity_hash'] != b['identity_hash'] for a,b in zip(bank['meta'],sampled['meta']))
    checks['resampling_same_pairs_rows_new_context'] = True
    zero = export(network('query'), {'mean': np.zeros(16), 'scale': np.ones(16), 'episode_only': np.array(False)})
    _, summary, _ = evaluate('query', fixed, zero)
    checks['R25_epoch0_curve_error'] = float(np.max(np.abs(summary['results']['model']['brier_mean']-fixed['baseline_brier'])))
    assert checks['R25_epoch0_curve_error'] < 1e-7
    mlp_eps = mlp_validation(name, engine)
    responses, subjects, items = _load_directory(DATA_ROOT, name)
    groups = responses.groupby('subject_id', sort=False)
    sid = next(s for s in subjects if str(s) == mlp_eps[0]['subject_id'])
    old = _score_group(groups.get_group(sid), subjects[sid], items, name, sid, 73, 64,
                       {'model': tuple(initial[k] for k in ('W1','b1','W2','b2','W3','b3'))}, (2.,))
    _, summary, _ = evaluate('mlp', mlp_eps[:1], initial)
    # Historical reference retains every response; only summation precision differs.
    metric_error = float(np.max(np.abs(np.asarray(old[0]['model:residual_2'])-
                                      summary['results']['model']['brier_mean'])))
    assert metric_error < 1e-7, metric_error
    checks['mlp_raw_response_equivalence_error'] = metric_error
    checks['mlp_raw_row_reference'] = old
    checks['mlp_aggregated_curve'] = summary['results']['model']['brier_mean']
    write_json('runs/research_loop/training_mechanism_v5/implementation_audit.json', checks)
    print(checks)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
