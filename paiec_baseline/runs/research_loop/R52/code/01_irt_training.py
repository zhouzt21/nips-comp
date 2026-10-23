"""R52: fixed-sample author calibration, source-only ability fit and cold start.

The calibration objective is E[-log p], as in reeval's released notebook,
with PAIEC pair-equal weighting.  Validation responses never select a fit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .amortized_irt import AmortizedIRT, fit_training_abilities_counts, make_nuisance_abilities
from .reproduction_training import fit_normalizer, save_checkpoint, transform, write_json


class IncompleteIRTFit(RuntimeError):
    """A predeclared resource/optimization limit was reached, not a negative result."""


def low_budget_gate(summary, method='irt_calibration'):
    """Fixed R52 entry gate for one source pool; ALC is reported separately."""
    baselines = ('R25', 'smoothed_mean_2', 'training_selected_smoothed')
    benchmarks = list(summary['by_benchmark'].values())
    if not benchmarks:
        raise ValueError('Gate requires benchmark-level evidence')
    candidate = summary['results'][method]['brier_mean']
    b0_wins = sum(group['brier_mean'][method][0] < .25 for group in benchmarks)
    b1_delta = {base: candidate[1]-summary['results'][base]['brier_mean'][1] for base in baselines}
    equal_curves = {name: np.mean([group['brier_mean'][name] for group in benchmarks], axis=0)
                    for name in (method, *baselines)}
    equal_delta = {base: float(equal_curves[method][1]-equal_curves[base][1]) for base in baselines}
    checks = {'pair_equal_b0_below_point25': candidate[0] < .25,
              'strict_benchmark_majority_b0_below_point25': b0_wins > len(benchmarks)/2,
              'pair_equal_b1_beats_all_three': all(value < 0 for value in b1_delta.values()),
              'benchmark_equal_b1_beats_all_three': all(value < 0 for value in equal_delta.values())}
    return {'admitted': all(checks.values()), 'checks': checks, 'pair_equal_b0': candidate[0],
            'benchmark_b0_wins': b0_wins, 'benchmark_count': len(benchmarks),
            'pair_equal_b1_deltas': b1_delta, 'benchmark_equal_b1_deltas': equal_delta,
            'benchmark_equal_brier': {name: curve.tolist() for name, curve in equal_curves.items()}}


def training_fingerprint(episodes):
    """Include fitted inputs and query observations; support labels are unused."""
    digest = hashlib.sha256()
    for ep in episodes:
        metadata = {key: ep[key] for key in ('benchmark', 'subject_id', 'subject_key', 'seed',
                                            'support_item_ids', 'query_item_ids')}
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        for key in ('support', 'query', 'counts', 'successes', 'subject'):
            value = np.ascontiguousarray(ep[key])
            digest.update(key.encode()); digest.update(str(value.shape).encode())
            digest.update(str(value.dtype).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def pack_fit_queries(episodes, norm):
    """Only query responses become fitting targets; subject keys are indices only."""
    if not episodes:
        raise ValueError('Cannot fit an empty source split')
    keys = sorted({ep['subject_key'] for ep in episodes})
    index = {key: i for i, key in enumerate(keys)}
    features = {}; items = []; counts = []; successes = []; subjects = []; weights = []
    ledger = []
    for ep in episodes:
        x = np.asarray(ep['query']); n = np.asarray(ep['counts'], np.float64)
        y = np.asarray(ep['successes'], np.float64)
        if (x.ndim != 2 or x.shape[1] != 640 or n.shape != (len(x),) or y.shape != n.shape
                or not len(x) or not np.isfinite(x).all() or not np.isfinite(n).all()
                or not np.isfinite(y).all() or np.any(n < 1) or np.any(y < 0)
                or np.any(y > n) or np.any(n != n.round()) or np.any(y != y.round())):
            raise ValueError('Invalid source query response counts/features')
        if set(ep['support_item_ids']) & set(ep['query_item_ids']):
            raise ValueError('Support/query item overlap in training episode')
        key = ep['subject_key']; feature = np.asarray(ep['subject'], np.float64)
        if feature.shape != (256,) or not np.isfinite(feature).all():
            raise ValueError('Invalid subject feature vector')
        if key in features and not np.array_equal(features[key], feature):
            raise ValueError('Canonical subject key has inconsistent allowed features')
        features[key] = feature
        items.append(transform(x, norm).astype(np.float64))
        counts.append(n); successes.append(y)
        subjects.append(np.full(len(x), index[key], np.int64))
        weights.append(np.full(len(x), 1 / (len(episodes) * n.sum()), np.float64))
        ledger.append({key_: ep[key_] for key_ in ('benchmark', 'subject_id', 'subject_key', 'seed',
                                                  'support_item_ids', 'query_item_ids')})
        ledger[-1].update({'query_counts': n.astype(int).tolist(),
                          'query_successes': y.astype(int).tolist()})
    tensors = {name: torch.from_numpy(np.concatenate(values)) for name, values in (
        ('items', items), ('counts', counts), ('successes', successes),
        ('subject_index', subjects), ('row_weights', weights))}
    tensors['subject_features'] = torch.from_numpy(np.stack([features[key] for key in keys]))
    return tensors, keys, ledger


def expected_pair_nll(difficulty, counts, successes, subject_index, nuisance, row_weights):
    """A chunk's contribution to the fixed full-fit expected Bernoulli NLL.

    row_weights[j]=1/(number_of_pairs*responses_in_pair_j).  Thus sum_j
    counts[j]*row_weights[j] equals one across the complete fit set.
    """
    logits = nuisance[:, subject_index] - difficulty[None, :]
    contribution = F.softplus(-logits) * successes + F.softplus(logits) * (counts-successes)
    return (contribution.mean(0) * row_weights).sum()


def ridge_ability(features, targets, alpha=1.0):
    """Uniform-subject ridge; the intercept is not penalized."""
    x = np.asarray(features, np.float64); y = np.asarray(targets, np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or alpha <= 0:
        raise ValueError('Invalid ridge inputs')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Ridge requires finite training-subject features and abilities')
    xmean, ymean = x.mean(0), float(y.mean())
    centered = x-xmean
    weight = np.linalg.solve(centered.T@centered + alpha*np.eye(x.shape[1]), centered.T@(y-ymean))
    bias = ymean-float(xmean@weight)
    return weight, bias


def train_amortized_irt(training, validation, folder, *, seed=5207, threads=4,
                        chunk_size=2048, max_outer_iterations=100, time_limit=1800,
                        source_fingerprint=None):
    """Fit one predeclared R52 direction; validation is used only for split IDs."""
    if chunk_size < 1 or max_outer_iterations < 1 or threads < 1 or time_limit <= 0:
        raise ValueError('Training limits must be positive')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    fit_benchmarks = sorted({ep['benchmark'] for ep in training})
    validation_benchmarks = sorted({ep['benchmark'] for ep in validation})
    if set(fit_benchmarks) & set(validation_benchmarks):
        raise ValueError('Fit/validation benchmarks overlap')
    contract = {'model_name': 'irt_calibration', 'seed': seed, 'threads': threads,
                'source_fingerprint': source_fingerprint,
                'training_fingerprint': training_fingerprint(training),
                'fit_benchmarks': fit_benchmarks, 'validation_benchmarks': validation_benchmarks,
                'training_pairs': len(training), 'validation_pairs': len(validation),
                'objective': 'fixed150 normal expected Bernoulli NLL; pair-equal response weighting',
                'query_targets_only': True, 'support_labels_used_in_fit': False,
                'normalization': 'common fit_normalizer over fit-only support/query features',
                'nuisance_samples': 150, 'nuisance_distribution': 'N(0,1)',
                'initialization': 'zero difficulty weight and bias', 'dtype': 'float64',
                'optimizer': 'LBFGS', 'learning_rate': .1, 'max_iter': 20,
                'history_size': 10, 'line_search_fn': 'strong_wolfe', 'weight_decay': 0.,
                'chunk_size': chunk_size, 'max_outer_iterations': max_outer_iterations,
                'convergence': {'gradient_inf': 1e-5, 'stable_loss_tolerance': 1e-9,
                                'stable_loss_rounds': 3, 'stable_gradient_inf': 1e-4},
                'time_limit_seconds': time_limit, 'training_ability_map_precision': 1.,
                'coldstart_ridge_alpha': 1., 'coldstart_ridge_weighting': 'uniform subjects',
                'coldstart_intercept_penalized': False, 'online_map_precision': 1.,
                'validation_epoch_selection': False}
    if (folder/'fit_contract.json').exists():
        if json.loads((folder/'fit_contract.json').read_text()) != contract:
            raise RuntimeError(f'Fit contract changed at {folder}')
        if (folder/'fit_summary.json').exists():
            existing = json.loads((folder/'fit_summary.json').read_text())
            if existing.get('status') == 'complete':
                for name, digest in existing['checkpoint_sha256'].items():
                    if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                        raise RuntimeError(f'Checkpoint changed at {folder/name}')
                return existing
        raise IncompleteIRTFit(f'Existing incomplete fit at {folder}; use an explicit recovery directory')
    started = time.monotonic(); history = []; closures = 0
    write_json(folder/'fit_contract.json', contract)

    def check_time():
        elapsed = time.monotonic()-started
        if elapsed > time_limit:
            raise IncompleteIRTFit(f'Predeclared {time_limit}-second per-fit CPU limit')

    torch.set_num_threads(threads); torch.manual_seed(seed)
    norm = fit_normalizer(training)
    tensors, subject_keys, ledger = pack_fit_queries(training, norm)
    write_json(folder/'fit_query_ledger.json', ledger)
    nuisance = make_nuisance_abilities(len(subject_keys), seed=seed, samples=150).double()
    torch.save(nuisance, folder/'nuisance_abilities.pt')
    model = AmortizedIRT(difficulty_bias=True).double()
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    model.ability.requires_grad_(False)
    optimizer = torch.optim.LBFGS(model.difficulty.parameters(), lr=.1, max_iter=20,
                                  history_size=10, line_search_fn='strong_wolfe')
    rows = len(tensors['items'])

    def closure():
        nonlocal closures
        check_time(); optimizer.zero_grad(set_to_none=True); total = 0.
        for start in range(0, rows, chunk_size):
            check_time(); ix = slice(start, min(start+chunk_size, rows))
            difficulty = model.difficulty(tensors['items'][ix]).squeeze(-1)
            loss = expected_pair_nll(difficulty, tensors['counts'][ix], tensors['successes'][ix],
                                     tensors['subject_index'][ix], nuisance, tensors['row_weights'][ix])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite calibration objective')
            loss.backward(); total += float(loss.detach())
        closures += 1
        return torch.tensor(total, dtype=torch.float64)

    def gradient_inf():
        return max(float(p.grad.abs().max()) for p in model.difficulty.parameters())

    try:
        objective = float(closure()); grad = gradient_inf(); stable_rounds = 0
        history.append({'outer_iteration': 0, 'training_expected_nll': objective,
                        'gradient_inf': grad, 'closure_calls': closures, 'elapsed_seconds': time.monotonic()-started})
        write_json(folder/'history.json', history)
        converged = grad <= 1e-5
        for iteration in range(1, max_outer_iterations+1):
            if converged: break
            check_time(); previous = objective
            optimizer.step(closure)
            objective = float(closure()); grad = gradient_inf()
            delta = abs(objective-previous)
            stable_rounds = stable_rounds+1 if delta <= 1e-9 and grad <= 1e-4 else 0
            converged = grad <= 1e-5 or stable_rounds >= 3
            history.append({'outer_iteration': iteration, 'training_expected_nll': objective,
                            'loss_change': delta, 'gradient_inf': grad, 'stable_rounds': stable_rounds,
                            'closure_calls': closures, 'elapsed_seconds': time.monotonic()-started})
            write_json(folder/'history.json', history)
            print(json.dumps({'fit': str(folder), **history[-1]}), flush=True)
        if not converged:
            raise IncompleteIRTFit('Maximum outer iterations reached without predeclared convergence')
        check_time(); model.difficulty.requires_grad_(False)
        with torch.no_grad():
            difficulty = torch.cat([model.difficulty(tensors['items'][start:start+chunk_size]).squeeze(-1)
                                    for start in range(0, rows, chunk_size)])
        theta, present = fit_training_abilities_counts(difficulty, tensors['counts'], tensors['successes'],
                                                       tensors['subject_index'], len(subject_keys),
                                                       prior_precision=1.)
        if not bool(present.all()) or not bool(torch.isfinite(theta).all()):
            raise RuntimeError('Every packed training subject must have a finite MAP estimate')
        check_time()
        sx = tensors['subject_features'].numpy(); theta_np = theta.numpy()
        weight, intercept = ridge_ability(sx, theta_np, alpha=1.)
        estimate = sx@weight+intercept; residual = estimate-theta_np
        with torch.no_grad():
            model.ability.weight.copy_(torch.from_numpy(weight[None, :]))
            model.ability.bias.fill_(intercept)
        totals = np.bincount(tensors['subject_index'].numpy(), weights=tensors['counts'].numpy(), minlength=len(subject_keys))
        wins = np.bincount(tensors['subject_index'].numpy(), weights=tensors['successes'].numpy(), minlength=len(subject_keys))
        subjects = []
        for idx, key in enumerate(subject_keys):
            matching = [entry for entry in ledger if entry['subject_key'] == key]
            subjects.append({'subject_key': key, 'index': idx, 'responses': int(totals[idx]),
                             'successes': int(wins[idx]), 'pairs': len(matching),
                             'benchmarks': sorted({entry['benchmark'] for entry in matching}),
                             'theta_map': float(theta_np[idx]), 'theta_from_features': float(estimate[idx]),
                             'ridge_residual': float(residual[idx])})
        diagnostics = {'subjects': subjects, 'ridge_rmse': float(np.sqrt(np.mean(residual**2))),
                       'ridge_mae': float(np.mean(np.abs(residual))), 'ridge_max_abs': float(np.max(np.abs(residual))),
                       'ridge_alpha': 1., 'ridge_intercept': intercept,
                       'ridge_weight_norm': float(np.linalg.norm(weight)),
                       'theta_mean': float(theta_np.mean()), 'theta_std': float(theta_np.std()),
                       'subject_weighting': 'uniform; response counts are diagnostics only'}
        write_json(folder/'subject_ability_diagnostics.json', diagnostics)
        np.savez_compressed(folder/'fit_subject_features.npz', subject_keys=np.asarray(subject_keys),
                            subject_features=sx, theta_map=theta_np, theta_from_features=estimate,
                            response_counts=totals, successes=wins)
        check_time(); model.float().eval()
        metadata = {**contract, 'converged': True, 'closure_calls': closures,
                    'final_gradient_inf': grad, 'final_expected_nll': objective}
        save_checkpoint(folder/'final.pt', model, 'irt_calibration', norm, len(history)-1, metadata)
        with torch.no_grad():
            model.ability.weight.zero_(); model.ability.bias.zero_()
        save_checkpoint(folder/'zero_subject_prior.pt', model, 'irt_calibration', norm, len(history)-1,
                        {**metadata, 'ablation': 'zero entire subject prior head; no retraining'})
        summary = {'status': 'complete', 'model_name': 'irt_calibration', 'converged': True,
                   'iterations': len(history)-1, 'closure_calls': closures,
                   'training_pairs': len(training), 'query_rows': rows, 'training_subjects': len(subject_keys),
                   'elapsed_seconds': time.monotonic()-started,
                   'final_expected_nll': objective, 'gradient_inf': grad,
                   'ridge_rmse': diagnostics['ridge_rmse'],
                   'checkpoint_sha256': {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                         for name in ('final.pt', 'zero_subject_prior.pt')}}
        check_time(); write_json(folder/'fit_summary.json', summary)
        return summary
    except (IncompleteIRTFit, FloatingPointError) as exc:
        stop = {'status': 'incomplete', 'reason': str(exc), 'elapsed_seconds': time.monotonic()-started,
                'last_completed_iteration': history[-1]['outer_iteration'] if history else None,
                'closure_calls': closures, 'not_a_completed_negative_result': True}
        write_json(folder/'resource_stop.json', stop); write_json(folder/'fit_summary.json', stop)
        raise IncompleteIRTFit(f'{folder}: {exc}') from exc
