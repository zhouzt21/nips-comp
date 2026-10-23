"""Matched training traces, source-only selection, and checkpoint diagnostics."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import BUDGETS, ALC_WEIGHTS
from .evidence_reliability import forward, QUERY_START
from .model import forward_batch
from .reproduction_episodes import aggregate, brier, accuracy
from .training_mechanism_data import digest, write_json, load_npz

PARAMS = ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')


def network(family, output_scale=1.):
    torch.manual_seed(2026 if family == 'mlp' else 4307)
    dims = (2048, 128, 64, 1) if family == 'mlp' else (16, 16, 8, 1)
    model = torch.nn.Sequential(torch.nn.Linear(dims[0], dims[1]), torch.nn.ReLU(),
        torch.nn.Linear(dims[1], dims[2]), torch.nn.ReLU(), torch.nn.Linear(dims[2], dims[3]))
    if family != 'mlp':
        torch.nn.init.zeros_(model[-1].weight)
        torch.nn.init.zeros_(model[-1].bias)
    else:
        with torch.no_grad():
            model[-1].weight.mul_(output_scale)
            model[-1].bias.mul_(output_scale)
    return model


def export(model, normalizer=None):
    result = {} if normalizer is None else dict(normalizer)
    for i, j in enumerate((0, 2, 4), 1):
        result[f'W{i}'] = model[j].weight.detach().numpy().T.copy()
        result[f'b{i}'] = model[j].bias.detach().numpy().copy()
    return result


def normalize(z, artifact):
    z = z.astype(float).copy()
    if bool(artifact['episode_only']):
        z[:, QUERY_START:] = 0
    return np.clip((z-artifact['mean'])/artifact['scale'], -5, 5).astype(np.float32)


def normalizer(data, episode_only):
    z = data['z'].astype(float).copy()
    if episode_only:
        z[:, QUERY_START:] = 0
    mean = np.average(z, axis=0, weights=data['weight'])
    scale = np.sqrt(np.average((z-mean)**2, axis=0, weights=data['weight']))
    return {'mean': mean, 'scale': np.where(scale > 1e-8, scale, 1.),
            'episode_only': np.array(episode_only)}


def diagnostic(x, initial, current):
    """Exact first directional derivative of the 3-linear-layer logit at theta0."""
    x = x.astype(float)
    a1 = x @ initial['W1'] + initial['b1']
    h1 = np.maximum(a1, 0)
    a2 = h1 @ initial['W2'] + initial['b2']
    h2 = np.maximum(a2, 0)
    f0 = h2 @ initial['W3'] + initial['b3']
    delta = {k: current[k].astype(float) - initial[k] for k in PARAMS}
    dh1 = (x @ delta['W1'] + delta['b1']) * (a1 > 0)
    dh2 = (dh1 @ initial['W2'] + h1 @ delta['W2'] + delta['b2']) * (a2 > 0)
    linear = f0 + dh2 @ initial['W3'] + h2 @ delta['W3'] + delta['b3']
    ht1 = np.maximum(x @ current['W1'] + current['b1'], 0)
    ht2 = np.maximum(ht1 @ current['W2'] + current['b2'], 0)
    ft = ht2 @ current['W3'] + current['b3']
    u, v = h2 - h2.mean(0), ht2 - ht2.mean(0)
    denominator = np.linalg.norm(u.T@u) * np.linalg.norm(v.T@v)
    return {'taylor_relative_error': float(np.linalg.norm(ft-linear)/(np.linalg.norm(ft-f0)+1e-8)),
            'taylor_rmse': float(np.sqrt(np.mean((ft-linear)**2))),
            'logit_change_rmse': float(np.sqrt(np.mean((ft-f0)**2))),
            'hidden_cka': float(np.linalg.norm(u.T@v)**2/max(denominator, 1e-20))}


def predict_mlp(episodes, artifact):
    sizes = [len(ep['query']) for ep in episodes]
    features = np.concatenate([np.concatenate([ep['support'], ep['query']]) for ep in episodes])
    p = forward_batch(features, tuple(artifact[k] for k in PARAMS)).astype(float)
    result, offset = [], 0
    for ep, size in zip(episodes, sizes):
        support, query = p[offset:offset+31], p[offset+31:offset+31+size]
        offset += 31 + size
        residual = np.cumsum(ep['labels'] - support)
        result.append([np.clip(query + (residual[n-1]/(n+2) if n else 0), 0, 1) for n in BUDGETS])
    return result


def predict_reliability(data, artifact):
    g = forward(data['z'], artifact)
    p = np.clip(data['base'].astype(float) + g*data['delta'], 0, 1)
    result = []
    for ep in data['meta']:
        q, start = ep['q'], ep['start']
        result.append([np.full(q, .5), np.full(q, (1+ep['labels'][0])/3)] +
                      [p[start+j*q:start+(j+1)*q] for j in range(4)])
    return result


def score_predictions(episodes, predictions):
    rows = []
    for ep, probs in zip(episodes, predictions):
        counts, successes = np.asarray(ep['counts']), np.asarray(ep['successes'])
        rows.append({'benchmark': ep['benchmark'], 'subject_id': ep['subject_id'], 'seed': ep['seed'],
                     'brier': {'model': [brier(p, counts, successes) for p in probs]},
                     'accuracy': {'model': [accuracy(p, counts, successes) for p in probs]}})
    return rows, aggregate(rows)


def evaluate(family, validation, artifact):
    predictions = predict_mlp(validation, artifact) if family == 'mlp' else predict_reliability(validation, artifact)
    episodes = validation if family == 'mlp' else validation['meta']
    rows, summary = score_predictions(episodes, predictions)
    return rows, summary, predictions


def gradient_audit(model, data):
    """Full-fit gradient at the common initialization, not a minibatch proxy."""
    gradients = []
    x, y = torch.from_numpy(data['x']), torch.from_numpy(data['y'])
    for weight in (data['pair_weight'], data['mixture_weight']):
        model.zero_grad()
        w = torch.from_numpy(weight)
        for start in range(0, len(x), 256):
            sl = slice(start, start+256)
            ((torch.sigmoid(model(x[sl]).flatten())-y[sl]).square()*w[sl]).sum().div(len(x)).backward()
        gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]).detach().numpy().astype(float))
    a, b = gradients
    model.zero_grad()
    return {'pair_norm': float(np.linalg.norm(a)), 'mixture_norm': float(np.linalg.norm(b)),
            'cosine': float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
            'norm_ratio': float(np.linalg.norm(b)/np.linalg.norm(a))}


def train(family, training, validation, out, *, epochs=30, episode_only=False,
          risk='pair', epoch_data=None, output_scale=1.):
    torch.set_num_threads(4)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = {str(p): digest(p.read_bytes()) for p in [Path(__file__),
              Path('mlp/training_mechanism_data.py'), Path('../research_loop/TRAINING_MECHANISM_V5_SPEC.md')]}
    config = {'family': family, 'epochs': epochs, 'episode_only': episode_only, 'risk': risk,
              'resampled': epoch_data is not None, 'output_scale': output_scale, 'sources': source,
              'fit_hash': training['data_hash'] if family == 'mlp' else digest(*(training[k].tobytes() for k in ('z','base','delta','fraction','weight'))),
              'validation_identity': [(e['benchmark'], e['subject_id'], e['seed']) for e in
                                      (validation if family == 'mlp' else validation['meta'])]}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text()) == json.loads(json.dumps(config)), 'Changed fit contract'
        if (out/'complete.json').exists():
            return json.loads((out/'history.json').read_text())
        raise RuntimeError(f'Incomplete fit requires explicit inspection: {out}')
    write_json(out/'config.json', config)
    start = time.monotonic()
    model = network(family, output_scale)
    norm = None if family == 'mlp' else normalizer(training, episode_only)
    if family == 'mlp':
        write_json(out/'risk_audit.json', {**training['weight_audit'], 'gradient': gradient_audit(model, training)})
    initial = export(model, norm)
    probe = (training['x'] if family == 'mlp' else normalize(training['z'], norm))[:128]
    np.savez_compressed(out/'probe.npz', x=probe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001 if family == 'mlp' else .001)
    rng = np.random.default_rng(4307)
    history, steps = [], 0
    fixed_rows = len(training['x'] if family == 'mlp' else training['z'])
    batch = 256 if family == 'mlp' else 1024

    def tensors(data):
        if family == 'mlp':
            return {'x': torch.from_numpy(data['x']), 'y': torch.from_numpy(data['y']),
                    'weight': torch.from_numpy(data['pair_weight'] if risk == 'pair' else data['mixture_weight'])}
        return {'x': torch.from_numpy(normalize(data['z'], norm)),
                **{k: torch.from_numpy(data[k]) for k in ('base', 'delta', 'fraction')},
                'weight': torch.from_numpy(data['weight']*len(data['z'])/data['weight'].sum())}

    reference = tensors(training)
    for epoch in range(epochs+1):
        if time.monotonic()-start > 1800:
            write_json(out/'incomplete.json', {'reason': '1800s_fit_limit', 'last_epoch': epoch-1})
            raise TimeoutError(str(out))
        active = training if epoch_data is None or epoch == 0 else epoch_data(epoch)
        values = reference if active is training else tensors(active)
        assert len(values['x']) == fixed_rows
        grad_norms, losses = [], []
        if epoch:
            order = torch.randperm(fixed_rows) if family == 'mlp' else torch.from_numpy(rng.permutation(fixed_rows))
            for pos in range(0, fixed_rows, batch):
                idx = order[pos:pos+batch]
                output = model(values['x'][idx]).flatten()
                if family == 'mlp':
                    error = (torch.sigmoid(output)-values['y'][idx]).square()
                else:
                    g = 4*torch.sigmoid(output+float(np.log(1/3)))
                    p = torch.clamp(values['base'][idx]+g*values['delta'][idx], 0, 1)
                    f = values['fraction'][idx]
                    error = p*p-2*p*f+f+.01*(g-1).square()
                loss = (values['weight'][idx]*error).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                optimizer.zero_grad()
                loss.backward()
                grad_norms.append(float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))))
                losses.append(float(loss.detach()))
                optimizer.step()
                steps += 1
        artifact = export(model, norm)
        np.savez_compressed(out/f'epoch{epoch}.npz', **artifact)
        with torch.no_grad():
            logits = model(reference['x']).flatten()
            if family == 'mlp':
                p = torch.sigmoid(logits).numpy()
                train_metric = float(np.mean(training['pair_weight']*(p-training['y'])**2))
                expected = p[:32]
                exported = forward_batch(training['x'][:32], tuple(artifact[k] for k in PARAMS))
            else:
                g = 4*torch.sigmoid(logits+float(np.log(1/3)))
                p = torch.clamp(reference['base']+g*reference['delta'], 0, 1).numpy().astype(float)
                f = training['fraction'].astype(float)
                train_metric = float((np.sum(training['weight']*(p*p-2*p*f+f))+training['constant'])/training['pairs'])
                expected = g.numpy()[:32]
                exported = forward(training['z'][:32], artifact)
        export_error = float(np.max(np.abs(expected-exported)))
        assert export_error < 1e-5, export_error
        rows, summary, _ = evaluate(family, validation, artifact)
        write_json(out/f'metrics{epoch}.json', {'rows': rows, 'summary': summary})
        entry = {'epoch': epoch, 'steps': steps, 'train_metric': train_metric,
                 'validation': summary, 'diagnostic': diagnostic(probe, initial, artifact),
                 'export_error': export_error, 'elapsed_seconds': time.monotonic()-start,
                 'optimization_loss': float(np.mean(losses)) if losses else None,
                 'gradient_norm': float(np.mean(grad_norms)) if grad_norms else None,
                 'active_data_hash': digest(*(active[k].tobytes() for k in
                                      (('x','y','groups') if family == 'mlp' else ('z','base','delta','fraction','weight'))))}
        history.append(entry)
        write_json(out/'history.json', history)
        if epoch % 5 == 0:
            print(json.dumps({'fit': str(out), 'epoch': epoch, 'train': train_metric,
                              'val_alc': summary['results']['model']['alc_mean'],
                              'val_b0': summary['results']['model']['brier_mean'][0],
                              'seconds': entry['elapsed_seconds']}), flush=True)
    write_json(out/'complete.json', {'epochs': epochs, 'updates': steps,
               'seconds': time.monotonic()-start, 'checkpoint_count': epochs+1})
    return history


def selections(curve, family):
    first = 1 if family == 'mlp' else 0
    best, wait, stop = first, 0, len(curve)-1
    for epoch in range(first+1, len(curve)):
        if curve[epoch] < curve[best]-1e-5:
            best, wait = epoch, 0
        else:
            wait += 1
        if wait == 5:
            stop = epoch
            break
    best20 = first + int(np.argmin(curve[first:21]))
    best30 = first + int(np.argmin(curve[first:31]))
    recovery = [i for i in range(stop+1, len(curve)-2)
                if max(curve[i:i+3]) <= curve[best]-.001]
    windows = [list(range(i, i+3)) for i in range(1, len(curve)-2)
               if max(curve[i:i+3])-min(curve[i:i+3]) <= .001
               and max(curve[i:i+3]) <= curve[best]+.001]
    return {'patience_epoch': best, 'stop_epoch': stop,
            'best_same_steps': first+int(np.argmin(curve[first:stop+1])),
            'best20': best20, 'best30': best30, 'fixed20': 20, 'fixed30': 30,
            'recovery_starts': recovery, 'stable_window': windows[0] if windows else None}


def trace_analysis(histories, family):
    counts = np.asarray([h[0]['validation']['pairs'] for h in histories])
    key = 'brier_mean' if family == 'mlp' else 'alc_mean'
    curves = [[h['validation']['results']['model'][key][0] if family == 'mlp'
               else h['validation']['results']['model'][key] for h in history] for history in histories]
    curve = np.average(curves, axis=0, weights=counts)
    directions = []
    for history, values in zip(histories, curves):
        selection = selections(values, family)
        early = np.mean([h['train_metric'] for h in history[1:6]])
        late = np.mean([h['train_metric'] for h in history[21:31]])
        no_recovery = not selection['recovery_starts']
        overfit = early-late >= .001 and np.mean(values[21:31])-values[selection['patience_epoch']] >= .001 and no_recovery
        d = [h['diagnostic'] for h in history]
        initial_lazy = np.median([x['taylor_relative_error'] for x in d[1:11]]) <= .1 and np.median([x['hidden_cka'] for x in d[1:11]]) >= .9
        later_change = any(x['taylor_relative_error'] >= .3 or x['hidden_cka'] <= .8 for x in d[15:31])
        later_gain = any(max(values[i:i+3]) <= values[selection['patience_epoch']]-.001 for i in range(15, len(values)-2))
        directions.append({'selection': selection, 'early_train': float(early), 'late_train': float(late),
                           'late_validation': float(np.mean(values[21:31])), 'overfit_no_recovery': bool(overfit),
                           'initial_lazy': bool(initial_lazy), 'later_change': bool(later_change),
                           'later_gain': bool(later_gain), 'initialization_trigger': bool(initial_lazy and later_change and later_gain)})
    return {'curve': curve.tolist(), 'selection': selections(curve.tolist(), family), 'directions': directions,
            'stop_delayed_search': all(d['overfit_no_recovery'] for d in directions)}
