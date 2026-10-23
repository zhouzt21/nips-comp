"""Fit-only transforms, paired metrics and bounded CPU neural-process training."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .evaluate import ALC_WEIGHTS, BUDGETS, _stable_seed
from .neural_process import CNP, AttentiveCNP
from .reproduction_episodes import accuracy, aggregate, brier


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fit_normalizer(episodes):
    """Only called on fit benchmarks; give each represented item one observation."""
    count = 0; total = np.zeros(640); squares = np.zeros(640)
    for episode in episodes:
        for field in ('support', 'query'):
            x = episode[field].astype(np.float64)
            count += len(x); total += x.sum(0); squares += (x*x).sum(0)
    mean = total/count
    scale = np.maximum(np.sqrt(np.maximum(squares/count-mean*mean, 0)), .05)
    return {'mean': mean.astype(np.float32), 'scale': scale.astype(np.float32)}


def transform(x, norm):
    return ((x-norm['mean'])/norm['scale']).astype(np.float32)


def tensor_episodes(episodes, norm):
    size = len(episodes); queries = max(len(ep['query']) for ep in episodes)
    result = {key: np.zeros(shape, np.float32) for key, shape in {
        'support': (size, 31, 640), 'query': (size, queries, 640),
        'labels': (size, 31), 'weights': (size, queries), 'rates': (size, queries),
        'subject': (size, 256)}.items()}
    for i, episode in enumerate(episodes):
        q = len(episode['query']); counts = episode['counts']
        result['support'][i] = transform(episode['support'], norm)
        result['query'][i, :q] = transform(episode['query'], norm)
        result['labels'][i] = episode['labels']
        result['weights'][i, :q] = counts/counts.sum()
        result['rates'][i, :q] = episode['successes']/counts
        result['subject'][i] = episode['subject']
    return {key: torch.from_numpy(value) for key, value in result.items()}


def make_model(name):
    if name == 'cnp': return CNP()
    if name == 'attentive_cnp': return AttentiveCNP()
    if name in ('irt_calibration', 'irt_joint'):
        from .amortized_irt import AmortizedIRT
        return AmortizedIRT(difficulty_bias=(name == 'irt_calibration'))
    raise ValueError(name)


def logits(model, name, tensors, index, budget, labels=None):
    context = tensors['support'][index, :budget]
    context_labels = tensors['labels'][index, :budget] if labels is None else labels
    args = (context, context_labels, tensors['query'][index])
    if name.startswith('irt_'):
        return model.predict_logits(*args, tensors['subject'][index])
    return model.predict_logits(*args)


@torch.no_grad()
def validation_curve(model, name, tensors, batch_size=16):
    model.eval(); size = len(tensors['labels']); scores = np.zeros(6)
    for start in range(0, size, batch_size):
        index = slice(start, min(start+batch_size, size))
        w, y = tensors['weights'][index], tensors['rates'][index]
        for j, budget in enumerate(BUDGETS):
            p = torch.sigmoid(logits(model, name, tensors, index, budget))
            scores[j] += float(((p*p-2*p*y+y)*w).sum())
    return scores/size


def save_checkpoint(path, model, name, norm, epoch, metadata):
    torch.save({'model_name': name, 'state_dict': model.state_dict(),
                'normalization': {k: torch.from_numpy(v) for k, v in norm.items()},
                'epoch': epoch, 'metadata': metadata}, path)


def load_checkpoint(path):
    artifact = torch.load(path, weights_only=True, map_location='cpu')
    model = make_model(artifact['model_name'])
    model.load_state_dict(artifact['state_dict']); model.eval()
    norm = {key: value.numpy() for key, value in artifact['normalization'].items()}
    return model, norm, artifact


def train_neural_process(training, validation, name, folder, *, epochs=60,
                         batch_size=8, seed=5107, threads=4, source_fingerprint=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    history_path = folder/'history.json'
    if history_path.is_file():
        history = json.loads(history_path.read_text())
        existing = json.loads((folder/'fit_contract.json').read_text())
        expected = {'seed': seed, 'epochs': epochs, 'batch_size': batch_size, 'threads': threads,
                    'model_name': name, 'source_fingerprint': source_fingerprint,
                    'fit_benchmarks': sorted({ep['benchmark'] for ep in training}),
                    'validation_benchmarks': sorted({ep['benchmark'] for ep in validation})}
        if any(existing.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f'Fit contract changed at {folder}')
        if (folder/'resource_stop.json').exists():
            raise RuntimeError(f'Resource-incomplete fit at {folder}')
        if len(history) == epochs+1 and (folder/f'epoch_{epochs:03d}.pt').is_file():
            return history
        raise RuntimeError(f'Incomplete fit at {folder}; use an explicit recovery directory')
    torch.set_num_threads(threads); torch.manual_seed(seed)
    norm = fit_normalizer(training)
    train = tensor_episodes(training, norm); valid = tensor_episodes(validation, norm)
    model = make_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(seed)
    metadata = {'seed': seed, 'model_name': name, 'source_fingerprint': source_fingerprint,
                'fit_benchmarks': sorted({ep['benchmark'] for ep in training}),
                'validation_benchmarks': sorted({ep['benchmark'] for ep in validation}),
                'training_pairs': len(training), 'validation_pairs': len(validation),
                'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .0001,
                'batch_size': batch_size, 'epochs': epochs, 'threads': threads,
                'objective': 'pair-equal six-budget-weighted Bernoulli NLL on disjoint query items',
                'parameters': sum(p.numel() for p in model.parameters())}
    assert set(metadata['fit_benchmarks']).isdisjoint(metadata['validation_benchmarks'])
    write_json(folder/'fit_contract.json', metadata)
    history = []; start_time = time.time()
    for epoch in range(epochs+1):
        if time.time()-start_time > 1800:
            write_json(folder/'resource_stop.json', {'status': 'incomplete', 'last_completed_epoch': epoch-1,
                       'reason': 'predeclared 1800-second per-fit CPU limit', 'elapsed_seconds': time.time()-start_time})
            raise RuntimeError(f'Predeclared resource limit reached at {folder}; not a completed negative result')
        train_loss = None
        if epoch:
            model.train(); accumulated = 0.
            order = rng.permutation(len(training))
            for offset in range(0, len(order), batch_size):
                index = order[offset:offset+batch_size]
                optimizer.zero_grad(set_to_none=True)
                for weight, budget in zip(ALC_WEIGHTS, BUDGETS):
                    output = logits(model, name, train, index, budget)
                    loss = (F.binary_cross_entropy_with_logits(output, train['rates'][index], reduction='none')
                            * train['weights'][index]).sum(-1).mean() * float(weight)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                    accumulated += float(loss.detach()) * len(index)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            train_loss = accumulated/len(training)
        curve = validation_curve(model, name, valid)
        history.append({'epoch': epoch, 'training_nll': train_loss,
                        'validation_brier': curve.tolist(), 'validation_alc': float(curve@ALC_WEIGHTS),
                        'elapsed_seconds': time.time()-start_time})
        save_checkpoint(folder/f'epoch_{epoch:03d}.pt', model, name, norm, epoch, metadata)
        write_json(history_path, history)
        if time.time()-start_time > 1800:
            write_json(folder/'resource_stop.json', {'status': 'incomplete', 'last_completed_epoch': epoch,
                       'reason': 'predeclared 1800-second per-fit CPU limit', 'elapsed_seconds': time.time()-start_time})
            raise RuntimeError(f'Predeclared resource limit reached at {folder}; not a completed negative result')
        if epoch % 10 == 0 or epoch == epochs:
            print(json.dumps({'fit': str(folder), **history[-1]}), flush=True)
    return history


def decomposition(probabilities, counts, successes):
    p = np.asarray(probabilities, float); m = counts.sum()
    p_mean = float(counts@p/m); y_mean = float(successes.sum()/m)
    variance_p = float(counts@((p-p_mean)**2)/m)
    covariance = float(successes@p/m-p_mean*y_mean)
    values = {'outcome_variance': y_mean*(1-y_mean), 'mean_bias_squared': (p_mean-y_mean)**2,
              'prediction_variance': variance_p, 'minus_twice_covariance': -2*covariance}
    assert abs(sum(values.values())-brier(p, counts, successes)) < 1e-10
    return values


@torch.no_grad()
def evaluate_checkpoint(path, episodes, *, permutation_repeats=4):
    model, norm, artifact = load_checkpoint(path); name = artifact['model_name']
    tensors = tensor_episodes(episodes, norm); rows = []
    for i, ep in enumerate(episodes):
        true_brier, true_acc, perm_brier, perm_acc = [], [], [], []
        details = []
        size = len(ep['query']); index = slice(i, i+1)
        for budget in BUDGETS:
            p = torch.sigmoid(logits(model, name, tensors, index, budget))[0, :size].numpy().astype(float)
            true_brier.append(brier(p, ep['counts'], ep['successes']))
            true_acc.append(accuracy(p, ep['counts'], ep['successes']))
            shuffled_losses, shuffled_accs, changes = [], [], []
            for repeat in range(permutation_repeats):
                seed = _stable_seed(5113+repeat, ep['benchmark'], f"{ep['subject_id']}|{ep['seed']}|{budget}")
                perm = np.random.default_rng(seed).permutation(budget)
                y = tensors['labels'][index, :budget][:, perm]
                shuffled = torch.sigmoid(logits(model, name, tensors, index, budget, labels=y))[0, :size].numpy().astype(float)
                shuffled_losses.append(brier(shuffled, ep['counts'], ep['successes']))
                shuffled_accs.append(accuracy(shuffled, ep['counts'], ep['successes']))
                changes.append(float(ep['counts']@np.abs(p-shuffled)/ep['counts'].sum()))
            perm_brier.append(float(np.mean(shuffled_losses)))
            perm_acc.append(float(np.mean(shuffled_accs)))
            details.append({**decomposition(p, ep['counts'], ep['successes']),
                            'permutation_mean_absolute_change': float(np.mean(changes))})
        rows.append({'benchmark': ep['benchmark'], 'subject_id': ep['subject_id'], 'seed': ep['seed'],
                     'brier': {name: true_brier, name+'_permuted': perm_brier},
                     'accuracy': {name: true_acc, name+'_permuted': perm_acc}, 'decomposition': details})
    return rows


def merge_rows(baselines, candidate):
    lookup = {(row['benchmark'], row['subject_id'], row['seed']): row for row in candidate}
    result = []
    for row in baselines:
        match = lookup[(row['benchmark'], row['subject_id'], row['seed'])]
        result.append({**row, 'brier': {**row['brier'], **match['brier']},
                       'accuracy': {**row['accuracy'], **match['accuracy']},
                       'decomposition': match.get('decomposition')})
    assert len(result) == len(candidate)
    return result
