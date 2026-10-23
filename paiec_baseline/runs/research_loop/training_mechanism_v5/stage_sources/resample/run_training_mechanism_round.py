"""Execute plan_2 source pilots sequentially; no outer-test model selection."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from mlp.cohort_semantic import CachedSemantic
from mlp.training_mechanism_data import (write_json, mlp_training, mlp_validation,
    reliability_fixed, concatenate, load_npz, digest)
from mlp.training_mechanism_training import train, trace_analysis

ROOT = Path('runs/research_loop/training_mechanism_v5')


def info(fold):
    source = json.loads(Path(f'runs/research_loop/reproduction_v4/source_fold{fold}.json').read_text())
    assert set(source['inner_halves'][0]).isdisjoint(source['inner_halves'][1])
    assert set(sum(source['inner_halves'], [])).isdisjoint(source['heldout_benchmarks'])
    return source


def combined_metrics(folders, epoch):
    from mlp.reproduction_episodes import aggregate
    rows = []
    for folder in folders:
        rows.extend(json.loads((folder/f'metrics{epoch}.json').read_text())['rows'])
    return {'rows': rows, 'summary': aggregate(rows)}


def trajectory():
    root = ROOT/'R54'
    engine = CachedSemantic()
    for fold in range(3):
        source = info(fold)
        write_json(root/f'source_fold{fold}.json', source)
        halves = source['inner_halves']
        names = halves[0]+halves[1]
        # One model family in memory, one fit running at a time.
        for family in ('mlp', 'query', 'episode'):
            analyses = root/f'fold{fold}_{family}_analysis.json'
            if analyses.exists():
                continue
            print(json.dumps({'stage': 'prepare', 'fold': fold, 'family': family}), flush=True)
            if family == 'mlp':
                data = {name: mlp_validation(name, engine) for name in names}
            else:
                data = {name: reliability_fixed(name, engine, source['semantic_ridge']) for name in names}
            histories, folders = [], []
            for inner in range(2):
                folder = root/f'fold{fold}_{family}_inner{inner}'
                if family == 'mlp':
                    training = mlp_training(halves[inner])
                    validation = [ep for name in halves[1-inner] for ep in data[name]]
                else:
                    training = concatenate([data[name] for name in halves[inner]])
                    validation = concatenate([data[name] for name in halves[1-inner]])
                histories.append(train(family, training, validation, folder, episode_only=family == 'episode'))
                folders.append(folder)
                del training, validation
                gc.collect()
            analysis = trace_analysis(histories, family)
            analysis['scores'] = {}
            for rule, epoch in analysis['selection'].items():
                if isinstance(epoch, int) and rule != 'stop_epoch':
                    values = combined_metrics(folders, epoch)
                    write_json(root/f'fold{fold}_{family}_{rule}.json', values)
                    analysis['scores'][rule] = values['summary']
            write_json(analyses, analysis)
            print(json.dumps({'stage': 'trajectory_complete', 'fold': fold, 'family': family,
                              'selection': analysis['selection'], 'stop_delayed_search': analysis['stop_delayed_search']}), flush=True)
            del data, histories
            gc.collect()
    all_analyses = {family: [json.loads((root/f'fold{fold}_{family}_analysis.json').read_text())
                            for fold in range(3)] for family in ('mlp', 'query', 'episode')}
    write_json(root/'summary.json', all_analyses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['trajectory','resample','benchmark_risk','averages','initialization'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    from mlp.training_mechanism_guard import execute
    with threadpool_limits(limits=4):
        if args.stage == 'trajectory':
            execute(args.stage, trajectory)
        else:
            from mlp import training_mechanism_stages
            execute(args.stage, getattr(training_mechanism_stages,args.stage))


if __name__ == '__main__':
    main()
