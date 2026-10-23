"""Execution provenance and resource-state guard, independent of fit arithmetic."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .training_mechanism_data import CACHE, write_json

ROOT = Path('runs/research_loop/training_mechanism_v5')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze_core():
    # Capture all historical Python dependencies, not only the three edited modules.
    files = [p for p in Path('mlp').rglob('*.py') if 'training_mechanism' not in p.name]
    files += [Path('mlp/training_mechanism_data.py'),Path('mlp/training_mechanism_training.py'),
              Path('../research_loop/TRAINING_MECHANISM_V5_SPEC.md'),Path('../doc/plan_2.md')]
    files += list(Path('runs/research_loop/reproduction_v4').glob('source_fold*.json'))
    files += list(Path('runs/research_loop/R43').glob('fold*_selection.json'))
    files += list(Path('runs/research_loop/R25').glob('fold*_selection.json'))
    hashes = {str(p):sha(p) for p in sorted(set(files))}
    path = ROOT/'core_provenance.json'
    if path.exists():
        assert json.loads(path.read_text())['sha256'] == hashes, 'Frozen dependency or split changed'
    else:
        root = Path.cwd().parent
        for p in files:
            relative = p.resolve().relative_to(root)
            target = ROOT/'source_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(p.read_bytes())
        write_json(path, {'sha256':hashes,'captured_utc':datetime.now(timezone.utc).isoformat(),
            'timing_note':'R54 completed before this dependency-wide snapshot; fit-level hashes and reviewer hashes predate it. No core edits occurred during R54.',
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'python':sys.version,'platform':platform.platform()})


def cache_inventory():
    path = ROOT/'cache_inventory.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    for p,expected in previous.items():
        assert sha(p) == expected, f'Cached input mutated: {p}'
    inputs = list(CACHE.rglob('*.npz'))
    inputs += list(Path('D:/paiec_research_cache/evidence_reliability').glob('*.npz'))
    inputs += list(Path('D:/paiec_research_cache/reproduction_v4/source').glob('*.npz'))
    current = {str(p):sha(p) for p in inputs}
    # Validation metadata/labels/counts and feature arrays are all covered by NPZ bytes.
    write_json(path, current)


def audit_completions():
    audits = []
    for path in ROOT.glob('R*/fold*/complete.json'):
        complete = json.loads(path.read_text())
        history = json.loads((path.parent/'history.json').read_text())
        valid = complete['seconds'] <= 1800 and len(history) == complete['epochs']+1
        if not valid:
            write_json(path.parent/'incomplete.json', {'reason':'post_fit_resource_or_history_audit','complete_record':complete})
            raise RuntimeError(f'Fit invalid: {path.parent}')
        audits.append({'fit':str(path.parent),'seconds':complete['seconds'],'epochs':complete['epochs'],
                       'updates':complete['updates'],'checkpoint_count':len(list(path.parent.glob('epoch*.npz')))})
    write_json(ROOT/'completion_audit.json',audits)


def execute(stage, callback):
    freeze_core()
    cache_inventory()
    stage_files = [Path('mlp/run/run_training_mechanism_round.py'),Path('mlp/training_mechanism_stages.py'),Path(__file__)]
    record = {'stage':stage,'start_utc':datetime.now(timezone.utc).isoformat(),
              'source_hashes':{str(p):sha(p) for p in stage_files if p.exists()},
              'command':f'powershell -NoProfile -ExecutionPolicy Bypass -File D:/nips-comp/paiec-runtime-win64-py312-cpu/python.ps1 -m mlp.run.run_training_mechanism_round {stage}'}
    contract_path = ROOT/'stage_contracts'/f'{stage}.json'
    if contract_path.exists():
        expected = json.loads(contract_path.read_text())
        assert expected['source_hashes'] == record['source_hashes'], 'Stage implementation changed before reuse'
    else:
        write_json(contract_path, {'source_hashes':record['source_hashes'],
                                  'frozen_utc':record['start_utc']})
    for p in stage_files:
        if p.exists():
            target = ROOT/'stage_sources'/stage/p.name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(p.read_bytes())
    try:
        callback()
        assert {str(p):sha(p) for p in stage_files if p.exists()} == record['source_hashes'], 'Stage changed during execution'
        audit_completions()
        freeze_core()
        cache_inventory()
        record['status'] = 'complete'
    except Exception as error:
        record.update(status='incomplete',error_type=type(error).__name__,error=str(error))
        for config in ROOT.glob('R*/fold*/config.json'):
            if not (config.parent/'complete.json').exists():
                write_json(config.parent/'incomplete.json',{'reason':'stage_exception','error_type':type(error).__name__,'error':str(error)})
        raise
    finally:
        record['end_utc'] = datetime.now(timezone.utc).isoformat()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write_json(ROOT/'execution_attempts'/f'{stage}_{stamp}.json',record)
        write_json(ROOT/f'{stage}_execution.json',record)
