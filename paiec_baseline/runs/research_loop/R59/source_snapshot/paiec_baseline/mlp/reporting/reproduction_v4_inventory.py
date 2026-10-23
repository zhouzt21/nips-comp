"""Capture runtime metadata and hashes of locally retained reproduction weights."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[3]
    runs = root/'paiec_baseline/runs/research_loop'
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'transformers',
             'sentence-transformers', 'huggingface-hub', 'pyarrow', 'matplotlib']
    packages = {}
    for name in names:
        try: packages[name] = version(name)
        except Exception: packages[name] = None
    artifacts = {}
    for round_name in ('R51', 'R52'):
        for path in sorted((runs/round_name).rglob('*.pt')):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
            artifacts[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    runtime = Path('D:/nips-comp/paiec-runtime-win64-py312-cpu')
    evidence = {
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'code_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'python': sys.version, 'executable': sys.executable,
        'os': {'system': platform.system(), 'release': platform.release(), 'version': platform.version(),
               'machine': platform.machine()},
        'packages': packages, 'hardware': 'AMD Ryzen 7 4800H, 8 cores / 16 threads; 16 GiB RAM; CPU only',
        'training_threads': 4, 'concurrent_training_processes': 1,
        'runtime_lock_hashes': {name: hashlib.sha256((runtime/name).read_bytes()).hexdigest()
                               for name in ('requirements-win-cpu.lock', 'requirements-external.lock', 'python.ps1')},
        'weights_are_local_git_ignored': True, 'weights': artifacts,
        'data_revision': '54074453da7b9911be0f48e4c560be547bc84d21',
        'embedding_revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
    }
    out = runs/'reproduction_v4/execution_inventory.json'
    out.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({'weights_hashed': len(artifacts), 'output': str(out)}))


if __name__ == '__main__': main()
