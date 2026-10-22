"""Verify the extracted runtime, external deps and optional project/model inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import site
import subprocess
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, help='nips-comp checkout root; run its tests')
    parser.add_argument('--model-dir', type=Path, help='optional local all-MiniLM-L6-v2 folder')
    parser.add_argument('--checksums', action='store_true', help='verify every packaged file')
    parser.add_argument('--output', type=Path, help='write a JSON verification report')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report = {'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'packages': {}, 'checks': {}}
    try:
        assert sys.version_info[:3] == (3, 12, 7), sys.version
        assert platform.system() == 'Windows' and platform.machine().lower() in {'amd64', 'x86_64'}
        assert Path(sys.executable).resolve().is_relative_to(root / 'runtime')
        assert not site.ENABLE_USER_SITE, 'User site must be disabled by python.ps1'
        external = Path(os.environ['PAIEC_RESEARCH_DEPS']).resolve()
        for filename, location in [('requirements-win-cpu.lock', root / 'runtime'),
                                   ('requirements-external.lock', external)]:
            for line in (root / filename).read_text(encoding='utf-8').splitlines():
                if not line or line.startswith('#'):
                    continue
                name, expected = line.split('==')
                dist = metadata.distribution(name)
                assert dist.version == expected, f'{name}: {dist.version} != {expected}'
                source = Path(dist.locate_file('')).resolve()
                assert source.is_relative_to(location), f'{name} came from {source}'
                report['packages'][name] = {'version': dist.version, 'source': str(source)}
        for name in ['numpy', 'pandas', 'pyarrow', 'scipy', 'sklearn', 'torch',
                     'transformers', 'tokenizers', 'safetensors', 'huggingface_hub',
                     'matplotlib', 'requests', 'pytest']:
            importlib.import_module(name)
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        assert torch.__version__ == '2.9.1+cpu' and torch.version.cuda is None
        torch.set_num_threads(4)
        layer = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(layer.parameters())
        loss = layer(torch.ones(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory(prefix='paiec-env-check-') as tmp:
            folder = Path(tmp)
            frame = pd.DataFrame({'label': [0, 1], 'probability': [0.25, 0.75]})
            frame.to_parquet(folder / 'roundtrip.parquet')
            pd.testing.assert_frame_equal(frame, pd.read_parquet(folder / 'roundtrip.parquet'))
            plt.plot(np.arange(3), [0.25, 0.20, 0.18])
            plt.savefig(folder / 'figure.png')
            plt.close('all')
            assert (folder / 'figure.png').stat().st_size > 0
        report['checks']['native_libraries_cpu_training_parquet_plot'] = 'passed'
        check = subprocess.run([sys.executable, '-s', '-m', 'pip', 'check'],
                               text=True, capture_output=True)
        report['checks']['pip_check'] = check.stdout.strip() + check.stderr.strip()
        assert check.returncode == 0, report['checks']['pip_check']
        if args.checksums:
            manifest = json.loads((root / 'file_manifest.json').read_text(encoding='utf-8'))
            for entry in manifest['files']:
                path = root / entry['path']
                assert path.resolve().is_relative_to(root), entry['path']
                assert path.stat().st_size == entry['bytes'], entry['path']
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                assert digest == entry['sha256'], entry['path']
            report['checks']['checksums'] = len(manifest['files'])
        if args.model_dir:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                                       clean_up_tokenization_spaces=False)
            model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).eval()
            with torch.no_grad():
                output = model(**tokenizer(['PAIEC offline migration check'], return_tensors='pt'))
            assert output.last_hidden_state.shape[-1] == 384
            assert torch.isfinite(output.last_hidden_state).all()
            report['checks']['offline_minilm'] = {'path': str(args.model_dir), 'dimensions': 384}
        if args.project:
            assert shutil.which('git'), 'Git for Windows is required by the repository packaging tests'
            project = args.project.resolve()
            run = subprocess.run([sys.executable, '-s', '-m', 'pytest', '-q'], cwd=project,
                                 text=True, capture_output=True)
            report['checks']['project_tests'] = run.stdout.strip() + run.stderr.strip()
            assert run.returncode == 0, report['checks']['project_tests']
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
