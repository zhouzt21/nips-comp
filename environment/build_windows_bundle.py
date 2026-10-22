"""Build the Windows x64 CPU runtime; never copy the external research deps.

Run with CPython 3.12 on Windows. Large files are written to --work, not the repo.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile


SOURCE = Path(__file__).resolve().parent
NAME = 'paiec-runtime-win64-py312-cpu'
PYTHON_URL = 'https://api.nuget.org/v3-flatcontainer/python/3.12.7/python.3.12.7.nupkg'
PYTHON_SHA256 = '149dd298e0b7a82250ca019471770fff079874088a4e8501ca20922d7df3a6ac'
VC_PACKAGE = 'vs2015_runtime-14.40.33807-h98bb1dd_1.tar.bz2'
VC_URL = 'https://repo.anaconda.com/pkgs/main/win-64/' + VC_PACKAGE
VC_SHA256 = 'd82b5b2b1a9a1b37da109039274c125355ce478a199c07c7826ac0fdc2178491'
EXTERNAL = {'huggingface_hub', 'transformers', 'tokenizers', 'safetensors'}


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def packaged_files(root: Path):
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if '__pycache__' in relative.parts or path.suffix in {'.pyc', '.pyo'}:
            continue
        if relative.parts[:2] == ('runtime', 'Scripts'):
            continue  # pip-generated exe entry points contain build-time paths.
        if relative.parts[0] == 'cache' or relative.as_posix() == 'file_manifest.json':
            continue
        yield path


def install_vc_runtime(work: Path, bundle: Path) -> dict:
    """Match the working source environment's C++ DLLs, including PyArrow's copy."""
    package = work / VC_PACKAGE
    if not package.exists():
        urllib.request.urlretrieve(VC_URL, package)
    assert digest(package) == VC_SHA256, 'C++ runtime package checksum mismatch'
    runtime = bundle / 'runtime'
    names = ['msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
             'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll', 'concrt140.dll',
             'vccorlib140.dll', 'vcomp140.dll', 'vcruntime140.dll',
             'vcruntime140_1.dll', 'vcruntime140_threads.dll']
    with tarfile.open(package, 'r:bz2') as archive:
        for name in names:
            (runtime / name).write_bytes(archive.extractfile(name).read())
        licenses = bundle / 'licenses'
        licenses.mkdir(exist_ok=True)
        (licenses / 'Anaconda-runtime-recipe-LICENSE.txt').write_bytes(
            archive.extractfile('info/recipe/parent/LICENSE').read())
    # Arrow 16.1's wheel bundles MSVCP 14.28; Torch 2.9 needs >=14.40.
    # Merely placing the newer DLL next to python.exe does not fix Arrow-first imports.
    target = runtime / 'Lib/site-packages/pyarrow/msvcp140.dll'
    wheel = next((work / 'wheelhouse').glob('pyarrow-16.1.0-*.whl'))
    with zipfile.ZipFile(wheel) as archive:
        original_digest = hashlib.sha256(archive.read('pyarrow/msvcp140.dll')).hexdigest()
    shutil.copyfile(runtime / 'msvcp140.dll', target)
    record = runtime / 'Lib/site-packages/pyarrow-16.1.0.dist-info/RECORD'
    with record.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        if row[0] == 'pyarrow/msvcp140.dll':
            row[1] = 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(digest(target))).decode().rstrip('=')
            row[2] = str(target.stat().st_size)
    with record.open('w', newline='', encoding='utf-8') as handle:
        csv.writer(handle).writerows(rows)
    return {'source': VC_URL, 'sha256': VC_SHA256, 'dll_version': '14.40.33810.0',
            'dlls': names, 'matches_original_anaconda_runtime': True,
            'pyarrow_override': {'path': 'pyarrow/msvcp140.dll',
                                 'original_sha256': original_digest,
                                 'replacement_sha256': digest(target)},
            'reason': 'Avoid Arrow 16.1 MSVCP 14.28 overriding the runtime required by Torch 2.9'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    args = parser.parse_args()
    assert sys.platform == 'win32' and sys.version_info[:2] == (3, 12)
    assert platform.machine().lower() in {'amd64', 'x86_64'}
    work = args.work.resolve()
    assert not work.is_relative_to(SOURCE.parent), 'Put build artifacts outside the Git workspace'
    work.mkdir(parents=True, exist_ok=True)
    base_lock = (SOURCE / 'requirements-win-cpu.lock').read_text(encoding='utf-8')
    lines = [line for line in base_lock.splitlines() if line and not line.startswith('#')]
    assert not any(line.split('==')[0].replace('-', '_') in EXTERNAL for line in lines)
    pypi_lock = work / 'requirements-pypi.lock'
    pypi_lock.write_text('\n'.join(line for line in lines if not line.startswith('torch==')) + '\n',
                          encoding='utf-8')
    commands = [
        ([sys.executable, '-m', 'pip', '--isolated', 'download', '--only-binary=:all:',
          '--no-deps', '--index-url', 'https://pypi.org/simple', '-r', str(pypi_lock),
          '--dest', str(work / 'wheelhouse'), '--disable-pip-version-check'], 'download-pypi.log'),
        ([sys.executable, '-m', 'pip', '--isolated', 'download', '--only-binary=:all:',
          '--no-deps', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.9.1+cpu',
          '--dest', str(work / 'torch-wheel'), '--disable-pip-version-check'], 'download-torch.log'),
    ]
    for command, filename in commands:
        print(filename, flush=True)
        with (work / filename).open('w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    package = work / 'python.3.12.7.nupkg'
    if not package.exists():
        urllib.request.urlretrieve(PYTHON_URL, package)
    assert digest(package) == PYTHON_SHA256, 'Python package checksum mismatch'
    bundle = work / NAME
    runtime = bundle / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    if not (runtime / 'python.exe').exists():
        with zipfile.ZipFile(package) as archive:
            for entry in archive.infolist():
                name = PurePosixPath(entry.filename)
                if entry.is_dir() or not name.parts or name.parts[0] != 'tools':
                    continue
                target = runtime.joinpath(*name.parts[1:]).resolve()
                assert target.is_relative_to(runtime), entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    print('Installing wheels into the isolated runtime', flush=True)
    with (work / 'install.log').open('w', encoding='utf-8') as log:
        subprocess.run([str(runtime / 'python.exe'), '-I', '-m', 'pip', '--isolated', 'install',
                        '--no-index', '--no-deps', '--no-compile',
                        '--find-links', str(work / 'wheelhouse'),
                        '--find-links', str(work / 'torch-wheel'),
                        '-r', str(SOURCE / 'requirements-win-cpu.lock'),
                        '--disable-pip-version-check'], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    vc_runtime = install_vc_runtime(work, bundle)
    for filename in ['python.ps1', 'verify_environment.py', 'requirements-win-cpu.lock',
                     'requirements-external.lock', 'source_environment.json', 'README.md']:
        shutil.copyfile(SOURCE / filename, bundle / filename)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    provenance = {
        'name': NAME, 'source_code_commit': commit,
        'interpreter': {'version': '3.12.7', 'source': PYTHON_URL, 'sha256': digest(package)},
        'dependency_sources': ['https://pypi.org/simple', 'https://download.pytorch.org/whl/cpu'],
        'windows_cpp_runtime': vc_runtime,
        'external_dependency_policy': 'Already copied by user. Not downloaded, copied or bundled.',
        'version_changes': [{'name': 'kiwisolver', 'source': '1.4.4', 'bundle': '1.4.5',
                             'reason': 'No CPython 3.12 Windows wheel for the original Conda build'}],
        'wheels': [{'filename': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
                   for directory in ['wheelhouse', 'torch-wheel']
                   for p in sorted((work / directory).glob('*.whl'))],
    }
    (bundle / 'bundle_metadata.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    files = list(packaged_files(bundle))
    for path in files:
        if 'site-packages' in path.parts:
            name = path.relative_to(runtime / 'Lib' / 'site-packages').parts[0]
            assert not any(name == p or name.startswith(p + '-') for p in EXTERNAL), name
    manifest = {'algorithm': 'sha256', 'files': [
        {'path': p.relative_to(bundle).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
        for p in files]}
    manifest_path = bundle / 'file_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    output = work / (NAME + '.zip')
    print('Writing archive', flush=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, NAME + '/' + path.relative_to(bundle).as_posix())
    (work / (output.name + '.sha256')).write_text(f'{digest(output)}  {output.name}\n', encoding='ascii')
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size,
                      'files': len(files) + 1, 'uncompressed_bytes': sum(p.stat().st_size for p in files)}, indent=2))


if __name__ == '__main__':
    main()
