"""Fetch only the two primary references and the linked author metric source."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
import requests

ROOT = Path('D:/nips-comp/paiec-dataset/low_budget_prior_sources')
ROOT.mkdir(parents=True, exist_ok=True)
URLS = {
    'gneiting_raftery_2007.pdf': 'https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf',
    'ovadia_2019.pdf': 'https://papers.nips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf',
}
records = []
for name, url in URLS.items():
    p = ROOT/name
    if not p.exists():
        r = requests.get(url, timeout=60); r.raise_for_status()
        assert r.content.startswith(b'%PDF'), name
        p.write_bytes(r.content)
    data = p.read_bytes()
    records.append({'url': url, 'path': str(p), 'sha256': hashlib.sha256(data).hexdigest()})
    pdftotext = Path('D:/texlive/2025/bin/windows/pdftotext.exe')
    if pdftotext.exists():
        subprocess.run([str(pdftotext), '-layout', str(p), str(ROOT/(name+'.txt'))], check=True)
    elif importlib.util.find_spec('pypdf'):
        from pypdf import PdfReader
        reader = PdfReader(p)
        (ROOT/(name+'.txt')).write_text('\n'.join(f'\n## PDF PAGE {i+1}\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages)), encoding='utf-8')
    elif importlib.util.find_spec('fitz'):
        import fitz
        doc = fitz.open(p)
        (ROOT/(name+'.txt')).write_text('\n'.join(f'\n## PDF PAGE {i+1}\n'+page.get_text() for i,page in enumerate(doc)), encoding='utf-8')
    else:
        raise RuntimeError('No existing PDF text reader')
(ROOT/'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
commit_file = ROOT/'author_repository.json'
if commit_file.exists():
    commit = json.loads(commit_file.read_text())['commit']
else:
    r = requests.get('https://api.github.com/repos/google-research/google-research/commits',
                     params={'path': 'uq_benchmark_2019', 'per_page': 1}, timeout=60)
    r.raise_for_status(); commit = r.json()[0]['sha']
sources = []
for file in ['README.md', 'metrics_lib.py', 'calibration_lib.py']:
    url = f'https://raw.githubusercontent.com/google-research/google-research/{commit}/uq_benchmark_2019/{file}'
    p = ROOT/'uq_benchmark_2019'/file
    if not p.exists():
        r = requests.get(url, timeout=60); r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(r.content)
    sources.append({'url': url, 'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
commit_file.write_text(json.dumps({'commit': commit, 'sources': sources}, indent=2), encoding='utf-8')
print('Author repository commit: '+commit)
print(json.dumps(records))
