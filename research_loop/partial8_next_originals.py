"""Fetch exact original URLs discovered through AnySearch, never search snippets."""
import requests,hashlib,json,subprocess
from pathlib import Path
OUT=Path(__file__).parent/'literature/partial8_next'
def main():
    manifest=[]
    for name,url in [('grandvalet04a.pdf','https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf'),
        ('sklearn_1_5_2_split.py','https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/sklearn/model_selection/_split.py'),
        ('MALSAR_Least_SRMTL.m','https://raw.githubusercontent.com/jiayuzhou/MALSAR/fb9751594983df020ddc4f7e4a40520ee7c37989/MALSAR/functions/SRMTL/Least_SRMTL.m')]:
        r=requests.get(url,timeout=60);r.raise_for_status();data=r.content
        if name.endswith('.pdf'):assert data.startswith(b'%PDF')
        p=OUT/name;p.write_bytes(data);manifest.append({'url':url,'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});print(name,len(data),flush=True)
    (OUT/'original_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    subprocess.run(['D:/texlive/2025/bin/windows/pdftotext.exe','-layout',str(OUT/'grandvalet04a.pdf'),str(OUT/'grandvalet04a.txt')],check=True)
if __name__=='__main__':main()
