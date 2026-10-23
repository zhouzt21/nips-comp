from historical_positive_common import *
import requests
root=ROOT/'R69'
urls={
 'pava_original.pdf':'https://scholarlypublications.universiteitleiden.nl/access/item%3A3515641/view',
 'author_monotone.c':'https://raw.githubusercontent.com/cran/monotone/master/src/monotone.c',
 'author_DESCRIPTION':'https://raw.githubusercontent.com/cran/monotone/master/DESCRIPTION',
 'scipy_pava.h':'https://raw.githubusercontent.com/scipy/scipy/v1.13.1/scipy/optimize/_pava/pava.h'}
records=[]
for name,url in urls.items():
    try:
        r=requests.get(url,timeout=30);r.raise_for_status();p=root/name
        if name.endswith('.pdf'):assert r.content.startswith(b'%PDF')
        p.write_bytes(r.content);records.append({'url':url,'path':str(p),'sha256':sha(p),'bytes':len(r.content)})
        if name.endswith('.pdf'):
            import fitz
            doc=fitz.open(p);text='\n'.join(f'PAGE {i+1}\n'+page.get_text() for i,page in enumerate(doc))
            (root/'pava_original.txt').write_text(text,encoding='utf-8')
    except Exception as e:records.append({'url':url,'error':str(e)[:300]})
write_json(root/'pava_retrieval.json',records);print(json.dumps(records))
