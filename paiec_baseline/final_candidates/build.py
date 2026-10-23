"""Deterministic packages from frozen source bodies and locally pinned model files."""
import ast,shutil,zipfile,sys
from common import *
WEIGHTS=Path('D:/paiec_pretrained/all-MiniLM-L6-v2')
NAMES={'r25_nopeer':'paiec_r25_nopeer_final_candidate','partial8_shared':'paiec_partial8_shared_final_candidate'}
MODEL_FILES=['config.json','model.safetensors','modules.json','README.md','sentence_bert_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt']
def extract(path,names):
    source=path.read_text(encoding='utf-8');tree=ast.parse(source);parts=[]
    for name in names:
        node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        parts.append(ast.get_source_segment(source,node))
    return '\n\n'.join(parts)+'\n'
def build():
    results={};src=REPO/'research_loop';dist=BASE/'dist';dist.mkdir(exist_ok=True)
    for kind,name in NAMES.items():
        folder=HERE/'packages'/name;pkg=folder/'candidate';weights=pkg/'weights';weights.mkdir(parents=True,exist_ok=True)
        (pkg/'__init__.py').write_text('',encoding='utf-8')
        (pkg/'core.py').write_text('import numpy as np\n\n'+extract(src/'shared_slope_model.py',['transform','centered_arrays','predict']),encoding='utf-8')
        adapter=(src/'shared_slope_adapter.py').read_text(encoding='utf-8').split('\ndef make_r25_encoder')[0].replace('from shared_slope_model import predict','from .core import predict')
        (pkg/'adapter.py').write_text(adapter,encoding='utf-8')
        wrapper=(src/'shared_slope_adapter_cpu.py').read_text(encoding='utf-8').split('\ndef make_r25_encoder')[0].replace('from shared_slope_adapter import','from .adapter import')
        (pkg/'adapter_cpu.py').write_text(wrapper,encoding='utf-8')
        for filename in ['structured_text.py','metadata_features.py']:shutil.copyfile(BASE/'mlp'/filename,pkg/filename)
        semantic=(BASE/'mlp/semantic_features.py').read_text(encoding='utf-8');tree=ast.parse(semantic)
        cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SemanticFeatures');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='canonical')
        canonical=ast.get_source_segment(semantic,fn)
        # Original static canonical body, free of the research SQLite implementation.
        (pkg/'semantic_features.py').write_text('from .structured_text import extract_text\nMODEL_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"\nclass SemanticFeatures:\n    @staticmethod\n    '+canonical+'\n',encoding='utf-8')
        cpu=extract(src/'dual_track_cpu_features.py',['CPUItemFeatures']).split('    def predict_demands')[0]
        cpu=cpu.replace("model_path='D:/paiec_pretrained/all-MiniLM-L6-v2'","model_path=None")
        cpu=cpu.replace('        import torch','        model_path = model_path or Path(__file__).resolve().parent / "weights"\n        import torch',1)
        cpu=cpu.replace('        self.model_revision=MODEL_REVISION','        # Detach file-backed safetensors storage so Windows ZIP-check cleanup can unlink weights.\n        for parameter in self.model.parameters():\n            parameter.data = parameter.data.clone()\n        for buffer in self.model.buffers():\n            buffer.data = buffer.data.clone()\n        self.model_revision=MODEL_REVISION')
        (pkg/'features.py').write_text('from pathlib import Path\nimport time\nimport numpy as np\nfrom .semantic_features import SemanticFeatures,MODEL_REVISION\nfrom .metadata_features import encode as metadata_encode\n\n'+cpu,encoding='utf-8')
        shutil.copyfile(HERE/'runtime.py',pkg/'runtime.py')
        entry='''"""Frozen {kind} candidate. See PACKAGE_STATUS.json before deployment."""
from candidate.runtime import Runtime
_runtime=Runtime(shared={shared})
def predict(input,labeled=None):
    return _runtime.run(input,labeled)['prediction']
def predict_with_roster(input,labeled,visible_subjects):
    """Diagnostic/integration API only; not provided by official predict contract."""
    return _runtime.run(input,labeled,visible_subjects)
'''.format(kind=kind,shared=(kind=='partial8_shared'))
        (folder/'model.py').write_text(entry,encoding='utf-8')
        for filename in MODEL_FILES:shutil.copyfile(WEIGHTS/filename,weights/filename)
        shutil.copyfile(HERE/'requirements.txt',folder/'requirements.txt')
        if (HERE/'dependencies.lock.txt').exists():shutil.copyfile(HERE/'dependencies.lock.txt',folder/'dependencies.lock.txt')
        (folder/'README.md').write_text('# '+name+'\n\nFixed semantic ridge 4, metadata ridge 16, dual ridge 1, Partial8 regularization 8.\nMiniLM and tokenization weights are bundled, Apache-2.0 (weights/README.md).\nInstall requirements before inference. No network or external cache required at inference.\n`predict(input, labeled)` is the official entry. No custom acquisition.\nShared candidate requires a trusted roster absent from the known public interface; official calls fall back to R25. `predict_with_roster` is diagnostic only.\nHistorical 4/4/1 scores are not scores of this fixed4 ZIP. See external acceptance report.\n',encoding='utf-8')
        write(folder/'PACKAGE_STATUS.json',{'candidate':kind,'status':'NOT_READY','semantic_ridge':4,'metadata_ridge':16,'dual_ridge':1,'partial_regularization':8,'hosted_score':None,'reason':'Acceptance report external; Hosted platform and shared roster contract unverified.'})
        paths=[src/n for n in ['shared_slope_model.py','shared_slope_adapter.py','shared_slope_adapter_cpu.py','dual_track_cpu_features.py']]+[BASE/'mlp'/n for n in ['semantic_features.py','metadata_features.py','structured_text.py']]
        write(folder/'PROVENANCE.json',{'sources':[{'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in paths],'model_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','weights':{n:sha(weights/n) for n in MODEL_FILES},'adaptations':['relative package imports','research cache helpers excluded','model path local to ZIP','batched pure-input feature LRU limited to 2048 entries; no labels retained','clone identical model tensors to release Windows mmap file handles'],'frozen_function_bodies':['transform','centered_arrays','predict','predict_raw','encode_items','canonical']})
        zip_path=dist/(name+'.zip')
        with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(folder.rglob('*')):
                if not p.is_file() or '__pycache__' in p.parts:continue
                zi=zipfile.ZipInfo(p.relative_to(folder).as_posix(),date_time=(2026,10,23,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;zi.external_attr=0o644<<16
                z.writestr(zi,p.read_bytes())
        results[kind]={'directory':str(folder),'zip':str(zip_path),'sha256':sha(zip_path),'bytes':zip_path.stat().st_size,'files':len(zipfile.ZipFile(zip_path).namelist())}
    write(HERE/'build_manifest.json',results);print(json.dumps(results,indent=2))
if __name__=='__main__':build()
