"""AST/body fidelity, package contents and path audit for the actual archives."""
import ast,zipfile
from common import *
def functions(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def main():
    report={};original=REPO/'research_loop'
    for kind,v in read(HERE/'build_manifest.json').items():
        with zipfile.ZipFile(v['zip']) as z:
            files=z.namelist();assert 'model.py' in files and not any(n.startswith('/') or '..' in Path(n).parts for n in files)
            core=functions(z.read('candidate/core.py').decode());ref=functions((original/'shared_slope_model.py').read_text(encoding='utf-8'))
            for n in ['transform','centered_arrays','predict']:assert core[n]==ref[n]
            adapter=functions(z.read('candidate/adapter.py').decode());ref=functions((original/'shared_slope_adapter.py').read_text(encoding='utf-8'))
            for n in ['canonical','subject_key','item_key','predict_raw']:assert adapter[n]==ref[n]
            def method(text,cls,name):
                c=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==cls)
                return ast.dump(next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
            assert method(z.read('candidate/features.py').decode(),'CPUItemFeatures','encode_items')==method((original/'dual_track_cpu_features.py').read_text(encoding='utf-8'),'CPUItemFeatures','encode_items')
            assert method(z.read('candidate/semantic_features.py').decode(),'SemanticFeatures','canonical')==method((BASE/'mlp/semantic_features.py').read_text(encoding='utf-8'),'SemanticFeatures','canonical')
            for n in ['structured_text.py','metadata_features.py']:assert z.read('candidate/'+n)==(BASE/'mlp'/n).read_bytes()
            code='\n'.join(z.read(n).decode() for n in files if n.endswith('.py'))
            for bad in ['D:/','D:\\','sqlite3','paiec_research_cache','paiec_payload','requests.get','urlopen','R79']:assert bad not in code,bad
            assert not any(n.endswith(('.sqlite','.parquet','.npz','.zip')) or '.env' in n for n in files)
            report[kind]={'sha256':sha(v['zip']),'core_and_adapter_functions_ast_identical':True,'semantic_encoder_ast_identical':True,'structured_text_metadata_bytes_identical':True,'no_absolute_research_paths':True,'files':len(files),'model_bytes':len(z.read('candidate/weights/model.safetensors'))}
    write(HERE/'source_fidelity.json',report);print(json.dumps(report))
if __name__=='__main__':main()
