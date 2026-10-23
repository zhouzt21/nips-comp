"""Local CPU feature extraction for unseen items, with no dataset/cache lookup.

The 126-token head/tail rule exactly follows the frozen PAIEC SemanticFeatures.
This module retains model weights only; it never stores or accepts outcome labels.
"""
from pathlib import Path
import sys,time
import numpy as np
BASE=Path(__file__).resolve().parents[1]/'paiec_baseline'
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
from mlp.semantic_features import SemanticFeatures,MODEL_REVISION
from mlp.metadata_features import encode as metadata_encode

class CPUItemFeatures:
    def __init__(self,model_path='D:/paiec_pretrained/all-MiniLM-L6-v2',threads=2):
        import torch
        from transformers import AutoModel,AutoTokenizer
        started=time.perf_counter()
        self.torch=torch;torch.set_num_threads(threads)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,clean_up_tokenization_spaces=False)
        self.model=AutoModel.from_pretrained(model_path,local_files_only=True).to('cpu').eval()
        self.model_revision=MODEL_REVISION
        self.cold_load_seconds=time.perf_counter()-started

    def encode_items(self,items):
        if not items:return np.empty((0,640),dtype=np.float64)
        texts=[SemanticFeatures.canonical(it.get('item_content','')) for it in items]
        blocks=[]
        for start in range(0,len(items),32):
            tokens=self.tokenizer(texts[start:start+32],add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids']
            seq=[]
            for token_ids in tokens:
                if len(token_ids)>126:token_ids=token_ids[:96]+token_ids[-30:]
                seq.append([self.tokenizer.cls_token_id,*token_ids,self.tokenizer.sep_token_id])
            width=max(map(len,seq));ids=np.zeros((len(seq),width),dtype=np.int64);mask=np.zeros_like(ids)
            for j,s in enumerate(seq):ids[j,:len(s)]=s;mask[j,:len(s)]=1
            torch=self.torch
            with torch.no_grad():
                attention=torch.from_numpy(mask)
                states=self.model(input_ids=torch.from_numpy(ids),attention_mask=attention).last_hidden_state
                pooled=(states*attention[:,:,None]).sum(1)/attention.sum(1)[:,None]
                sem=torch.nn.functional.normalize(pooled,p=2,dim=1).numpy().astype(np.float32)
            meta=np.stack([metadata_encode(it.get('item_features','')) for it in items[start:start+32]])
            blocks.append(np.concatenate([sem.astype(np.float64),meta.astype(np.float64)],axis=1))
        return np.concatenate(blocks)

    def raw(self,item):return self.encode_items([item])[0]

    def r25_encoder(self,semantic_ridge):
        if semantic_ridge<=0:raise ValueError('semantic_ridge must be positive and source selected')
        scale=np.r_[np.full(384,np.sqrt(semantic_ridge)),np.full(256,4.)]
        return lambda item:self.raw(item)/scale

    def predict_demands(self,item,heads):
        from demand_student_cpu import predict_demands
        return predict_demands(heads,self.encode_items([item]))[0]
