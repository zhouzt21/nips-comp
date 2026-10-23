"""R18b controlled input retention with the same frozen MiniLM weights."""
import hashlib
from pathlib import Path
import sqlite3
import numpy as np
from .semantic_features import SemanticFeatures,MODEL_REVISION
from .model import encode


class WindowFeatures(SemanticFeatures):
    def __init__(self,limit=256):
        if limit not in [128,256]:raise ValueError('only preregistered windows')
        super().__init__();self.db.close();self.limit=limit
        self.tokenizer.model_max_length=10**9  # tokenize only; manual truncation before forward
        self.fingerprint=hashlib.sha256(f'{MODEL_REVISION}|{limit}'.encode()+Path(__file__).read_bytes()+Path('mlp/semantic_features.py').read_bytes()+Path('mlp/structured_text.py').read_bytes()).hexdigest()
        folder=Path('D:/paiec_research_cache/window_features');folder.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(folder/(self.fingerprint+'.sqlite'))
        self.db.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, vector BLOB, window TEXT)')

    def encode_many(self,texts):
        canonical={hashlib.sha256(self.canonical(t).encode()).hexdigest():self.canonical(t) for t in texts}
        found={};pending=[]
        for key,text in canonical.items():
            row=self.db.execute('SELECT vector,window FROM features WHERE key=?',(key,)).fetchone()
            if row:found[key]=(np.frombuffer(row[0],dtype=np.float32).copy(),row[1])
            else:pending.append((key,text))
        for offset in range(0,len(pending),24):
            batch=pending[offset:offset+24];sequences=[];windows=[]
            tokens=self.tokenizer([text for key,text in batch],add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids']
            head=96 if self.limit==128 else 192;tail=self.limit-2-head
            for ids in tokens:
                if len(ids)>self.limit-2:ids=ids[:head]+ids[-tail:]
                windows.append(self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
                sequences.append([self.tokenizer.cls_token_id,*ids,self.tokenizer.sep_token_id])
            length=max(map(len,sequences));assert length<=self.limit
            ids=np.zeros((len(batch),length),np.int64);mask=np.zeros_like(ids)
            for i,seq in enumerate(sequences):ids[i,:len(seq)]=seq;mask[i,:len(seq)]=1
            torch=self.torch
            with torch.no_grad():
                attention=torch.from_numpy(mask);states=self.model(input_ids=torch.from_numpy(ids),attention_mask=attention).last_hidden_state
                pooled=(states*attention[:,:,None]).sum(1)/attention.sum(1)[:,None]
                pooled=torch.nn.functional.normalize(pooled,p=2,dim=1).numpy().astype(np.float32)
            for (key,text),vector,window in zip(batch,pooled,windows):
                self.db.execute('INSERT OR REPLACE INTO features VALUES (?,?,?)',(key,vector.tobytes(),window));found[key]=(vector,window)
            self.db.commit();self.encoded+=len(batch)
        base=encode([{},{}],include_item=False);result={}
        for key,(semantic,window) in found.items():
            hashed=(encode([{},{'item_content':window}])-base).astype(float);hashed/=max(np.linalg.norm(hashed),1e-12)
            result[key]={'semantic':semantic.astype(float),'text_hash':hashed}
        return result
