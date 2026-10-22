"""Frozen offline MiniLM applied to exact support-conditioned token windows."""
import hashlib
from pathlib import Path
import sqlite3
import numpy as np
from .semantic_features import SemanticFeatures,MODEL_REVISION
from .model import encode


class ContextFeatures(SemanticFeatures):
    def __init__(self):
        super().__init__();self.db.close();self.tokenizer.model_max_length=10**9
        self.fingerprint=hashlib.sha256(MODEL_REVISION.encode()+Path(__file__).read_bytes()+Path('mlp/context_tokens.py').read_bytes()+Path('mlp/structured_text.py').read_bytes()).hexdigest()
        folder=Path('D:/paiec_research_cache/context_tokens');folder.mkdir(parents=True,exist_ok=True)
        self.path=folder/(self.fingerprint+'.sqlite');self.db=sqlite3.connect(self.path)
        self.db.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY,vector BLOB)');self.encoded=0;self.lexical_cache={}

    def encode_tokens(self,sequences):
        tokens=list(dict.fromkeys(tuple(x) for x in sequences));found={};pending=[]
        for seq in tokens:
            assert len(seq)<=126
            key=hashlib.sha256(np.asarray(seq,dtype='<i4').tobytes()).hexdigest()
            row=self.db.execute('SELECT vector FROM features WHERE key=?',(key,)).fetchone()
            if row is None:pending.append((seq,key))
            else:found[seq]=np.frombuffer(row[0],np.float32).astype(float)
        for offset in range(0,len(pending),32):
            batch=pending[offset:offset+32];full=[[self.tokenizer.cls_token_id,*seq,self.tokenizer.sep_token_id] for seq,key in batch]
            length=max(map(len,full));assert length<=128
            ids=np.zeros((len(full),length),np.int64);mask=np.zeros_like(ids)
            for i,seq in enumerate(full):ids[i,:len(seq)]=seq;mask[i,:len(seq)]=1
            torch=self.torch
            with torch.no_grad():
                attention=torch.from_numpy(mask);states=self.model(input_ids=torch.from_numpy(ids),attention_mask=attention).last_hidden_state
                pooled=(states*attention[:,:,None]).sum(1)/attention.sum(1)[:,None]
                pooled=torch.nn.functional.normalize(pooled,p=2,dim=1).numpy().astype(np.float32)
            for (seq,key),vector in zip(batch,pooled):
                self.db.execute('INSERT INTO features VALUES (?,?)',(key,vector.tobytes()));found[seq]=vector.astype(float)
            self.db.commit();self.encoded+=len(batch)
            if offset%512==0:print('encoding changed windows',min(offset+32,len(pending)),'/',len(pending),flush=True)
        return found

    def lexical(self,seq):
        seq=tuple(seq)
        if seq not in self.lexical_cache:
            text=self.tokenizer.decode(seq,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            base=encode([{},{}],include_item=False)
            x=(encode([{},{'item_content':text}])-base).astype(float)
            self.lexical_cache[seq]=x/max(np.linalg.norm(x),1e-12)
        return self.lexical_cache[seq]
