"""Development-only frozen MiniLM features and matched retained-text hashes."""
import hashlib
from pathlib import Path
import sqlite3
import numpy as np
from .model import encode
from .structured_text import extract_text

MODEL_REVISION="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SemanticFeatures:
    def __init__(self):
        import torch
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;torch.set_num_threads(4)
        self.root=Path("D:/paiec_pretrained/all-MiniLM-L6-v2")
        self.tokenizer=AutoTokenizer.from_pretrained(self.root,local_files_only=True,clean_up_tokenization_spaces=False)
        self.model=AutoModel.from_pretrained(self.root,local_files_only=True).eval()
        self.fingerprint=hashlib.sha256(MODEL_REVISION.encode()+Path(__file__).read_bytes()+Path("mlp/structured_text.py").read_bytes()).hexdigest()
        folder=Path("D:/paiec_research_cache/semantic");folder.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(folder/(self.fingerprint+".sqlite"))
        self.db.execute("CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, vector BLOB, window TEXT)")
        self.encoded=0

    @staticmethod
    def canonical(text):
        text=extract_text(text)
        return text if len(text)<=20000 else text[:16000]+text[-4000:]

    def encode_many(self,texts):
        canonical={hashlib.sha256(self.canonical(text).encode()).hexdigest():self.canonical(text) for text in texts}
        found={};pending=[]
        for key,text in canonical.items():
            row=self.db.execute("SELECT vector,window FROM features WHERE key=?",(key,)).fetchone()
            if row:found[key]=(np.frombuffer(row[0],dtype=np.float32).copy(),row[1])
            else:pending.append((key,text))
        for offset in range(0,len(pending),32):
            batch=pending[offset:offset+32]; sequences=[];windows=[]
            tokens=self.tokenizer([text for key,text in batch],add_special_tokens=False,truncation=False,return_attention_mask=False)["input_ids"]
            for ids in tokens:
                if len(ids)>126:ids=ids[:96]+ids[-30:]
                windows.append(self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
                sequences.append([self.tokenizer.cls_token_id,*ids,self.tokenizer.sep_token_id])
            length=max(map(len,sequences));input_ids=np.zeros((len(batch),length),np.int64);mask=np.zeros_like(input_ids)
            for i,seq in enumerate(sequences):input_ids[i,:len(seq)]=seq;mask[i,:len(seq)]=1
            torch=self.torch
            with torch.no_grad():
                attention=torch.from_numpy(mask)
                states=self.model(input_ids=torch.from_numpy(input_ids),attention_mask=attention).last_hidden_state
                pooled=(states*attention[:,:,None]).sum(1)/attention.sum(1)[:,None]
                pooled=torch.nn.functional.normalize(pooled,p=2,dim=1).numpy().astype(np.float32)
            for (key,text),vector,window in zip(batch,pooled,windows):
                self.db.execute("INSERT OR REPLACE INTO features VALUES (?,?,?)",(key,vector.tobytes(),window))
                found[key]=(vector,window)
            self.db.commit();self.encoded+=len(batch)
        base=encode([{},{}],include_item=False)
        result={}
        for key,(semantic,window) in found.items():
            hashed=(encode([{}, {"item_content":window}])-base).astype(float)
            hashed/=max(np.linalg.norm(hashed),1e-12)
            result[key]={"semantic":semantic.astype(float),"text_hash":hashed}
        return result

    def key(self,text):return hashlib.sha256(self.canonical(text).encode()).hexdigest()
