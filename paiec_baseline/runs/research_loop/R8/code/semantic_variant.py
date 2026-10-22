"""Research feature backend for matched semantic/lexical MLP inputs."""
from functools import lru_cache
import numpy as np
from .model import encode
from .semantic_features import SemanticFeatures

@lru_cache(maxsize=1)
def engine():return SemanticFeatures()

def prepare(texts):engine().encode_many([str(text) for text in texts])

@lru_cache(maxsize=4096)
def content_features(text):
    backend=engine();key=backend.key(text);data=backend.encode_many([text])[key]
    window=backend.db.execute("SELECT window FROM features WHERE key=?",(key,)).fetchone()[0]
    lexical=encode([{}, {"item_content":window}],partitioned=True)[512:1536]
    lexical=lexical/max(float(np.linalg.norm(lexical)),1e-12)
    semantic=np.zeros(1024,np.float32);semantic[:384]=data["semantic"]
    return lexical,semantic

def encode_input(input,mode):
    subject,item=input
    vector=encode([subject,{**item,"item_content":""}],partitioned=True)
    lexical,semantic=content_features(str(item.get("item_content","")))
    vector[512:1536]=semantic if mode=="semantic_block" else lexical
    return vector
