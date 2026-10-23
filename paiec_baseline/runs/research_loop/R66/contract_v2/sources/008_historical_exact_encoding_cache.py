"""Bit-exact cache for the original additive float32 hash encoder.

Each coordinate is updated independently in the original encoder. Cache the
ordered item additions and their zero-start result. Replay only coordinates
whose subject-start value is nonzero, keeping the original addition order and
float32 assignment after *every* scalar addition. No feature approximation.
"""
from functools import lru_cache
from types import FunctionType
import hashlib
import numpy as np
from mlp.model import encode,encode_mode,_string,_ITEM_FIELDS,_SUBJECT_FIELDS,FEATURE_DIM

class ExactEncodingCache:
    def __init__(self):
        self.calls=0
        self.item_cached=lru_cache(maxsize=4096)(self.item)
        self.subject_cached=lru_cache(maxsize=32)(self.subject)

    @staticmethod
    def subject(key):
        x=encode([dict(zip(_SUBJECT_FIELDS,key)),{}],include_item=False)
        return x,np.flatnonzero(x)

    @staticmethod
    def item(key):
        updates={}
        def add(vector,namespace,token,weight=1.,*,partitioned=False):
            assert not partitioned
            if namespace.startswith('subject:'):return
            digest=hashlib.blake2b(f'{namespace}\0{token}'.encode('utf-8'),digest_size=8).digest()
            code=int.from_bytes(digest,'little');index=code%FEATURE_DIM
            value=weight if code&(1<<63) else -weight
            vector[index]+=value
            updates.setdefault(index,[]).append(value)
        # A private function object, with a private globals dict: do not mutate mlp.model.
        traced=FunctionType(encode.__code__,{**encode.__globals__,'_add':add},encode.__name__,encode.__defaults__,encode.__closure__)
        traced.__kwdefaults__=encode.__kwdefaults__.copy()
        zero=traced([{},dict(zip(_ITEM_FIELDS,key))])
        return zero,updates

    def __call__(self,input,mode='all'):
        if mode!='all':return encode_mode(input,mode)
        subject,item=input
        start,nonzero=self.subject_cached(tuple(_string(subject.get(k,'')) for k in _SUBJECT_FIELDS))
        zero,updates=self.item_cached(tuple(_string(item.get(k,'')) for k in _ITEM_FIELDS))
        result=zero.copy()
        for index in nonzero:
            result[index]=start[index]
            for value in updates.get(index,()):
                result[index]+=value
        self.calls+=1
        return result

    def stats(self):
        return {'calls':self.calls,'item_cache':self.item_cached.cache_info()._asdict(),
                'subject_cache':self.subject_cached.cache_info()._asdict()}
