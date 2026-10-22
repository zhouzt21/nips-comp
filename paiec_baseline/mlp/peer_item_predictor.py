"""R19 current-evidence exact-item residuals; pure, no evaluation-label inputs."""
import numpy as np


def means(context,alpha=4.,kappa=8.):
    groups={}
    for row in context:groups.setdefault(row['key'],[]).extend(row['labels'])
    total_n=sum(map(len,groups.values()));total_s=sum(map(sum,groups.values()));out={}
    for key,labels in groups.items():
        n=len(labels);s=sum(labels);peers=total_n-n
        base=(1+s)/(2+n)
        mean=(alpha*(kappa*.5+total_s-s)/(kappa+peers)+s)/(alpha+n) if peers else base
        out[key]=(base,mean)
    return out


def predict_many(target_key,query_keys,context,tau,alpha=4.,kappa=8.):
    rates=means(context,alpha,kappa);base,mean=rates.get(target_key,(.5,.5))
    result=np.full(len(query_keys),mean,float)
    if tau is None:return result
    if tau<=0:raise ValueError('tau must be positive or None')
    effects={}
    for row in context:
        if row['key']==target_key:continue
        if len(row['items'])!=len(row['labels']):raise ValueError('items and labels must align')
        for key,y in zip(row['items'],row['labels']):
            s,n=effects.get(key,(0.,0));effects[key]=(s+y-rates[row['key']][1],n+1)
    for i,key in enumerate(query_keys):
        s,n=effects.get(key,(0.,0));result[i]+=s/(tau+n)
    return np.clip(result,0,1)
