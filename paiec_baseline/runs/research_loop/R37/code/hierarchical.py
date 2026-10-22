"""Research-only finite quadrature for shared-evidence beta-binomial prediction."""
from functools import lru_cache
import json
import numpy as np


class HierarchicalCounts:
    def __init__(self, kappa=2., concentrations=(1.,4.,16.), nodes=128):
        if kappa<=0 or any(c<=0 for c in concentrations):raise ValueError('positive strengths required')
        x,w=np.polynomial.legendre.leggauss(nodes)
        self.theta=(x+1)/2;self.c=np.asarray(concentrations,float)[:,None]
        self.log_prior=np.log(w/2)+(kappa/2-1)*(np.log(self.theta)+np.log1p(-self.theta))

    @lru_cache(maxsize=4096)
    def likelihood(self,s,n):
        if int(s)!=s or int(n)!=n or not 0<=s<=n:raise ValueError('invalid binary counts')
        a=self.c*self.theta;b=self.c*(1-self.theta)
        out=np.zeros_like(a)
        for j in range(int(s)):out+=np.log(a+j)
        for j in range(int(n-s)):out+=np.log(b+j)
        for j in range(int(n)):out-=np.log(self.c+j)
        return out

    def predict_counts(self,counts):
        """Order matches returned predictions; no evidence persists between calls."""
        counts=[(int(s),int(n)) for s,n in counts]
        for s,n in counts:
            if not 0<=s<=n:raise ValueError('invalid counts')
        if sum(n>0 for s,n in counts)<=1:
            return np.array([(1+s)/(2+n) for s,n in counts]),float('nan')
        logw=np.broadcast_to(self.log_prior,(len(self.c),len(self.theta))).copy()
        for s,n in counts:logw+=self.likelihood(s,n)
        w=np.exp(logw-logw.max());w/=w.sum()
        predictions=np.array([np.sum(w*(s+self.c*self.theta)/(n+self.c)) for s,n in counts])
        return predictions,float(np.sum(w*self.c))


def predict(input,labeled=None,*,kappa=2.,adaptive=True):
    subject,item=input;benchmark=item.get('benchmark_id')
    key=lambda s:json.dumps(s,sort_keys=True,ensure_ascii=False)
    target=key(subject);groups={target:[0,0]}
    for (other,other_item),label in (labeled or []):
        if not benchmark or other_item.get('benchmark_id')!=benchmark:continue
        if label not in (0,1):raise ValueError('binary label required')
        row=groups.setdefault(key(other),[0,0]);row[0]+=label;row[1]+=1
    solver=HierarchicalCounts(kappa, (1.,4.,16.) if adaptive else (4.,))
    return float(solver.predict_counts(list(groups.values()))[0][0])
