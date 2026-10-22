import unittest
import numpy as np
from mlp.run_component_round import score
from mlp.local_ridge import kernel_predict


class ComponentTests(unittest.TestCase):
    def test_isolated_matches_kernel_and_zero_one_slopes(self):
        rng=np.random.default_rng(3)
        rows=[]
        for i in range(4):
            support=rng.normal(size=(31,8));query=rng.normal(size=(17,8))
            support/=np.linalg.norm(support,axis=1)[:,None];query/=np.linalg.norm(query,axis=1)[:,None]
            rows.append({'key':str(i),'support':support,'query':query,'a':rng.integers(2,size=31),'y':rng.integers(2,size=17)})
        out=score(rows,'toy',1,4.,4.,8.)
        np.testing.assert_allclose(out['peer'],out['base'])
        np.testing.assert_allclose(out['combined'],out['local'])
        shared=score(rows,'toy',4,4.,4.,8.)
        np.testing.assert_allclose(shared['combined'][:,:2],shared['peer'][:,:2])
        np.testing.assert_allclose(shared['local'][:,:2],shared['base'][:,:2])
        for row in rows:
            single=score([row],'toy',1,4.,4.,8.)
            for j,n in enumerate([0,1,3,7,15,31]):
                p=kernel_predict(row['support']@row['support'].T,row['query']@row['support'].T,row['a'][:n],4.)
                self.assertAlmostEqual(single['local'][0,j],np.mean((p-row['y'])**2))


if __name__=='__main__':unittest.main()
