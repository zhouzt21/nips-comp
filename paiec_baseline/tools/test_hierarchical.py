import unittest
import numpy as np
from shared_mean.hierarchical import HierarchicalCounts,predict


class HierarchyTests(unittest.TestCase):
    def test_symmetry_order_and_quadrature(self):
        for kappa in [2,8]:
            model=HierarchicalCounts(kappa);fine=HierarchicalCounts(kappa,nodes=256)
            for counts in [[(0,31),(31,31),(0,31),(31,31),(0,31),(31,31)],[(0,31)]*6,[(7,15),(1,15),(15,15)],[(0,1),(1,1)]]:
                p,_=model.predict_counts(counts)
                np.testing.assert_allclose(p,fine.predict_counts(counts)[0],atol=2e-10)
                np.testing.assert_allclose(p,model.predict_counts(counts[::-1])[0][::-1],atol=1e-12)
                np.testing.assert_allclose(p,1-model.predict_counts([(n-s,n) for s,n in counts])[0],atol=1e-12)

    def test_fallback_scope_and_no_state(self):
        target=[{'model':'a'},{'benchmark_id':'x'}]
        own=[(target,1)]
        self.assertAlmostEqual(predict(target,own),2/3)
        self.assertAlmostEqual(predict(target,own+[([{'model':'b'},{'benchmark_id':'other'}],0)]),2/3)
        predict(target,own+[([{'model':'b'},{'benchmark_id':'x'}],0)])
        self.assertEqual(predict(target,[]),.5)


if __name__=='__main__':unittest.main()
