import unittest
import numpy as np
from mlp.kernel_covariance import predict_kernel


class CovarianceTests(unittest.TestCase):
    def test_zero_eta_is_exact_baseline_and_one_label_uses_similarity(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(35,8));x/=np.linalg.norm(x,axis=1,keepdims=True)
        k=x[:31]@x[:31].T;c=x[31:]@x[:31].T;y=rng.integers(0,2,31)
        for n in [0,1,3,7,15,31]:
            np.testing.assert_allclose(predict_kernel(k,c,y[:n],0),(1+y[:n].sum())/(2+n),atol=1e-12)
        p=predict_kernel(np.ones((1,1)),np.array([[1.],[0.]]),[1],1)
        self.assertGreater(p[0],p[1])
        self.assertTrue(np.all((p>=0)&(p<=1)))


if __name__=="__main__":unittest.main()
