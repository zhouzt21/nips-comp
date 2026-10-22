"""Support-only conditioning and dual/primal equivalence for local adaptation."""
import unittest
import numpy as np
from mlp.local_ridge import kernel_predict, predict


class LocalRidgeTests(unittest.TestCase):
    def test_dual_matches_centered_primal(self):
        rng = np.random.default_rng(4)
        support = rng.normal(size=(7, 5))
        query = rng.normal(size=(9, 5))
        labels = np.array([0,1,0,1,1,0,0])
        xs = support-support.mean(0)
        xq = query-support.mean(0)
        beta = np.linalg.solve(xs.T@xs+2*np.eye(5), xs.T@(labels-labels.mean()))
        expected = np.clip((1+labels.sum())/9+xq@beta,0,1)
        actual = kernel_predict(support@support.T,query@support.T,labels,2)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_empty_single_and_other_pair_labels(self):
        target = [{"provider":"a"},{"benchmark_id":"x","item_content":"easy"}]
        obs = [[target,1]]
        self.assertEqual(predict(target,[]), .5)
        self.assertAlmostEqual(predict(target,obs),2/3)
        unrelated = [[[{"provider":"b"},target[1]],0],[[target[0],{"benchmark_id":"y"}],0]]
        self.assertEqual(predict(target,obs),predict(target,obs+unrelated))
        predict(target,obs+[[[target[0],{"benchmark_id":"x","item_content":"hard"}],0]])
        self.assertEqual(predict(target,[]), .5)


if __name__ == "__main__":
    unittest.main()
