import unittest
import numpy as np
from mlp.audit.diagnose_local_signal import decompose


class DiagnosticTest(unittest.TestCase):
    def test_exact_decomposition_and_permutation(self):
        p = np.array([.8, .2, .8]); y = np.array([1., 0., 1.])
        d = decompose(p, y, .5)
        self.assertAlmostEqual(d['baseline']-d['actual'], d['mean_gain']+d['item_gain'])
        import itertools
        perm = [np.mean((np.array(v)-y)**2) for v in itertools.permutations(p)]
        self.assertAlmostEqual(d['query_permutation_expectation'], np.mean(perm))
        self.assertGreater(d['item_gain'], 0)

    def test_wrong_ranking_is_penalized(self):
        d = decompose([.9,.1], [0.,1.], .5)
        self.assertLess(d['item_gain'], 0)
        self.assertEqual(d['mean_gain'], 0)


if __name__ == '__main__': unittest.main()
