import unittest
import numpy as np
from mlp.relative_ability import fit
from mlp.run_ability_round import corrected
from mlp.audit_pooling_mechanism import plugin


class AbilityTests(unittest.TestCase):
    def test_training_scope_and_recovery(self):
        rows=[{'benchmark':b,'key':s,'q':q} for b in ['train1','train2'] for s,q in [('a',.8),('b',.2)]]
        model=fit(rows,['train1','train2'])
        self.assertGreater(model['ability']['a'],model['ability']['b'])
        self.assertLess(model['max_normal_equation_residual'],1e-6)
        with self.assertRaises(ValueError):fit(rows,['train1'])

    def test_unknown_identity_exactly_falls_back(self):
        rows=[{'subject':{'normalized_name':str(i)},'key':str(i),'a':[i%2]*31,'evaluation_mean':.3+i*.1} for i in range(4)]
        np.testing.assert_allclose(corrected(rows,'b',{'ability':{}},4,8),plugin(rows,'b',4,8),atol=1e-12)


if __name__=='__main__':unittest.main()
