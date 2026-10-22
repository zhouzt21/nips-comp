import numpy as np
from mlp.audit.audit_input_equivalence import class_loss,digest


def test_class_loss_known_counts_and_coarsening():
    fine,_=class_loss(['a','b'],[2,2],[0,2]);coarse,k=class_loss(['a','a'],[2,2],[0,2])
    assert fine==0 and coarse==.25 and k==1
    duplicate,_=class_loss(['a','a','b','b'],[1]*4,[0,1,1,1])
    aggregated,_=class_loss(['a','b'],[2,2],[1,2])
    assert duplicate==aggregated==.125
    np.testing.assert_allclose(class_loss(['b','a'],[2,2],[2,1])[0],aggregated)


def test_keys_preserve_field_boundaries():
    assert digest('ab','c')!=digest('a','bc')
    assert digest('ab','c')==digest('ab','c')
