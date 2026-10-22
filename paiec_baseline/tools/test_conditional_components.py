import numpy as np
from mlp.conditional_components import components
from mlp.pooled_item_slope import predict_many


def test_factored_effects_replay_original_predictors_including_duplicate_keys():
    rng=np.random.default_rng(3601)
    for n in [0,1,3,15]:
        rows=[{'key':key,'labels':rng.integers(0,2,n).astype(float),'vectors':rng.normal(size=(n,9))} for key in ['a','b','a','c']]
        queries=[rng.normal(size=(8,9)) for _ in rows]
        result=components(rows,queries,4.,1.,.25,rng)
        for r,q,c in zip(rows,queries,result):
            np.testing.assert_allclose(np.clip(c['peer']+c['pooled'],0,1),predict_many(r['key'],q,rows,4.),atol=1e-12)
            np.testing.assert_allclose(np.clip(c['peer']+c['own'],0,1),predict_many(r['key'],q,rows,1.,False),atol=1e-12)
            np.testing.assert_allclose(np.clip(c['peer']+c['original'],0,1),predict_many(r['key'],q,rows,.25,False),atol=1e-12)


def test_no_peers_pooling_and_permutation_reduce_to_own_effect():
    rng=np.random.default_rng(3602)
    row={'key':'a','labels':np.array([0.,1.,0.]),'vectors':rng.normal(size=(3,7))};q=rng.normal(size=(5,7))
    c=components([row],[q],1.,1.,1.,rng)[0]
    np.testing.assert_allclose(c['pooled'],c['own'],atol=1e-12)
    for shuffled in c['permuted']:np.testing.assert_allclose(shuffled,c['pooled'],atol=1e-12)
    assert c['base']==c['peer']
