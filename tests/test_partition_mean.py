import numpy as np
from shared_mean.partition import partitions, prior, predict_counts


def test_partition_counts_and_prior_normalization():
    assert [len(partitions(n)) for n in [1,2,3,4]]==[1,2,5,15]
    for n in [1,2,3,4]:
        for alpha in [1.,4.,16.]:
            _,logs=prior(n,alpha);np.testing.assert_allclose(np.exp(logs).sum(),1.,atol=1e-12)


def test_independent_single_group_and_zero_label_limits():
    counts=np.array([[1,3],[7,15],[0,1]])
    np.testing.assert_array_equal(predict_counts(counts,None)[0],(counts[:,0]+1)/(counts[:,1]+2))
    for alpha in [1.,4.,16.]:
        np.testing.assert_allclose(predict_counts([[3,7]],alpha)[0],[4/9],atol=1e-12)
        np.testing.assert_allclose(predict_counts([[0,0]]*4,alpha)[0],[.5]*4,atol=1e-12)


def test_order_and_label_complement_invariance():
    counts=np.array([[1,7],[6,7],[2,7],[3,7]]);order=np.array([2,0,3,1])
    for update in [False,True]:
        p,w=predict_counts(counts,4.,update)
        pp,ww=predict_counts(counts[order],4.,update)
        np.testing.assert_allclose(pp,p[order],atol=1e-12);np.testing.assert_allclose(ww,w,atol=1e-12)
        reverse=counts.copy();reverse[:,0]=counts[:,1]-counts[:,0]
        np.testing.assert_allclose(predict_counts(reverse,4.,update)[0],1-p,atol=1e-12)


def test_conflicting_high_counts_favor_independence_over_common_rate():
    _,prior_weight=predict_counts([[0,100],[100,100]],1.,False)
    p,posterior_weight=predict_counts([[0,100],[100,100]],1.,True)
    assert posterior_weight>prior_weight and posterior_weight>.999
    np.testing.assert_allclose(p,[1/102,101/102],atol=1e-8)
