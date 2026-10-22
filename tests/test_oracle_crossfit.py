import numpy as np
from mlp.oracle_crossfit import fitted_predictions,crossfit
from mlp.measurement_episodes import weighted_brier


def test_scored_outcomes_never_choose_their_own_scale():
    d=np.array([.2,-.2,.1,-.1]);m=np.ones(4);s=np.array([1.,0.,0.,1.])
    p,c,_=fitted_predictions(.5,d,m,s,[0,1],[2,3])
    s[2:]=1-s[2:]
    q,e,_=fitted_predictions(.5,d,m,s,[0,1],[2,3])
    np.testing.assert_array_equal(p,q);assert c==e


def test_known_relation_generalizes_and_repeat_weights_match():
    d=np.array([.2,-.2,.2,-.2]);m=np.array([3,2,4,5]);s=np.array([3,0,4,0])
    scores,c=crossfit(.5,d,m,s,np.arange(4),4801)
    assert scores['crossfit']==0.;np.testing.assert_allclose(c,[2.5,2.5])
    p=np.array([.2,.3,.6,.7]);expanded=np.repeat(p,m)
    outcomes=np.concatenate([np.r_[np.ones(a),np.zeros(b-a)] for a,b in zip(s,m)])
    np.testing.assert_allclose(weighted_brier(p,m,s),np.mean((expanded-outcomes)**2))


def test_zero_direction_cannot_exploit_extra_labels():
    scores,c=crossfit(.4,np.zeros(4),np.ones(4),[0,1,1,1],np.arange(4),4801)
    np.testing.assert_allclose(scores['crossfit'],weighted_brier(.4,np.ones(4),np.array([0,1,1,1])));assert c==[0.,0.]
