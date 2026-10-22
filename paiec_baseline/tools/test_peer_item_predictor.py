import copy
import numpy as np
from mlp.peer_item_predictor import predict_many,means


def test_current_evidence_only_and_no_state():
    empty=[{'key':'a','items':[],'labels':[]},{'key':'b','items':[],'labels':[]}]
    full=[{'key':'a','items':['z'],'labels':[0]},{'key':'b','items':['x','y'],'labels':[1,0]}]
    expected=predict_many('a',['x','other'],empty,1)
    np.testing.assert_array_equal(expected,[.5,.5])
    p=predict_many('a',['x','other'],full,1)
    assert p[0]>p[1]
    assert p[1]==means(full)['a'][1]
    np.testing.assert_array_equal(predict_many('a',['x','other'],empty,1),expected)
    shuffled=copy.deepcopy(full);shuffled[1]['items'].reverse()
    assert predict_many('a',['x'],shuffled,1)[0]<p[0]


def test_complement_order_and_no_peer_fallback():
    context=[{'key':'a','items':['z'],'labels':[1]},{'key':'b','items':['x','y'],'labels':[0,1]}]
    p=predict_many('a',['x','y','other'],context,4)
    np.testing.assert_allclose(p,predict_many('a',['x','y','other'],context[::-1],4),atol=1e-12)
    flipped=[{**r,'labels':[1-y for y in r['labels']]} for r in context]
    np.testing.assert_allclose(1-p,predict_many('a',['x','y','other'],flipped,4),atol=1e-12)
    np.testing.assert_array_equal(predict_many('a',['x','y'],context[:1],1),[2/3,2/3])
