import numpy as np
from mlp.metadata_features import fields,encode
from mlp.run.run_metadata_round import prediction


def test_field_parsing_provenance_exclusion_and_order():
    a='category=physics;difficulty=2;source_task_id=private;source_image_path=some/path.jpg'
    assert fields(a)=={'category':'physics','difficulty':'2'}
    np.testing.assert_array_equal(encode(a),encode('{"difficulty":"2","category":"physics"}'))
    np.testing.assert_array_equal(encode('source_task_id=anything'),np.zeros(256))


def test_constant_metadata_preserves_semantic_prediction():
    rng=np.random.default_rng(9);s=rng.normal(size=(7,8));q=rng.normal(size=(12,8));y=rng.integers(2,size=7).astype(float)
    meta=encode('category=physics')
    combined=prediction(np.c_[s,np.tile(meta,(7,1))],np.c_[q,np.tile(meta,(12,1))],y,4)
    np.testing.assert_allclose(combined,prediction(s,q,y,4),atol=1e-12)
