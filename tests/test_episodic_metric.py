import unittest
import numpy as np
import torch
from mlp.episodic_metric import probabilities,torch_probabilities


class EpisodicMetricTests(unittest.TestCase):
    def test_export_and_budget_isolation(self):
        rng=np.random.default_rng(10)
        x=rng.normal(size=(42,2048)).astype(np.float32)
        y=rng.integers(0,2,31).astype(np.float32)
        w=rng.normal(0,.2,2048).astype(np.float32)
        for n in [0,1,3,7,15,31]:
            reference=torch_probabilities(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(w),n).numpy()
            np.testing.assert_allclose(probabilities(x,y,w,n),reference,atol=2e-6)
        changed=y.copy();changed[3:]=1-changed[3:]
        np.testing.assert_array_equal(probabilities(x,y,w,3),probabilities(x,changed,w,3))


if __name__=="__main__":unittest.main()
