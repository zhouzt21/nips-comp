"""Checks for response counting, fit-only transforms and portable inference."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mlp.neural_process import CNP
from mlp.reproduction_episodes import accuracy, aggregate, brier
from mlp.reproduction_training import (decomposition, fit_normalizer, load_checkpoint,
                                      save_checkpoint, tensor_episodes)


def fixture(offset=0):
    rng = np.random.default_rng(88+offset)
    return {'support': rng.normal(size=(31,640)).astype(np.float32),
            'query': rng.normal(size=(3,640)).astype(np.float32),
            'labels': rng.integers(0,2,size=31).astype(np.float32),
            'counts': np.array([2.,5.,3.]), 'successes': np.array([1.,4.,0.]),
            'subject': np.zeros(256,np.float32)}


class ReproductionPipelineTests(unittest.TestCase):
    def test_response_metrics_and_decomposition_equal_expansion(self):
        p=np.array([.2,.8,.5]); counts=np.array([2.,5.,3.]); wins=np.array([1.,4.,0.])
        ys=np.array([1,0,1,1,1,1,0,0,0,0]); ps=np.repeat(p,counts.astype(int))
        self.assertAlmostEqual(brier(p,counts,wins),np.mean((ps-ys)**2),places=14)
        self.assertAlmostEqual(accuracy(p,counts,wins),np.mean((ps>=.5)==ys),places=14)
        self.assertAlmostEqual(sum(decomposition(p,counts,wins).values()),np.mean((ps-ys)**2),places=14)
        # Rate-MSE omits empirical outcome variance and must not replace Brier.
        self.assertGreater(brier(p,counts,wins),np.average((p-wins/counts)**2,weights=counts))

    def test_counted_nll_matches_repeated_binary_likelihood(self):
        episode=fixture(); norm=fit_normalizer([episode]); data=tensor_episodes([episode],norm)
        logit=torch.tensor([[.2,-1.3,.7]],requires_grad=True)
        aggregated=(F.binary_cross_entropy_with_logits(logit,data['rates'],reduction='none')*data['weights']).sum()
        expanded=torch.repeat_interleave(logit[0],torch.tensor([2,5,3]))
        labels=torch.tensor([1.,0.,1.,1.,1.,1.,0.,0.,0.,0.])
        reference=F.binary_cross_entropy_with_logits(expanded,labels)
        self.assertAlmostEqual(float(aggregated.detach()),float(reference.detach()),places=6)
        actual_grad=torch.autograd.grad(aggregated,logit,retain_graph=True)[0]
        reference_grad=torch.autograd.grad(reference,logit)[0]
        torch.testing.assert_close(actual_grad,reference_grad,atol=1e-7,rtol=1e-7)

    def test_fit_transform_does_not_depend_on_any_labels(self):
        episode=fixture(); norm=fit_normalizer([episode])
        mutated={**episode,'labels':1-episode['labels'],'successes':episode['counts']-episode['successes']}
        other=fit_normalizer([mutated])
        for key in norm:np.testing.assert_array_equal(norm[key],other[key])
        self.assertTrue(np.all(norm['scale']>=.05))

    def test_padded_targets_have_zero_loss_mass(self):
        first,second=fixture(),fixture(1)
        second={**second,'query':second['query'][:2],'counts':second['counts'][:2],
                'successes':second['successes'][:2]}
        data=tensor_episodes([first,second],fit_normalizer([first,second]))
        torch.testing.assert_close(data['weights'].sum(1),torch.ones(2))
        self.assertEqual(float(data['weights'][1,2]),0.)

    def test_pair_equal_and_benchmark_equal_are_distinct(self):
        rows=[{'benchmark':bm,'brier':{'m':[score]*6},'accuracy':{'m':[.5]*6}}
              for bm,score in [('a',.1),('a',.1),('b',.4)]]
        summary=aggregate(rows)
        self.assertAlmostEqual(summary['results']['m']['alc_mean'],.2)
        self.assertAlmostEqual(summary['benchmark_equal_alc']['m'],.25)

    def test_checkpoint_keeps_transform_and_predictions(self):
        episode=fixture(); norm=fit_normalizer([episode]); data=tensor_episodes([episode],norm)
        torch.manual_seed(1); model=CNP().eval()
        original=model(data['support'][:,:3],data['labels'][:,:3],data['query']).detach()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            save_checkpoint(path,model,'cnp',norm,1,{'fit_benchmarks':['a']})
            restored,norm2,_=load_checkpoint(path)
            actual=restored(data['support'][:,:3],data['labels'][:,:3],data['query']).detach()
        torch.testing.assert_close(original,actual,rtol=0,atol=0)
        for key in norm:np.testing.assert_array_equal(norm[key],norm2[key])


if __name__=='__main__':unittest.main()
