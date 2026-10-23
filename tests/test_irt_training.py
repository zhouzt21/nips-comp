"""Bounded synthetic checks for R52 weighting, leakage boundaries and artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mlp.irt_training import (
    IncompleteIRTFit, expected_pair_nll, low_budget_gate, pack_fit_queries,
    ridge_ability, train_amortized_irt, training_fingerprint,
)
from mlp.reproduction_training import fit_normalizer, load_checkpoint
from mlp.run.run_amortized_irt_round import merge_ablation


def episodes():
    rng = np.random.default_rng(62); result = []
    for i in range(3):
        support = np.zeros((31, 640), np.float32); support[:, :3] = rng.normal(size=(31, 3))
        query = np.zeros((3, 640), np.float32); query[:, :3] = rng.normal(size=(3, 3))
        subject = np.zeros(256, np.float32); subject[i % 2] = 1
        result.append({'benchmark': f'fit{i % 2}', 'subject_id': f'local{i}',
                       'subject_key': f'canonical{i % 2}', 'seed': 73,
                       'support_item_ids': [f's{j}' for j in range(31)],
                       'query_item_ids': [f'q{j}' for j in range(3)],
                       'support': support, 'query': query,
                       'labels': np.asarray([j % 2 for j in range(31)], np.float32),
                       'counts': np.array([4, 4, 4], np.float64),
                       'successes': np.array([1, 3, 2], np.float64), 'subject': subject})
    return result


class IRTTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_expected_nll_pair_weight_and_chunk_gradient(self):
        # Pair 1 has four responses; pair 2 has one. Each contributes half.
        d = torch.tensor([.7, -.2, .5], dtype=torch.float64, requires_grad=True)
        n = torch.tensor([1., 3., 1.], dtype=torch.float64)
        s = torch.tensor([1., 1., 0.], dtype=torch.float64)
        index = torch.tensor([0, 0, 1]); nuisance = torch.tensor([[-1., 2.], [1., -2.]], dtype=torch.float64)
        w = torch.tensor([1/8, 1/8, 1/2], dtype=torch.float64)
        actual = expected_pair_nll(d, n, s, index, nuisance, w)
        z = nuisance[:, index]-d
        p = torch.sigmoid(z)
        bernoulli = -(s*torch.log(p)+(n-s)*torch.log1p(-p))
        expected = ((bernoulli[:, :2].sum(-1)/4 + bernoulli[:, 2])/2).mean()
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual, d)[0]
        chunks = sum(expected_pair_nll(d[i:i+1], n[i:i+1], s[i:i+1], index[i:i+1], nuisance, w[i:i+1])
                     for i in range(3))
        torch.testing.assert_close(torch.autograd.grad(chunks, d)[0], gradient)

    def test_only_query_targets_and_shared_subject_nuisance_index(self):
        fit = episodes(); norm = fit_normalizer(fit)
        data, keys, ledger = pack_fit_queries(fit, norm)
        self.assertEqual(len(keys), 2)
        self.assertEqual(data['subject_index'].tolist(), [0, 0, 0, 1, 1, 1, 0, 0, 0])
        self.assertAlmostEqual(float((data['counts']*data['row_weights']).sum()), 1.)
        changed = copy.deepcopy(fit)
        for ep in changed: ep['labels'][:] = 1-ep['labels']
        self.assertEqual(training_fingerprint(fit), training_fingerprint(changed))
        changed[0]['successes'][0] += 1
        self.assertNotEqual(training_fingerprint(fit), training_fingerprint(changed))
        self.assertEqual(sum(sum(row['query_counts']) for row in ledger), 36)

    def test_ridge_unpenalized_intercept_and_uniform_subjects(self):
        x = np.array([[0., 0.], [1., 0.], [2., 0.]])
        theta = np.array([11., 12., 13.])
        weight, bias = ridge_ability(x, theta)
        np.testing.assert_allclose(weight, [2/3, 0.])
        self.assertAlmostEqual(bias, 12-2/3)
        self.assertAlmostEqual(np.mean(x@weight+bias-theta), 0.)

    def test_tiny_fit_no_validation_targets_and_checkpoint_contract(self):
        fit = episodes()
        # Only a split identifier is present. Accessing validation response data
        # would fail; the training API must never read it or select an epoch.
        validation = [{'benchmark': 'heldout'}]
        with tempfile.TemporaryDirectory(prefix='irtfit-', dir=Path(__file__).resolve().parents[1]) as directory:
            summary = train_amortized_irt(fit, validation, directory, seed=42, threads=1,
                                          chunk_size=4, max_outer_iterations=20, time_limit=30,
                                          source_fingerprint='synthetic')
            self.assertEqual(summary['status'], 'complete')
            self.assertLessEqual(summary['gradient_inf'], 1e-4)
            model, norm, artifact = load_checkpoint(Path(directory)/'final.pt')
            zero, _, ablation = load_checkpoint(Path(directory)/'zero_subject_prior.pt')
            self.assertEqual(artifact['model_name'], 'irt_calibration')
            self.assertFalse(artifact['metadata']['validation_epoch_selection'])
            self.assertTrue(torch.equal(model.difficulty.weight, zero.difficulty.weight))
            self.assertEqual(float(zero.ability.weight.detach().abs().sum()), 0.)
            self.assertEqual(float(zero.ability.bias.detach().abs().sum()), 0.)
            self.assertIn('ablation', ablation['metadata'])
            np.testing.assert_array_equal(norm['mean'], fit_normalizer(fit)['mean'])
            details = json.loads((Path(directory)/'subject_ability_diagnostics.json').read_text())
            self.assertEqual(sum(row['responses'] for row in details['subjects']), 36)
            resumed = train_amortized_irt(fit, validation, directory, seed=42, threads=1,
                                         chunk_size=4, max_outer_iterations=20, time_limit=30,
                                         source_fingerprint='synthetic')
            self.assertEqual(summary, resumed)

    def test_timeout_is_incomplete_not_negative_result(self):
        with tempfile.TemporaryDirectory(prefix='irtstop-', dir=Path(__file__).resolve().parents[1]) as directory:
            with self.assertRaises(IncompleteIRTFit):
                train_amortized_irt(episodes(), [{'benchmark': 'heldout'}], directory,
                                     threads=1, time_limit=1e-12)
            stop = json.loads((Path(directory)/'resource_stop.json').read_text())
            self.assertEqual(stop['status'], 'incomplete')
            self.assertTrue(stop['not_a_completed_negative_result'])
            self.assertFalse((Path(directory)/'final.pt').exists())

    def test_gate_requires_each_b1_baseline_and_strict_majority(self):
        names = ('irt_calibration', 'R25', 'smoothed_mean_2', 'training_selected_smoothed')
        summary = {'results': {name: {'brier_mean': [.24, .2, .2, .2, .2, .2]} for name in names},
                   'by_benchmark': {str(i): {'brier_mean': {name: [.24, .2, .2, .2, .2, .2] for name in names}}
                                    for i in range(2)}}
        summary['results']['irt_calibration']['brier_mean'][1] = .19
        for group in summary['by_benchmark'].values(): group['brier_mean']['irt_calibration'][1] = .19
        self.assertTrue(low_budget_gate(summary)['admitted'])
        summary['by_benchmark']['0']['brier_mean']['irt_calibration'][0] = .25
        self.assertFalse(low_budget_gate(summary)['admitted'])
        summary['by_benchmark']['0']['brier_mean']['irt_calibration'][0] = .24
        summary['results']['training_selected_smoothed']['brier_mean'][1] = .19
        self.assertFalse(low_budget_gate(summary)['admitted'])

    def test_ablation_names_do_not_replace_primary(self):
        row = {'benchmark': 'b', 'subject_id': 's', 'seed': 73,
               'brier': {'irt_calibration': [.1]*6, 'irt_calibration_permuted': [.1]*6},
               'accuracy': {'irt_calibration': [.8]*6, 'irt_calibration_permuted': [.8]*6},
               'decomposition': []}
        ablation = copy.deepcopy(row); ablation['brier']['irt_calibration'] = [.3]*6
        result = merge_ablation([row], [ablation])[0]
        self.assertEqual(result['brier']['irt_calibration'], [.1]*6)
        self.assertEqual(result['brier']['irt_zero_subject_prior'], [.3]*6)
        self.assertIn('irt_zero_subject_prior_permuted', result['brier'])


if __name__ == '__main__':
    unittest.main()
