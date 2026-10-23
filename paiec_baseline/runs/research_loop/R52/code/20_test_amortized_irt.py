"""Small numerical contracts, including correspondence to the frozen source."""
import io
import unittest

import torch
from torch.nn import functional as F

from paiec_baseline.mlp.amortized_irt import (
    AmortizedIRT, fit_training_abilities, fit_training_abilities_counts,
    make_nuisance_abilities, reeval_expected_nll, reeval_expected_nll_counts,
)


class AmortizedIRTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = AmortizedIRT(item_dim=3, subject_dim=2).double()
        self.subject = torch.randn(2, 2, dtype=torch.float64)
        self.query = torch.randn(2, 5, 3, dtype=torch.float64)
        self.context = torch.randn(2, 7, 3, dtype=torch.float64)
        self.labels = torch.tensor([[0, 1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1]], dtype=torch.float64)

    def test_b0_and_centered_difficulty_gauge(self):
        cx, cy = self.context[:, :0], self.labels[:, :0]
        logits = self.model.predict_logits(cx, cy, self.query, self.subject)
        expected = self.model.ability(self.subject) - self.model.difficulty(self.query).squeeze(-1)
        torch.testing.assert_close(logits, expected)
        centered = self.query - self.query.mean((0, 1), keepdim=True)
        self.assertAlmostEqual(self.model.difficulty(centered).mean().item(), 0, places=12)
        F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_map_stationarity_and_monotone_labels(self):
        self.model.eval()
        logits = self.model.predict_logits(self.context, self.labels, self.query, self.subject)
        theta = logits[:, 0] + self.model.difficulty(self.query[:, 0]).squeeze(-1)
        prior = self.model.ability(self.subject).squeeze(-1)
        p = torch.sigmoid(theta[:, None] - self.model.difficulty(self.context).squeeze(-1))
        gradient = (p - self.labels).sum(-1) + theta - prior
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-11, rtol=0)
        all_zero = self.model(self.context, torch.zeros_like(self.labels), self.query, self.subject)
        all_one = self.model(self.context, torch.ones_like(self.labels), self.query, self.subject)
        self.assertTrue(bool((all_one > all_zero).all()))

    def test_support_set_order_and_query_isolation(self):
        self.model.eval()
        expected = self.model(self.context, self.labels, self.query, self.subject)
        order = torch.tensor([3, 0, 5, 2, 6, 1, 4])
        actual = self.model(self.context[:, order], self.labels[:, order], self.query, self.subject)
        torch.testing.assert_close(expected, actual, atol=1e-12, rtol=0)
        # Rasch ability's sufficient statistic is total successes.  Unlike
        # CNP/ANP, shuffling labels within this fixed support set is invariant.
        shuffled_labels = self.model(self.context, self.labels[:, order], self.query, self.subject)
        torch.testing.assert_close(expected, shuffled_labels, atol=1e-12, rtol=0)
        one_query = self.model(self.context, self.labels, self.query[:, :1], self.subject)
        torch.testing.assert_close(expected[:, :1], one_query, atol=1e-12, rtol=0)
        self.assertFalse(hasattr(self.model, "query_labels"))

    def test_budget_binary_and_training_guards(self):
        with self.assertRaisesRegex(ValueError, "B0"):
            self.model(self.context, self.labels, self.query, self.subject)
        self.model.eval()
        with self.assertRaisesRegex(ValueError, "binary"):
            self.model(self.context, self.labels + .1, self.query, self.subject)
        with self.assertRaisesRegex(ValueError, "31"):
            self.model(self.context.repeat(1, 5, 1), self.labels.repeat(1, 5), self.query, self.subject)

    def test_state_roundtrip_and_extreme_finite_inputs(self):
        self.model.eval()
        expected = self.model(self.context, self.labels, self.query, self.subject)
        saved = io.BytesIO()
        torch.save(self.model.state_dict(), saved)
        saved.seek(0)
        loaded = AmortizedIRT(3, 2).double().eval()
        loaded.load_state_dict(torch.load(saved, weights_only=True))
        torch.testing.assert_close(loaded(self.context, self.labels, self.query, self.subject), expected)
        self.assertTrue(bool(torch.isfinite(loaded(self.context * 1e5, self.labels, self.query * 1e5, self.subject)).all()))

    def test_exact_notebook_expected_nll_and_gradient(self):
        # Original sign is easiness z; local difficulty is -z.
        nuisance = make_nuisance_abilities(2).double()
        torch.testing.assert_close(nuisance, make_nuisance_abilities(2).double(), rtol=0, atol=0)
        difficulty = torch.tensor([-.4, .7, 1.2], dtype=torch.float64, requires_grad=True)
        matrix = torch.tensor([[0., 1., 0.], [1., 0., 1.]], dtype=torch.float64)
        mask = torch.tensor([[1., 1., 0.], [0., 1., 1.]], dtype=torch.float64)
        probability = torch.sigmoid(nuisance[:, :, None] - difficulty[None, None, :])
        reference = -(torch.distributions.Bernoulli(probs=probability).log_prob(matrix) * mask).mean()
        item = torch.tensor([0, 1, 1, 2])
        subject = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0., 1., 0., 1.], dtype=torch.float64)
        actual = reeval_expected_nll(difficulty[item], labels, subject, nuisance, grid_size=6)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        reference_grad = torch.autograd.grad(reference, difficulty, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual, difficulty)[0]
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-12)

    def test_repeats_are_exact_not_majority_vote(self):
        nuisance = make_nuisance_abilities(1).double()
        difficulty = torch.tensor([.4, -.7], dtype=torch.float64)
        grouped = reeval_expected_nll_counts(difficulty, torch.tensor([3., 2.]),
                                             torch.tensor([1., 2.]), torch.tensor([0, 0]), nuisance)
        raw_d = difficulty[torch.tensor([0, 0, 0, 1, 1])]
        raw_y = torch.tensor([0., 0., 1., 1., 1.], dtype=torch.float64)
        raw_subject = torch.zeros(5, dtype=torch.long)
        raw = reeval_expected_nll(raw_d, raw_y, raw_subject, nuisance)
        torch.testing.assert_close(grouped, raw, rtol=1e-12, atol=1e-12)
        a, valid_a = fit_training_abilities(raw_d, raw_y, raw_subject, 1, prior_precision=1)
        b, valid_b = fit_training_abilities_counts(difficulty, torch.tensor([3., 2.]),
                                                  torch.tensor([1., 2.]), torch.tensor([0, 0]), 1,
                                                  prior_precision=1)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        self.assertEqual(valid_a.tolist(), valid_b.tolist())

    def test_mle_nonidentification_and_regularized_map(self):
        d = torch.tensor([-.5, .4, .7, -.3], dtype=torch.float64)
        y = torch.tensor([0., 1., 1., 1.], dtype=torch.float64)
        index = torch.tensor([0, 0, 1, 1])
        theta, finite = fit_training_abilities(d, y, index, 3)
        self.assertEqual(finite.tolist(), [True, False, False])
        self.assertTrue(bool(torch.isnan(theta[1:]).all()))
        self.assertAlmostEqual(theta[0].item(), -.05, places=12)
        theta, present = fit_training_abilities(d, y, index, 3, prior_precision=1)
        self.assertEqual(present.tolist(), [True, True, False])
        residual = torch.sigmoid(theta[index] - d) - y
        derivative = torch.zeros(3, dtype=d.dtype).index_add_(0, index, residual) + theta
        torch.testing.assert_close(derivative[:2], torch.zeros(2, dtype=d.dtype), atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
