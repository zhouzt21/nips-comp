"""Equation-level and deployment-contract checks for deterministic neural processes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mlp.neural_process import (
    AttentiveCNP, CNP, dot_product_attention,
    load_neural_process, save_neural_process,
)


def numpy_mlp(module, value):
    for layer in module:
        if isinstance(layer, nn.Linear):
            value = value @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()
        elif isinstance(layer, nn.ReLU):
            value = np.maximum(value, 0)
        else:
            raise AssertionError(type(layer))
    return value


class NeuralProcessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(28)
        torch.set_num_threads(1)
        self.cx = torch.randn(2, 7, 5, dtype=torch.float64)
        self.cy = torch.randint(0, 2, (2, 7)).double()
        self.tx = torch.randn(2, 9, 5, dtype=torch.float64)

    def models(self):
        return [CNP(5, 12, 8).double(), AttentiveCNP(5, 12, 8).double()]

    def test_full_forward_matches_independent_numpy_equations(self):
        # Same stored dense parameters; independent matrix and softmax equations.
        for model in self.models():
            cx, cy, tx = self.cx.numpy(), self.cy.numpy(), self.tx.numpy()
            values = numpy_mlp(model.context_encoder, np.concatenate([cx, cy[..., None]], -1))
            if isinstance(model, AttentiveCNP):
                keys = numpy_mlp(model.key_query_encoder, cx)
                queries = numpy_mlp(model.key_query_encoder, tx)
                scores = np.einsum("bqd,bnd->bqn", queries, keys) / np.sqrt(keys.shape[-1])
                weights = np.exp(scores - scores.max(-1, keepdims=True))
                weights /= weights.sum(-1, keepdims=True)
                rep = np.einsum("bqn,bnr->bqr", weights, values)
            else:
                rep = np.repeat(values.mean(1, keepdims=True), tx.shape[1], axis=1)
            logits = numpy_mlp(model.decoder, np.concatenate([rep, tx], -1))[..., 0]
            expected = 1 / (1 + np.exp(-logits))
            actual = model(self.cx, self.cy, self.tx).detach().numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_context_set_and_target_order_contracts(self):
        permutation = torch.tensor([4, 0, 2, 6, 1, 5, 3])
        target_order = torch.tensor([8, 4, 2, 0, 6, 1, 3, 7, 5])
        for model in self.models():
            expected = model(self.cx, self.cy, self.tx)
            torch.testing.assert_close(
                model(self.cx[:, permutation], self.cy[:, permutation], self.tx), expected,
                atol=1e-12, rtol=1e-12,
            )
            torch.testing.assert_close(
                model(self.cx, self.cy, self.tx[:, target_order]), expected[:, target_order],
                atol=1e-12, rtol=1e-12,
            )
            separate = torch.cat([
                model(self.cx, self.cy, self.tx[:, index:index + 1])
                for index in range(self.tx.shape[1])
            ], 1)
            torch.testing.assert_close(separate, expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(
                model(self.cx[:1], self.cy[:1], self.tx[:1]), expected[:1],
                atol=1e-12, rtol=1e-12,
            )

    def test_empty_context_and_six_budget_forward(self):
        cx = torch.randn(2, 31, 5, dtype=torch.float64)
        cy = torch.randint(0, 2, (2, 31)).double()
        for model in self.models():
            for budget in (0, 1, 3, 7, 15, 31):
                prediction = model(cx[:, :budget], cy[:, :budget], self.tx)
                self.assertEqual(prediction.shape, (2, 9))
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())
            representation = model.context_representation(cx[:, :0], cy[:, :0], self.tx)
            torch.testing.assert_close(representation, torch.zeros_like(representation))
            expected = torch.sigmoid(model.decoder(torch.cat([representation, self.tx], -1))).squeeze(-1)
            torch.testing.assert_close(model(cx[:, :0], cy[:, :0], self.tx), expected)

    def test_support_label_correspondence_is_used(self):
        for model in self.models():
            original = model(self.cx, self.cy, self.tx)
            changed = model(self.cx, 1 - self.cy, self.tx)
            permuted = model(self.cx, self.cy.flip(1), self.tx)
            self.assertGreater((original - changed).abs().max().item(), 1e-8)
            self.assertGreater((original - permuted).abs().max().item(), 1e-8)
            labels = self.cy.detach().clone().requires_grad_(True)
            model(self.cx, labels, self.tx).sum().backward()
            self.assertGreater(labels.grad.abs().sum().item(), 1e-8)

    def test_attention_weights_follow_paired_labels(self):
        keys = torch.tensor([[[-2.0], [2.0]]])
        values = torch.tensor([[[0.0], [1.0]]])
        actual = dot_product_attention(keys, keys, values)
        self.assertLess(actual[0, 0, 0].item(), 0.001)
        self.assertGreater(actual[0, 1, 0].item(), 0.999)
        torch.testing.assert_close(dot_product_attention(keys, keys, values.flip(1)), 1 - actual)

    def test_finite_training_update_and_portable_reload(self):
        target_y = torch.randint(0, 2, (2, 9)).float()
        for model in self.models():
            model = model.float()
            inputs = (self.cx.float(), self.cy.float(), self.tx.float())
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            before = model.decoder[-1].weight.detach().clone()
            loss = nn.functional.binary_cross_entropy_with_logits(model.predict_logits(*inputs), target_y)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
            self.assertFalse(torch.equal(before, model.decoder[-1].weight))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "neural_process.pt"
                save_neural_process(model, path)
                restored = load_neural_process(path)
                self.assertEqual(model.export_config(), restored.export_config())
                torch.testing.assert_close(model(*inputs), restored(*inputs), rtol=0, atol=0)

    def test_reject_mismatched_shapes(self):
        model = CNP(5)
        with self.assertRaises(ValueError):
            model(self.cx, self.cy[:, :1], self.tx)
        with self.assertRaises(ValueError):
            model(self.cx, self.cy, self.tx[:, :, :4])


if __name__ == "__main__":
    unittest.main()
