# Copyright 2018-2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Bernoulli CNP and deterministic cross-attentive CNP for PAIEC episodes.

Architecture ported from google-deepmind/neural-processes, commit
aca7e38ea64b718fbd7f311ccae5d51d73447d15 (Apache-2.0), specifically
DeterministicEncoder/Decoder, batch_mlp and normalised dot_product_attention.
See the retained upstream LICENSE and source mapping in
research_loop/literature/reproduction_v4/cnp_anp.md.

Changes: PyTorch, Bernoulli logits instead of Gaussian regression parameters,
smaller configurable widths, explicit empty context, and state_dict export.
AttentiveCNP ports only the deterministic cross-attention path: it has no
self-attention, latent variable or ELBO and is NOT the complete latent ANP.
"""

from pathlib import Path

import torch
from torch import Tensor, nn


def _mlp(sizes: list[int]) -> nn.Sequential:
    """Pointwise MLP with ReLU on every layer except the last, as upstream."""
    layers = []
    for index, (in_size, out_size) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(in_size, out_size))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def dot_product_attention(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    """ANP section 2.2: softmax(Q K^T / sqrt(d_k)) V, over context keys."""
    weights = torch.softmax(
        (queries @ keys.transpose(-1, -2)) / (queries.shape[-1] ** 0.5),
        dim=-1,
    )
    return weights @ values


class CNP(nn.Module):
    """CNP equations (1)-(3) with a Bernoulli target likelihood.

    Every batch element is an independent episode with the same context count.
    Inputs are context_x[B,N,D], context_y[B,N], target_x[B,Q,D]. There is no
    target-label input and no interaction between target queries. An empty
    context uses a fixed zero representation (ANP section 2.1).
    """

    model_type = "cnp"

    def __init__(
        self, input_dim: int = 640, hidden_dim: int = 64,
        representation_dim: int = 64,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, representation_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.representation_dim = representation_dim
        self.context_encoder = _mlp(
            [input_dim + 1, hidden_dim, hidden_dim, hidden_dim, representation_dim]
        )
        self.decoder = _mlp(
            [representation_dim + input_dim, hidden_dim, hidden_dim, 1]
        )

    def _check_shapes(self, context_x: Tensor, context_y: Tensor, target_x: Tensor):
        if context_x.ndim != 3 or target_x.ndim != 3 or context_y.ndim != 2:
            raise ValueError("Expected context_x[B,N,D], context_y[B,N], target_x[B,Q,D]")
        if context_x.shape[:2] != context_y.shape:
            raise ValueError("Context labels must match the context batch and item counts")
        if context_x.shape[0] != target_x.shape[0]:
            raise ValueError("Context and target batch sizes must agree")
        if context_x.shape[-1] != self.input_dim or target_x.shape[-1] != self.input_dim:
            raise ValueError("Input feature dimension does not match model input_dim")

    def _aggregate(self, context_x: Tensor, target_x: Tensor, values: Tensor) -> Tensor:
        return values.mean(dim=1, keepdim=True).expand(-1, target_x.shape[1], -1)

    def context_representation(
        self, context_x: Tensor, context_y: Tensor, target_x: Tensor,
    ) -> Tensor:
        self._check_shapes(context_x, context_y, target_x)
        if context_x.shape[1] == 0:
            return target_x.new_zeros(
                (target_x.shape[0], target_x.shape[1], self.representation_dim)
            )
        labels = context_y.to(device=context_x.device, dtype=context_x.dtype)
        values = self.context_encoder(torch.cat([context_x, labels.unsqueeze(-1)], dim=-1))
        return self._aggregate(context_x, target_x, values)

    def predict_logits(self, context_x: Tensor, context_y: Tensor, target_x: Tensor) -> Tensor:
        representation = self.context_representation(context_x, context_y, target_x)
        # The upstream decoder also concatenates representation before target_x.
        return self.decoder(torch.cat([representation, target_x], dim=-1)).squeeze(-1)

    def forward(self, context_x: Tensor, context_y: Tensor, target_x: Tensor) -> Tensor:
        return torch.sigmoid(self.predict_logits(context_x, context_y, target_x))

    def export_config(self) -> dict:
        return {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "representation_dim": self.representation_dim,
        }


class AttentiveCNP(CNP):
    """CNP with ANP's deterministic learned dot-product cross-attention.

    Keys and queries share an MLP, matching upstream Attention(rep='mlp',
    att_type='dot_product', normalise=True). All context values retain their
    paired labels. There is no target self-attention or latent path.
    """

    model_type = "attentive_cnp"

    def __init__(
        self, input_dim: int = 640, hidden_dim: int = 64,
        representation_dim: int = 64,
    ):
        super().__init__(input_dim, hidden_dim, representation_dim)
        self.key_query_encoder = _mlp([input_dim, hidden_dim, representation_dim])

    def _aggregate(self, context_x: Tensor, target_x: Tensor, values: Tensor) -> Tensor:
        keys = self.key_query_encoder(context_x)
        queries = self.key_query_encoder(target_x)
        return dot_product_attention(queries, keys, values)


def build_neural_process(config: dict) -> CNP:
    options = dict(config)
    model_type = options.pop("model_type")
    classes = {"cnp": CNP, "attentive_cnp": AttentiveCNP}
    if model_type not in classes:
        raise ValueError(f"Unsupported neural process model_type: {model_type}")
    return classes[model_type](**options)


def save_neural_process(model: CNP, path: str | Path) -> None:
    """Save portable CPU tensors plus explicit constructor parameters."""
    torch.save({
        "format_version": 1,
        "config": model.export_config(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }, path)


def load_neural_process(path: str | Path, device: str = "cpu") -> CNP:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1:
        raise ValueError("Unsupported neural process export format")
    model = build_neural_process(artifact["config"])
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model.to(device).eval()
