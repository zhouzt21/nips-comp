"""Content-based Rasch predictors and the reeval calibration objective.

Source: Truong et al. (ICML 2025), reeval commit 726cfe1c, calibration.ipynb
code cell 15 and cat/adap_test.py.  The notebook minimizes expected negative
log likelihood over 150 fixed standard-normal ability samples; this is NOT
negative log marginal likelihood.  See the literature note for that distinction.

PAIEC adaptations are explicit: frozen MiniLM/metadata features, feature-based
subject cold start, preserved repeated responses, and a Gaussian MAP update
using only the supplied support.  This does not reproduce Agent Psychometrics'
two-stage SVI/ridge fit or its categorical LLM/scaffold ability decomposition.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_nuisance_abilities(num_subjects: int, *, seed: int = 0,
                           samples: int = 150) -> Tensor:
    """Draw once per fit and reuse each subject's draws across all its items."""
    if num_subjects < 1 or samples < 1:
        raise ValueError("num_subjects and samples must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(samples, num_subjects, generator=generator)


def reeval_expected_nll(difficulty: Tensor, labels: Tensor,
                       subject_index: Tensor, nuisance_abilities: Tensor,
                       *, grid_size: int | None = None) -> Tensor:
    """Sparse-row equivalent of the author's masked expected Bernoulli NLL.

    Inputs are one row per observed response (repeats retained).  Set grid_size
    to subjects * items to reproduce a dense masked matrix's denominator.
    The default averages observed rows, an objective-proportional denominator
    change when the original mask is fixed.  No query/held-out labels belong
    here.  Batch selection and fit-only data separation are caller responsibilities.
    """
    if (difficulty.ndim != 1 or labels.shape != difficulty.shape
            or subject_index.shape != difficulty.shape or not difficulty.numel()):
        raise ValueError("difficulty, labels and subject_index must be nonempty [R]")
    if nuisance_abilities.ndim != 2 or not nuisance_abilities.numel():
        raise ValueError("nuisance_abilities must have shape [samples, subjects]")
    if subject_index.dtype != torch.long:
        raise ValueError("subject_index must be torch.long")
    if bool((subject_index < 0).any()) or bool((subject_index >= nuisance_abilities.shape[1]).any()):
        raise ValueError("subject_index is outside nuisance_abilities")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be binary responses")
    return reeval_expected_nll_counts(difficulty, torch.ones_like(labels), labels,
                                     subject_index, nuisance_abilities, grid_size=grid_size)


def reeval_expected_nll_counts(difficulty: Tensor, counts: Tensor,
                              successes: Tensor, subject_index: Tensor,
                              nuisance_abilities: Tensor, *, grid_size: int | None = None
                              ) -> Tensor:
    """Equivalent expected NLL after lossless grouping of repeated responses.

    Each row is one subject/item with integer count and integer successes.
    This retains Bernoulli contributions, rather than replacing repeats by
    a majority-vote label.  Nuisance samples remain shared by subject.
    """
    if (difficulty.ndim != 1 or not difficulty.numel()
            or counts.shape != difficulty.shape or successes.shape != difficulty.shape
            or subject_index.shape != difficulty.shape or subject_index.dtype != torch.long):
        raise ValueError("expected aligned nonempty vectors and long subject_index")
    if nuisance_abilities.ndim != 2 or not nuisance_abilities.numel():
        raise ValueError("nuisance_abilities must have shape [samples, subjects]")
    if bool((subject_index < 0).any()) or bool((subject_index >= nuisance_abilities.shape[1]).any()):
        raise ValueError("subject_index is outside nuisance_abilities")
    if (not bool(torch.isfinite(counts).all()) or not bool(torch.isfinite(successes).all())
            or bool((counts < 1).any()) or bool((successes < 0).any())
            or bool((successes > counts).any())
            or bool((counts != counts.round()).any()) or bool((successes != successes.round()).any())):
        raise ValueError("counts/successes must be valid nonempty integer response counts")
    denominator = counts.sum() if grid_size is None else grid_size
    if denominator < counts.sum():
        raise ValueError("grid_size must cover every observed response")
    logits = nuisance_abilities[:, subject_index] - difficulty.unsqueeze(0)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("ability samples and difficulties must be finite")
    # Stable even when |logit| is large; algebraically sum Bernoulli BCE.
    losses = F.softplus(-logits) * successes + F.softplus(logits) * (counts - successes)
    return losses.sum() / (nuisance_abilities.shape[0] * denominator)


@torch.no_grad()
def fit_training_abilities(difficulty: Tensor, labels: Tensor,
                           subject_index: Tensor, num_subjects: int,
                           *, prior_precision: float = 0.0
                           ) -> tuple[Tensor, Tensor]:
    """MLE abilities after freezing source-fit item difficulties.

    The root is sum_j(sigmoid(theta - difficulty_j) - y_j) = 0.
    A subject with no labels or only one class has no finite unique MLE;
    it receives NaN and false in the returned mask, and must be excluded from
    feature-to-ability regression.  Optional positive prior_precision instead
    fits a finite Gaussian MAP around zero (a PAIEC stabilization adaptation);
    its returned mask then marks subjects having any training response.
    This exposes an issue hidden by stopping
    the original unconstrained LBFGS after a finite number of iterations.
    """
    if num_subjects < 1 or difficulty.ndim != 1 or labels.shape != difficulty.shape:
        raise ValueError("expected positive num_subjects and aligned response vectors")
    if subject_index.shape != labels.shape or subject_index.dtype != torch.long:
        raise ValueError("subject_index must be an aligned torch.long vector")
    if bool((subject_index < 0).any()) or bool((subject_index >= num_subjects).any()):
        raise ValueError("subject index outside training set")
    if not bool(torch.isfinite(difficulty).all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("finite difficulty and binary responses are required")
    return fit_training_abilities_counts(difficulty, torch.ones_like(labels), labels,
                                         subject_index, num_subjects,
                                         prior_precision=prior_precision)


@torch.no_grad()
def fit_training_abilities_counts(difficulty: Tensor, counts: Tensor, successes: Tensor,
                                  subject_index: Tensor, num_subjects: int,
                                  *, prior_precision: float = 0.0
                                  ) -> tuple[Tensor, Tensor]:
    """Count-preserving version of fit_training_abilities; see its MAP caveat."""
    if (num_subjects < 1 or difficulty.ndim != 1 or counts.shape != difficulty.shape
            or successes.shape != difficulty.shape or subject_index.shape != difficulty.shape
            or subject_index.dtype != torch.long):
        raise ValueError("expected aligned vectors and positive num_subjects")
    if (not bool(torch.isfinite(difficulty).all()) or not bool(torch.isfinite(counts).all())
            or not bool(torch.isfinite(successes).all()) or bool((counts < 1).any())
            or bool((successes < 0).any()) or bool((successes > counts).any())
            or bool((counts != counts.round()).any()) or bool((successes != successes.round()).any())):
        raise ValueError("expected finite difficulty and valid integer response counts")
    if bool((subject_index < 0).any()) or bool((subject_index >= num_subjects).any()):
        raise ValueError("subject index outside training set")
    if not math.isfinite(prior_precision) or prior_precision < 0:
        raise ValueError("prior_precision must be finite and nonnegative")
    work = difficulty.to(torch.float64)
    n = counts.to(dtype=work.dtype, device=work.device)
    y = successes.to(dtype=work.dtype, device=work.device)
    count = work.new_zeros(num_subjects).index_add_(0, subject_index, n)
    wins = work.new_zeros(num_subjects).index_add_(0, subject_index, y)
    finite = (wins > 0) & (wins < count) if prior_precision == 0 else count > 0
    theta = work.new_full((num_subjects,), float("nan"))
    if not bool(finite.any()):
        return theta.to(difficulty.dtype), finite
    if prior_precision:
        low, high = -count / prior_precision, count / prior_precision
    else:
        low = work.new_full((num_subjects,), float(work.min()) - 40.0)
        high = work.new_full((num_subjects,), float(work.max()) + 40.0)
    for _ in range(64):
        mid = (low + high) / 2
        residual = torch.sigmoid(mid[subject_index] - work) * n - y
        gradient = work.new_zeros(num_subjects).index_add_(0, subject_index, residual)
        gradient = gradient + prior_precision * mid
        low = torch.where(gradient < 0, mid, low)
        high = torch.where(gradient >= 0, mid, high)
    theta[finite] = ((low + high) / 2)[finite]
    return theta.to(difficulty.dtype), finite


class AmortizedIRT(nn.Module):
    """Linear content difficulty + feature-conditioned ability, then support MAP.

    For the joint BCE control leave difficulty_bias=False and center item
    features on FIT data: mean fitted difficulty is fixed to zero, removing
    the additive ability/difficulty gauge.  For source calibration set
    difficulty_bias=True: fixed N(0,1) nuisance samples anchor its scale and
    location; freeze difficulty before fitting the subject ability head.

    Training uses B0 only.  Nonempty support is an inference-only one-dimensional
    MAP estimate with precision 1, centered on the predicted subject ability.
    Unit precision is a predeclared PAIEC choice motivated by the original
    standard-normal ability convention, not the author's fitted online prior.
    """

    def __init__(self, item_dim: int = 640, subject_dim: int = 256,
                 *, difficulty_bias: bool = False, prior_precision: float = 1.0):
        super().__init__()
        if item_dim < 1 or subject_dim < 1:
            raise ValueError("feature dimensions must be positive")
        if not math.isfinite(prior_precision) or prior_precision <= 0:
            raise ValueError("prior_precision must be finite and positive")
        self.item_dim = item_dim
        self.subject_dim = subject_dim
        self.difficulty = nn.Linear(item_dim, 1, bias=difficulty_bias)
        self.ability = nn.Linear(subject_dim, 1)
        self.register_buffer("prior_precision", torch.tensor(float(prior_precision)))

    def _validate(self, context_x: Tensor, context_y: Tensor, target_x: Tensor,
                  subject: Tensor) -> None:
        if (context_x.ndim != 3 or target_x.ndim != 3
                or context_x.shape[0] != target_x.shape[0]
                or context_x.shape[-1] != self.item_dim
                or target_x.shape[-1] != self.item_dim):
            raise ValueError("context_x and target_x must be [B,N,D] / [B,Q,D]")
        if context_y.shape != context_x.shape[:2]:
            raise ValueError("context_y must have shape [B,N]")
        if subject.shape != (target_x.shape[0], self.subject_dim):
            raise ValueError("subject must have shape [B,S]")
        if context_x.shape[1] > 31:
            raise ValueError("PAIEC support budget cannot exceed 31 responses")
        if not bool(((context_y == 0) | (context_y == 1)).all()):
            raise ValueError("context_y must contain binary responses")
        if any(not bool(torch.isfinite(x).all()) for x in (context_x, target_x, subject)):
            raise ValueError("all input features must be finite")

    @torch.no_grad()
    def _map_ability(self, context_difficulty: Tensor, context_y: Tensor,
                     prior: Tensor) -> Tensor:
        # Solve the strictly increasing derivative in offset coordinates.
        # |sum(p-y)| <= N implies the root is bracketed by +/- N/precision.
        dtype = prior.dtype
        prior64 = prior.to(torch.float64)
        difficulty64 = context_difficulty.to(torch.float64)
        labels64 = context_y.to(torch.float64)
        precision = self.prior_precision.to(torch.float64)
        radius = context_y.shape[1] / precision
        low, high = -torch.ones_like(prior64) * radius, torch.ones_like(prior64) * radius
        for _ in range(48):
            offset = (low + high) / 2
            probability = torch.sigmoid(prior64[:, None] + offset[:, None] - difficulty64)
            gradient = (probability - labels64).sum(-1) + precision * offset
            low = torch.where(gradient < 0, offset, low)
            high = torch.where(gradient >= 0, offset, high)
        return (prior64 + (low + high) / 2).to(dtype)

    def predict_logits(self, context_x: Tensor, context_y: Tensor,
                       target_x: Tensor, subject: Tensor) -> Tensor:
        self._validate(context_x, context_y, target_x, subject)
        prior = self.ability(subject).squeeze(-1)
        if context_x.shape[1]:
            if self.training:
                raise ValueError("train with empty B0 support; call eval() for support MAP")
            prior = self._map_ability(self.difficulty(context_x).squeeze(-1), context_y, prior)
        return prior[:, None] - self.difficulty(target_x).squeeze(-1)

    def forward(self, context_x: Tensor, context_y: Tensor,
                target_x: Tensor, subject: Tensor) -> Tensor:
        return torch.sigmoid(self.predict_logits(context_x, context_y, target_x, subject))
