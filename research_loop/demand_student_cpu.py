"""R74 small, serializable CPU ridge heads and response priors.

These are PAIEC adaptations, not the ADeLe author's RandomForest assessor.
Only existing numeric item features enter these models. IDs never do.
"""
import numpy as np


def fit_ridge(x, y, weights=None, alpha=.1, divisor=None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    mu = w @ x
    std = np.maximum(np.sqrt(w @ ((x - mu) ** 2)), .05)
    divisor = np.full(x.shape[1], np.sqrt(x.shape[1])) if divisor is None else np.asarray(divisor, float)
    z = (x - mu) / std / divisor
    intercept = float(w @ y)
    a = z * np.sqrt(w[:, None])
    target = (y - intercept) * np.sqrt(w)
    if len(x) < x.shape[1]:
        coefficient = a.T @ np.linalg.solve(a @ a.T + alpha * np.eye(len(x)), target)
    else:
        coefficient = np.linalg.solve(a.T @ a + alpha * np.eye(x.shape[1]), a.T @ target)
    return dict(mean=mu, std=std, divisor=divisor, coefficient=coefficient,
                intercept=np.array(intercept), alpha=np.array(alpha))


def predict_ridge(model, x, lower=0., upper=1.):
    return np.clip(((np.asarray(x, float) - model['mean']) / model['std'] /
                    model['divisor']) @ model['coefficient'] + model['intercept'], lower, upper)


def predict_demands(models, x):
    return np.column_stack([predict_ridge(m, x, 0., 5.) for m in models])


def response_features(x, demand=None, only_demand=False):
    z = np.zeros_like(x) if only_demand else x
    return np.column_stack([z, np.zeros((len(x), 4)) if demand is None else demand])


def fit_response(x, y, weight):
    return fit_ridge(x, y, weight, alpha=.01,
                     divisor=np.r_[np.full(640, np.sqrt(640)), np.full(4, 2.)])


def predict_segment(model, support_x, query_x, support_d, query_d, labels,
                    r25, mode='augmented'):
    only = mode == 'demand_only'
    no_demand = mode == 'base'
    q = predict_ridge(model, response_features(query_x, None if no_demand else query_d, only))
    a = predict_ridge(model, response_features(support_x[:1], None if no_demand else support_d[:1], only))[0]
    output = np.array(r25, copy=True)
    output[:, 0] = q
    output[:, 1] = np.clip(q + (labels[0] - a) / 3., 0., 1.)
    return output


def load_demand_models(folder):
    """Load the four saved fit-split heads for new 640-coordinate item inputs."""
    from pathlib import Path
    return [dict(np.load(Path(folder) / f'student_{r}.npz', allow_pickle=False))
            for r in ['AS', 'MCr', 'QLl', 'KNf']]
