"""Mixed-factor GPR + Bernoulli GPC, selected by EI * P(feasible).

Lengths are normalized; motor labels use equality, never numeric distance.
Optuna is only the durable trial ledger. No failed objective or penalty is used
to fit the performance GP. GPC uses scikit-learn's logistic Laplace approximation.
"""
from __future__ import annotations

from itertools import product
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, Matern, ConstantKernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning


class MixedKernel(Kernel):
    """Matérn-5/2 on two lengths, multiplied by a motor Hamming kernel."""
    def __init__(self, length_scale=(0.25, 0.25), motor_penalty=(1.0, 1.0),
                 length_scale_bounds=(0.03, 3.0), motor_penalty_bounds=(0.05, 5.0)):
        self.length_scale = length_scale
        self.motor_penalty = motor_penalty
        self.length_scale_bounds = length_scale_bounds
        self.motor_penalty_bounds = motor_penalty_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds, 2)

    @property
    def hyperparameter_motor_penalty(self):
        return Hyperparameter("motor_penalty", "numeric", self.motor_penalty_bounds, 2)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=float)
        same = Y is None
        Y = X if same else np.asarray(Y, dtype=float)
        if eval_gradient and not same:
            raise ValueError("kernel gradients require Y=None")
        continuous = Matern(self.length_scale, self.length_scale_bounds, nu=2.5)
        if eval_gradient:
            kc, grad = continuous(X[:, :2], eval_gradient=True)
        else:
            kc = continuous(X[:, :2], Y[:, :2])
        different = (X[:, None, 2:] != Y[None, :, 2:]).astype(float)
        penalties = np.asarray(self.motor_penalty)
        km = np.exp(-np.sum(different * penalties, axis=-1))
        k = kc * km
        if not eval_gradient:
            return k
        return k, np.concatenate((grad * km[..., None],
                                  -k[..., None] * different * penalties), axis=-1)

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return True


def key(params):
    return (round(float(params["thigh_m"]), 10), round(float(params["calf_m"]), 10),
            str(params["knee_motor"]), str(params["ankle_motor"]))


class ConstrainedSurrogate:
    def __init__(self, ranges, knees, ankles, *, seed=42, optimize=True):
        self.ranges = ranges
        self.knees, self.ankles = list(knees), list(ankles)
        self.seed, self.optimize = int(seed), optimize
        self.regressor = self.classifier = None

    def encode(self, params):
        encoded = []
        for p in params:
            row = []
            for field in ("thigh", "calf"):
                lo, hi = self.ranges[field]
                value = float(p[field + "_m"])
                if not np.isfinite(value) or not lo <= value <= hi or hi <= lo:
                    raise ValueError("design outside continuous search bounds")
                row.append((value - lo) / (hi - lo))
            row.extend((self.knees.index(p["knee_motor"]), self.ankles.index(p["ankle_motor"])))
            encoded.append(row)
        return np.asarray(encoded, dtype=float).reshape(-1, 4)

    def decode(self, X):
        rows = []
        for x in X:
            row = {field + "_m": float(self.ranges[field][0] + x[i] *
                   (self.ranges[field][1] - self.ranges[field][0]))
                   for i, field in enumerate(("thigh", "calf"))}
            row.update(knee_motor=self.knees[int(x[2])], ankle_motor=self.ankles[int(x[3])])
            rows.append(row)
        return rows

    def fit(self, observations):
        self.observations = list(observations)
        if not observations:
            raise ValueError("constrained GP requires evaluated observations")
        self.X = self.encode([o["params"] for o in observations])
        labels = np.array([o["feasible"] for o in observations])
        if any(type(o["feasible"]) is not bool for o in observations):
            raise ValueError("feasibility labels must be booleans")
        self.feasible_count = int(labels.sum())
        self.feasibility_count = len(labels)
        self.regressor = self.classifier = None
        self.best = None
        self.fit_warnings = []
        optimizer = "fmin_l_bfgs_b" if self.optimize else None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if self.feasible_count:
                y = np.array([o["mae"] for o in observations if o["feasible"]], dtype=float)
                if not np.isfinite(y).all() or (y < 0).any():
                    raise ValueError("successful observations require finite nonnegative MAE")
                self.best = float(y.min())
                self.regressor = GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0, (0.05, 20.0)) * MixedKernel()
                           + WhiteKernel(1e-3, (1e-6, 0.2)),
                    normalize_y=True, alpha=1e-8, optimizer=optimizer, random_state=self.seed)
                self.regressor.fit(self.X[labels], y)
            if 0 < self.feasible_count < len(labels):
                self.classifier = GaussianProcessClassifier(
                    kernel=ConstantKernel(2.0, (0.1, 30.0)) * MixedKernel(),
                    optimizer=optimizer, random_state=self.seed, max_iter_predict=100)
                self.classifier.fit(self.X, labels.astype(int))
            self.fit_warnings = [str(w.message) for w in caught]
        # One-class histories cannot fit a binary classifier. A Beta(1,1)
        # posterior is explicit uncertainty, not a claim of 100% feasibility.
        self.constant_probability = (self.feasible_count + 1) / (len(labels) + 2)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float).reshape(-1, 4)
        p = (self.classifier.predict_proba(X)[:, 1] if self.classifier is not None
             else np.full(len(X), self.constant_probability))
        if self.regressor is None:
            return np.full(len(X), np.nan), np.full(len(X), np.nan), p
        mean, std = self.regressor.predict(X, return_std=True)
        return mean, np.maximum(std, 0), p

    def acquisition(self, X):
        mean, std, p = self.predict(X)
        if self.regressor is None:
            return p
        improvement = self.best - mean
        z = np.divide(improvement, std, out=np.zeros_like(mean), where=std > 1e-12)
        ei = improvement * ndtr(z) + std * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        ei = np.where(std > 1e-12, ei, np.maximum(improvement, 0))
        return np.maximum(ei, 0) * p

    def propose(self, *, pool_power=8, refine=True):
        # Enumerate every motor pair. Sobol provides continuous candidate seeds,
        # not a fixed training grid; optional local optimization refines them.
        seed = self.seed + len(self.observations)
        points = qmc.Sobol(2, scramble=True, seed=seed).random_base2(pool_power)
        points = np.vstack((points, [[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]]))
        pairs = list(product(range(len(self.knees)), range(len(self.ankles))))
        pool = np.vstack([np.column_stack((points, np.tile(pair, (len(points), 1)))) for pair in pairs])
        seen = {key(o["params"]) for o in self.observations}
        pool = np.array([x for x, p in zip(pool, self.decode(pool)) if key(p) not in seen])
        if not len(pool):
            raise ValueError("no unevaluated candidate in acquisition pool")
        if refine and self.regressor is not None:
            extras = []
            scores = self.acquisition(pool)
            for pair in pairs:
                ix = np.where((pool[:, 2:] == pair).all(axis=1))[0]
                if not len(ix):
                    continue
                start = pool[ix[np.argmax(scores[ix])], :2]
                result = minimize(lambda u: -float(self.acquisition([[*u, *pair]])[0]),
                                  start, bounds=[(0, 1), (0, 1)], method="L-BFGS-B",
                                  options={"maxiter": 25})
                candidate = np.array([*result.x, *pair])
                if np.isfinite(candidate).all() and key(self.decode([candidate])[0]) not in seen:
                    extras.append(candidate)
            if extras:
                pool = np.vstack((pool, extras))
        scores = self.acquisition(pool)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite constrained acquisition")
        mode = "EI_times_probability_feasible"
        if self.regressor is None or scores.max() <= 1e-16:
            # No feasible incumbent, or EI has numerically vanished: explicitly
            # seek feasibility/uncertainty rather than silently drawing at random.
            distance = np.min(np.sum((pool[:, None, :2] - self.X[None, :, :2])**2, axis=-1)
                              + np.sum(pool[:, None, 2:] != self.X[None, :, 2:], axis=-1), axis=1)
            _, std, p = self.predict(pool)
            scores = p * (1 + distance) if self.regressor is None else p * (std + 1e-12)
            mode = "feasibility_exploration" if self.regressor is None else "EI_underflow_uncertainty_fallback"
        index = int(np.argmax(scores))
        mean, std, probability = self.predict(pool[index:index + 1])
        audit = {
            "method": mode, "seed": seed, "candidate_pool_size": len(pool),
            "performance_training_count": self.feasible_count,
            "feasibility_training_count": self.feasibility_count,
            "classifier": "logistic_GPC_Laplace" if self.classifier is not None else "one_class_Beta_prior",
            "best_feasible_mae_mps": self.best,
            "predicted_mae_mps": float(mean[0]) if self.regressor is not None else None,
            "predicted_mae_std_mps": float(std[0]) if self.regressor is not None else None,
            "predicted_feasibility": float(probability[0]), "acquisition_value": float(scores[index]),
            "performance_kernel": str(self.regressor.kernel_) if self.regressor is not None else None,
            "feasibility_kernel": str(self.classifier.kernel_) if self.classifier is not None else None,
            "fit_warnings": self.fit_warnings,
        }
        return self.decode(pool[index:index + 1])[0], audit
