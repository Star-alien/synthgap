"""Frozen-backbone linear probe (Sec. 3.3).

Multinomial logistic regression on frozen, L2-normalised DINOv2 [CLS] features.
Weight decay is selected on a held-out *synthetic* validation split by default;
selecting it on real data leaks real information and is only available behind
``learner.protocol: real_val_upper_bound``, where it is reported as an upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class LinearProbe:
    n_classes: int
    weight_decay_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    max_iter: int = 400
    loss: str = "logreg"
    val_fraction: float = 0.2
    seed: int = 0
    w: np.ndarray | None = None
    b: np.ndarray | None = None
    weight_decay: float | None = None
    history: list[dict] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg, n_classes: int) -> "LinearProbe":
        return cls(
            n_classes=int(n_classes),
            weight_decay_grid=tuple(float(x) for x in cfg.learner.weight_decay_grid),
            max_iter=int(cfg.learner.max_iter),
            loss=str(cfg.learner.loss),
            val_fraction=float(cfg.learner.synthetic_val_fraction),
        )

    # ------------------------------------------------------------------ internals
    def _split(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        train_idx, val_idx = [], []
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            n_val = int(round(self.val_fraction * len(idx)))
            if len(idx) >= 3:
                n_val = max(1, min(n_val, len(idx) - 1))
            else:
                n_val = 0
            rng.shuffle(idx)
            val_idx.append(idx[:n_val])
            train_idx.append(idx[n_val:])
        train_idx = np.concatenate(train_idx)
        val_idx = np.concatenate(val_idx) if any(len(v) for v in val_idx) else train_idx
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _solve_logreg(self, X: np.ndarray, y: np.ndarray, weight_decay: float):
        n, d = X.shape
        n_classes = self.n_classes
        onehot = np.zeros((n, n_classes))
        onehot[np.arange(n), y] = 1.0

        def objective(theta: np.ndarray):
            W = theta[: d * n_classes].reshape(d, n_classes)
            b = theta[d * n_classes :]
            logits = X @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            loss = float(-(onehot * log_probs).sum() / n + 0.5 * weight_decay * np.sum(W * W))
            grad = (np.exp(log_probs) - onehot) / n
            grad_w = X.T @ grad + weight_decay * W
            grad_b = grad.sum(axis=0)
            return loss, np.concatenate([grad_w.ravel(), grad_b])

        result = minimize(
            objective,
            np.zeros(d * n_classes + n_classes),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "maxcor": 20},
        )
        W = result.x[: d * n_classes].reshape(d, n_classes)
        b = result.x[d * n_classes :]
        return W, b

    def _solve_ridge(self, X: np.ndarray, y: np.ndarray, weight_decay: float):
        n, d = X.shape
        onehot = np.zeros((n, self.n_classes))
        onehot[np.arange(n), y] = 1.0
        Xb = np.hstack([X, np.ones((n, 1))])
        reg = np.eye(d + 1) * float(weight_decay)
        reg[-1, -1] = 0.0
        solution = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ onehot)
        return solution[:d], solution[d]

    def _solve(self, X: np.ndarray, y: np.ndarray, weight_decay: float):
        if self.loss == "ridge":
            return self._solve_ridge(X, y, weight_decay)
        return self._solve_logreg(X, y, weight_decay)

    @staticmethod
    def _accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
        if len(y) == 0:
            return float("nan")
        return float(np.mean((X @ w + b).argmax(axis=1) == y))

    # ------------------------------------------------------------------ public API
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        weight_decay: float | None = None,
        seed: int | None = None,
    ) -> "LinearProbe":
        if seed is not None:
            self.seed = int(seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if weight_decay is not None or self.loss == "ridge" and len(self.weight_decay_grid) == 1:
            grid = [float(weight_decay if weight_decay is not None else self.weight_decay_grid[0])]
        else:
            grid = list(self.weight_decay_grid)

        if X_val is None:
            X_tr, y_tr, X_val, y_val = self._split(X, y)
        else:
            X_tr, y_tr = X, np.asarray(y, dtype=int)
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=int)

        best = None
        for candidate in grid:
            w, b = self._solve(X_tr, y_tr, candidate)
            accuracy = self._accuracy(X_val, y_val, w, b)
            self.history.append({"weight_decay": candidate, "val_accuracy": accuracy})
            if best is None or (np.isfinite(accuracy) and accuracy > best[0]):
                best = (accuracy, candidate)
        self.weight_decay = float(best[1]) if best else float(grid[0])
        self.w, self.b = self._solve(X, y, self.weight_decay)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) @ self.w + self.b).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


def probe_accuracy(
    cfg,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    X_real_val: np.ndarray | None = None,
    y_real_val: np.ndarray | None = None,
    weight_decay: float | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Fit a probe on (X_train, y_train) and return (test accuracy, chosen weight decay)."""
    probe = LinearProbe.from_config(cfg, n_classes)
    protocol = str(cfg.learner.protocol)
    if protocol == "real_val_upper_bound" and X_real_val is not None and len(y_real_val or []) > 0:
        probe.fit(X_train, y_train, X_real_val, y_real_val, weight_decay=weight_decay, seed=seed)
    else:
        probe.fit(X_train, y_train, weight_decay=weight_decay, seed=seed)
    return probe.accuracy(X_test, y_test), float(probe.weight_decay or 0.0)
