"""Few-shot pixel classifier for frozen per-voxel features.

`FewShotClassifier` is a classifier for the `SegmentationModel` protocol. It
is meant for very few painted voxels on top of frozen, pretrained features.
It combines two predictors and averages their class probabilities:

- a small MLP, ``Linear(D, 256) - GELU - Dropout(0.2) - Linear(256, C)``,
  trained with class-weighted cross-entropy and AdamW for a fixed number of
  minibatch steps;
- a k-nearest-neighbour vote over a bank of L2-normalized training rows
  (cosine similarity, top 16). Each vote is weighted by the inverse of its
  class's share of the bank, so a sparsely painted class is not outvoted.

The MLP generalizes smoothly in feature space. The kNN vote follows the
painted examples closely. The defaults are the configuration used in our
few-shot cryo-ET experiments.

Labels
------
Painted labels store 0 for unannotated voxels and k > 0 for class k.
`DataManager.get_training_data` drops the unannotated voxels and subtracts
one, so `SemanticSegmentationManager.fit` passes zero-based classes and no
unannotated rows. `fit` also accepts unannotated rows: rows whose label
equals ``unlabelled`` (default -1, i.e. a stored 0 after the shift by one)
are ignored. `predict` returns values from ``classes_``, the labels seen
in `fit`, so the usual "add one before saving" rule still applies.

Features
--------
Rows are used as given; the classifier does not rescale them. It expects
features standardized per channel over the whole volume (zero mean, unit
variance), as is usual for frozen network features. Statistics of the painted
rows alone would be a biased sample of the volume.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Query rows per similarity block in the kNN vote. A block of 1024 rows
# against a bank of 60 000 rows takes about 0.25 GB (float32).
_KNN_BLOCK = 1024


def _gelu_cdf(a: np.ndarray) -> np.ndarray:
    """Standard normal CDF, the gate of the exact GELU."""
    return 0.5 * (1.0 + erf(a / math.sqrt(2.0)))


def _gelu(a: np.ndarray) -> np.ndarray:
    """Exact (erf) GELU, as `torch.nn.GELU()`."""
    return a * _gelu_cdf(a)


def _gelu_grad(a: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Derivative of the exact GELU, given its gate ``cdf = _gelu_cdf(a)``."""
    pdf = np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    return cdf + a * pdf


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    z = logits - logits.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (zero rows stay zero)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class FewShotClassifier(ClassifierMixin, BaseEstimator):
    """MLP and class-balanced kNN vote on frozen features, averaged.

    The class satisfies the `SegmentationModel` protocol (``fit(X, y)``,
    ``predict(X)``) and follows the scikit-learn estimator conventions.
    It needs only NumPy and SciPy and runs on the CPU.

    Cost, for 414 features on 8 CPU threads: fitting takes under a minute
    whatever the number of painted voxels. With the default bank of 20 000
    rows, prediction takes about 0.1 ms per voxel, almost all of it in the
    kNN vote; a smaller ``max_bank_size`` reduces this proportionally.

    Fast mode: ``knn_weight=0`` uses the MLP alone and predicts about 50
    times faster (under 10 microseconds per voxel). On our cryo-ET benchmark
    it is as good or slightly better for large regions, but loses small
    particles and thin filaments, which the kNN vote keeps.

    Parameters
    ----------
    hidden : int
        Width of the MLP's hidden layer.
    dropout : float
        Dropout probability after the hidden layer, during training only.
    steps : int
        Number of AdamW steps. Every step draws a minibatch with replacement,
        so the training cost does not grow with the number of painted voxels.
    batch_size : int
        Rows per minibatch (fewer if fewer rows are painted).
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW decoupled weight decay.
    n_neighbors : int
        Number of most similar bank rows that vote.
    max_bank_size : int
        Maximum number of rows kept in the kNN bank. Larger training sets are
        subsampled uniformly, which keeps the painted class ratio. 20 000
        costs almost no accuracy against 60 000 and predicts about 3 times
        faster; 5 000 is a reasonable faster setting.
    knn_weight : float
        Weight of the kNN vote in the averaged probabilities; the MLP gets
        ``1 - knn_weight``. 0 gives the MLP alone (fast mode), 1 the kNN
        vote alone.
    unlabelled : int or None
        Label value of rows to ignore in `fit`. None uses every row.
    random_state : int or None
        Seed for weight initialization, minibatches, dropout and the bank
        subsample.

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted labels seen in `fit`, without ``unlabelled``. The columns of
        `predict_proba` follow this order.
    n_features_in_ : int
        Feature width seen in `fit`.
    """

    def __init__(
        self,
        hidden: int = 256,
        dropout: float = 0.2,
        steps: int = 400,
        batch_size: int = 8192,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-2,
        n_neighbors: int = 16,
        max_bank_size: int = 20000,
        knn_weight: float = 0.5,
        unlabelled: int | None = -1,
        random_state: int | None = 0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_neighbors = n_neighbors
        self.max_bank_size = max_bank_size
        self.knn_weight = knn_weight
        self.unlabelled = unlabelled
        self.random_state = random_state

    def fit(self, X, y) -> FewShotClassifier:
        """Fit the MLP and build the kNN bank.

        Parameters
        ----------
        X : array-like
            (n, d) feature rows.
        y : array-like
            (n,) integer labels; rows equal to ``unlabelled`` are ignored.

        Returns
        -------
        self : FewShotClassifier
            The fitted classifier.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Expected X with shape (n, d) and y with shape (n,)")
        if y.dtype.kind not in "iu":
            raise ValueError("Labels must be integers")
        if not 0.0 <= self.knn_weight <= 1.0:
            raise ValueError("knn_weight must be between 0 and 1")
        if self.unlabelled is not None:
            labelled = y != self.unlabelled
            X, y = X[labelled], y[labelled]
        if len(y) == 0:
            raise ValueError("No labelled rows to fit; paint at least one class")
        if not np.all(np.isfinite(X)):
            raise ValueError("Features must be finite")

        self.classes_, y_index = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        if len(self.classes_) < 2:
            # One class only: every row is predicted as that class.
            self.mlp_ = None
            self.bank_ = None
            return self
        self.mlp_ = self._fit_mlp(X, y_index, rng)
        self.bank_ = self._build_bank(X, y_index, rng)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, columns ordered as ``classes_``.

        Parameters
        ----------
        X : array-like
            (n, d) feature rows.

        Returns
        -------
        probabilities : np.ndarray
            (n, n_classes) float32 array; each row sums to one.
        """
        check_is_fitted(self, "classes_")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected X with shape (n, {self.n_features_in_}), got {X.shape}"
            )
        n_classes = len(self.classes_)
        if self.mlp_ is None:
            return np.ones((len(X), n_classes), dtype=np.float32)
        out = np.empty((len(X), n_classes), dtype=np.float32)
        for start in range(0, len(X), _KNN_BLOCK):
            block = X[start : start + _KNN_BLOCK]
            p = self._mlp_proba(block)
            if self.knn_weight > 0:
                p = (1 - self.knn_weight) * p + self.knn_weight * self._knn_proba(block)
            out[start : start + _KNN_BLOCK] = p
        return out

    def predict(self, X) -> np.ndarray:
        """Most probable class per row.

        Parameters
        ----------
        X : array-like
            (n, d) feature rows.

        Returns
        -------
        labels : np.ndarray
            (n,) labels taken from ``classes_``.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------ MLP
    def _fit_mlp(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Train the MLP with class-weighted cross-entropy and AdamW."""
        n, d = X.shape
        n_classes = len(self.classes_)
        params = {}
        # PyTorch's default nn.Linear initialization: U(-1/sqrt(fan_in), ...).
        for name, fan_in, shape in (
            ("w1", d, (d, self.hidden)),
            ("b1", d, (self.hidden,)),
            ("w2", self.hidden, (self.hidden, n_classes)),
            ("b2", self.hidden, (n_classes,)),
        ):
            bound = 1.0 / math.sqrt(fan_in)
            params[name] = rng.uniform(-bound, bound, shape).astype(np.float32)

        # Balanced class weights: each class contributes equally to the loss.
        counts = np.bincount(y, minlength=n_classes).astype(np.float32)
        class_weight = n / (counts * n_classes)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        batch = min(n, self.batch_size)
        keep = 1.0 - self.dropout
        for step in range(1, self.steps + 1):
            idx = rng.integers(0, n, batch)
            xb, yb = X[idx], y[idx]
            pre = xb @ params["w1"] + params["b1"]
            cdf = _gelu_cdf(pre)
            act = pre * cdf
            mask = (rng.random(act.shape) < keep).astype(np.float32) / keep
            hid = act * mask
            prob = _softmax(hid @ params["w2"] + params["b2"])

            # Gradient of the weighted mean cross-entropy w.r.t. the logits.
            row_weight = class_weight[yb]
            grad = prob
            grad[np.arange(batch), yb] -= 1.0
            grad *= (row_weight / row_weight.sum())[:, None]
            grads = {"w2": hid.T @ grad, "b2": grad.sum(axis=0)}
            grad = (grad @ params["w2"].T) * mask * _gelu_grad(pre, cdf)
            grads["w1"] = xb.T @ grad
            grads["b1"] = grad.sum(axis=0)

            # AdamW (decoupled weight decay on all parameters, as torch.optim).
            lr = self.learning_rate
            for k, p in params.items():
                p *= 1.0 - lr * self.weight_decay
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1**step)
                v_hat = v[k] / (1 - beta2**step)
                p -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)
        return params

    def _mlp_proba(self, X: np.ndarray) -> np.ndarray:
        """MLP class probabilities (no dropout)."""
        p = self.mlp_
        return _softmax(_gelu(X @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"])

    # ------------------------------------------------------------------ kNN
    def _build_bank(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bank rows (L2-normalized), their class indices and vote weights."""
        if len(X) > self.max_bank_size:
            keep = np.sort(rng.choice(len(X), self.max_bank_size, replace=False))
            X, y = X[keep], y[keep]
        share = np.bincount(y, minlength=len(self.classes_)) / len(y)
        vote_weight = np.where(share > 0, 1.0 / np.maximum(share, 1e-9), 0.0)
        return _l2_normalize(X), y, vote_weight.astype(np.float32)

    def _knn_proba(self, X: np.ndarray) -> np.ndarray:
        """Class-balanced vote of the most similar bank rows (cosine)."""
        bank, bank_y, vote_weight = self.bank_
        k = min(self.n_neighbors, len(bank))
        sim = _l2_normalize(X) @ bank.T
        if k < len(bank):
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        else:
            top = np.broadcast_to(np.arange(len(bank)), sim.shape)
        votes = bank_y[top][:, :, None] == np.arange(len(self.classes_))
        frac = votes.mean(axis=1, dtype=np.float32) * vote_weight
        return frac / np.maximum(frac.sum(axis=1, keepdims=True), 1e-9)
