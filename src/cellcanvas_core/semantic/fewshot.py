"""Few-shot pixel classifier for frozen per-voxel features."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted, check_X_y

# Rows predicted at once; bounds the (block, bank) similarity matrix.
_BLOCK = 256


def _gelu_cdf(a):
    return 0.5 * (1.0 + erf(a / math.sqrt(2.0)))


def _l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _generator(random_state):
    """A NumPy Generator from an int, None, Generator or RandomState."""
    if isinstance(random_state, np.random.RandomState):
        random_state = random_state.randint(np.iinfo(np.int32).max)
    return np.random.default_rng(random_state)


class FewShotClassifier(ClassifierMixin, BaseEstimator):
    """MLP and class-balanced kNN vote on frozen features, averaged.

    Meant for a few painted voxels on features from a frozen, pretrained
    network, standardized per channel over the whole volume. The MLP
    (``Linear - GELU - Dropout - Linear``) is trained with class-weighted
    cross-entropy and AdamW for a fixed number of minibatch steps, so fitting
    does not slow down as more voxels are painted. The kNN vote compares each
    row with a bank of painted rows (cosine similarity) and weights each vote
    by the inverse of its class's share of the bank.

    The kNN vote dominates prediction time. ``knn_margin`` limits it to rows
    the MLP is unsure about, and ``knn_weight=0`` skips it.

    Parameters
    ----------
    hidden : int
        Width of the MLP's hidden layer.
    dropout : float
        Dropout probability after the hidden layer, during training only.
    steps : int
        Number of AdamW steps, each on a minibatch drawn with replacement.
    batch_size : int
        Rows per minibatch (fewer if fewer rows are painted).
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW decoupled weight decay.
    n_neighbors : int
        Number of most similar bank rows that vote.
    max_bank_size : int
        Maximum number of rows in the kNN bank. Larger classes are subsampled
        first, so a sparsely painted class keeps all of its rows.
    knn_weight : float
        Weight of the kNN vote; the MLP gets ``1 - knn_weight``.
    knn_margin : float or None
        The kNN vote runs only on rows where the MLP's two most probable
        classes differ by less than this. None runs it on every row.
    unlabelled : int or None
        Label of rows to ignore in `fit`. None uses every row.
    random_state : int, Generator, RandomState or None
        Seed for initialization, minibatches, dropout and the bank subsample.

    Attributes
    ----------
    classes_ : np.ndarray
        Labels seen in `fit`; the columns of `predict_proba` follow this order.
    n_features_in_ : int
        Number of features seen in `fit`.
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
        knn_margin: float | None = None,
        unlabelled: int | None = -1,
        random_state=0,
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
        self.knn_margin = knn_margin
        self.unlabelled = unlabelled
        self.random_state = random_state

    def fit(self, X, y):
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
        self._check_params()
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        X, y = check_X_y(X, y, dtype=np.float32)
        if y.dtype.kind not in "iu":
            raise ValueError("Labels must be integers")
        if self.unlabelled is not None:
            X, y = X[y != self.unlabelled], y[y != self.unlabelled]
        if len(y) == 0:
            raise ValueError("No labelled rows to fit; paint at least one class")

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.mlp_ = self.bank_ = None
        if len(self.classes_) < 2:
            return self
        rng = _generator(self.random_state)
        if self.knn_weight < 1:
            self.mlp_ = self._fit_mlp(X, y, rng)
        if self.knn_weight > 0:
            self.bank_ = self._build_bank(X, y, rng)
        return self

    def predict_proba(self, X):
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
        X = self._check_X(X)
        out = np.empty((len(X), len(self.classes_)), dtype=np.float32)
        for start in range(0, len(X), _BLOCK):
            out[start : start + _BLOCK] = self._proba(X[start : start + _BLOCK])
        return out

    def predict(self, X):
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
        X = self._check_X(X)
        out = np.empty(len(X), dtype=self.classes_.dtype)
        for start in range(0, len(X), _BLOCK):
            proba = self._proba(X[start : start + _BLOCK])
            out[start : start + _BLOCK] = self.classes_[proba.argmax(axis=1)]
        return out

    def _check_params(self):
        if not 0.0 <= self.knn_weight <= 1.0:
            raise ValueError("knn_weight must be between 0 and 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be at least 0 and below 1")
        for name in ("hidden", "steps", "batch_size", "n_neighbors", "max_bank_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")

    def _check_X(self, X):
        check_is_fitted(self, "classes_")
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected X with shape (n, {self.n_features_in_}), got {X.shape}"
            )
        return X

    def _proba(self, X):
        """Probabilities for one block of rows."""
        X = np.asarray(X, dtype=np.float32)
        if len(self.classes_) < 2:
            return np.ones((len(X), 1), dtype=np.float32)
        if self.mlp_ is None or self.knn_weight == 1:
            return self._knn_proba(X)
        p = self._mlp_proba(X)
        if self.bank_ is None or self.knn_weight == 0:
            return p
        rows = slice(None)
        if self.knn_margin is not None:
            top2 = np.partition(p, -2, axis=1)[:, -2:]
            rows = np.flatnonzero(top2[:, 1] - top2[:, 0] < self.knn_margin)
            if len(rows) == 0:
                return p
        w = self.knn_weight
        p[rows] = (1 - w) * p[rows] + w * self._knn_proba(X[rows])
        return p

    def _fit_mlp(self, X, y, rng):
        """Train the MLP with class-weighted cross-entropy and AdamW."""
        n, d = X.shape
        n_classes = len(self.classes_)
        # PyTorch's nn.Linear initialization: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        params = {}
        for name, fan_in, shape in (
            ("w1", d, (d, self.hidden)),
            ("b1", d, (self.hidden,)),
            ("w2", self.hidden, (self.hidden, n_classes)),
            ("b2", self.hidden, (n_classes,)),
        ):
            bound = 1.0 / math.sqrt(fan_in)
            params[name] = rng.uniform(-bound, bound, shape).astype(np.float32)
        class_weight = compute_class_weight(
            "balanced", classes=np.arange(n_classes), y=y
        ).astype(np.float32)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = {k: np.zeros_like(p) for k, p in params.items()}
        v = {k: np.zeros_like(p) for k, p in params.items()}
        batch = min(n, self.batch_size)
        keep = 1.0 - self.dropout
        lr = self.learning_rate
        for step in range(1, self.steps + 1):
            idx = rng.integers(0, n, batch)
            xb, yb = X[idx], y[idx]
            pre = xb @ params["w1"] + params["b1"]
            cdf = _gelu_cdf(pre)
            mask = (rng.random(pre.shape, dtype=np.float32) < keep) / np.float32(keep)
            hid = pre * cdf * mask
            prob = softmax(hid @ params["w2"] + params["b2"], copy=False)

            # Gradient of the weighted mean cross-entropy w.r.t. the logits.
            row_weight = class_weight[yb]
            grad = prob
            grad[np.arange(batch), yb] -= 1.0
            grad *= (row_weight / row_weight.sum())[:, None]
            grads = {"w2": hid.T @ grad, "b2": grad.sum(axis=0)}
            pdf = np.exp(-0.5 * pre * pre) / math.sqrt(2.0 * math.pi)
            grad = (grad @ params["w2"].T) * mask * (cdf + pre * pdf)
            grads["w1"] = xb.T @ grad
            grads["b1"] = grad.sum(axis=0)

            # AdamW, as torch.optim.AdamW (decoupled decay on all parameters).
            for k, p in params.items():
                p *= 1.0 - lr * self.weight_decay
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1**step)
                v_hat = v[k] / (1 - beta2**step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return params

    def _mlp_proba(self, X):
        p = self.mlp_
        pre = X @ p["w1"] + p["b1"]
        return softmax((pre * _gelu_cdf(pre)) @ p["w2"] + p["b2"], copy=False)

    def _build_bank(self, X, y, rng):
        """Bank rows (L2-normalized), their class indices and vote weights."""
        n_classes = len(self.classes_)
        counts = np.bincount(y, minlength=n_classes)
        if counts.sum() > self.max_bank_size:
            # Fill the bank from the smallest class up, each class taking at
            # most an equal share of the room left.
            quota = np.zeros(n_classes, dtype=int)
            left = self.max_bank_size
            for i, c in enumerate(np.argsort(counts, kind="stable")):
                quota[c] = min(counts[c], left // (n_classes - i))
                left -= quota[c]
            keep = np.concatenate(
                [
                    rng.choice(np.flatnonzero(y == c), quota[c], replace=False)
                    for c in range(n_classes)
                ]
            )
            keep.sort()
            X, y = X[keep], y[keep]
            counts = quota
        share = counts / counts.sum()
        vote_weight = np.where(share > 0, 1.0 / np.maximum(share, 1e-12), 0.0)
        return _l2_normalize(X), y, vote_weight.astype(np.float32)

    def _knn_proba(self, X):
        """Class-balanced vote of the most similar bank rows (cosine)."""
        bank, bank_y, vote_weight = self.bank_
        k = min(self.n_neighbors, len(bank))
        sim = _l2_normalize(X) @ bank.T
        if k < len(bank):
            top = np.argpartition(sim, -k, axis=1)[:, -k:]
        else:
            top = np.broadcast_to(np.arange(len(bank)), sim.shape)
        votes = bank_y[top][:, :, None] == np.arange(len(vote_weight))
        votes = votes.sum(axis=1, dtype=np.float32) * vote_weight
        return votes / np.maximum(votes.sum(axis=1, keepdims=True), 1e-9)
