"""Few-shot pixel classifier for frozen per-voxel features."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted, check_X_y

# Rows predicted at once, and the (rows x bank) cells of the kNN similarity
# matrix allowed per block: 16 MB on the CPU, 128 MB on the GPU.
_BLOCK_MAX = {"numpy": 8192, "torch": 16384}
_BLOCK_CELLS = {"numpy": 4_000_000, "torch": 32_000_000}


def _l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _generator(random_state):
    """A NumPy Generator from an int, None, Generator or RandomState."""
    if isinstance(random_state, np.random.RandomState):
        random_state = random_state.randint(np.iinfo(np.int32).max)
    return np.random.default_rng(random_state)


class _NumPy:
    """Array operations on NumPy arrays."""

    name = "numpy"
    float32 = np.float32
    erf, exp, sqrt = staticmethod(erf), staticmethod(np.exp), staticmethod(np.sqrt)

    def put(self, a):
        return a

    def get(self, a):
        return a

    def arange(self, n):
        return np.arange(n)

    def softmax(self, a):
        return softmax(a, copy=False)

    def normalize(self, a):
        return _l2_normalize(a)

    def top_k(self, a, k):
        return np.argpartition(a, -k, axis=1)[:, -k:]


class _Torch:
    """The same operations on PyTorch tensors on one device."""

    name = "torch"

    def __init__(self, device):
        import torch

        self.torch, self.device = torch, torch.device(device)
        self.float32 = torch.float32
        self.erf, self.exp, self.sqrt = torch.erf, torch.exp, torch.sqrt

    def put(self, a):
        return self.torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

    def get(self, a):
        return a.cpu().numpy()

    def arange(self, n):
        return self.torch.arange(n, device=self.device)

    def softmax(self, a):
        return self.torch.softmax(a, dim=1)

    def normalize(self, a):
        return self.torch.nn.functional.normalize(a, dim=1, eps=1e-12)

    def top_k(self, a, k):
        return a.topk(k, dim=1).indices


def _backend(device):
    """The array backend for a ``device`` setting."""
    if device == "numpy":
        return _NumPy()
    try:
        import torch
    except ImportError:
        if device == "auto":
            return _NumPy()
        raise ImportError(
            f"device={device!r} needs PyTorch: pip install cellcanvas-core[gpu]"
        ) from None
    if device == "auto":
        return _Torch("cuda") if torch.cuda.is_available() else _NumPy()
    return _Torch(device)


class FewShotClassifier(ClassifierMixin, BaseEstimator):
    """MLP and class-balanced kNN vote on frozen features, averaged.

    Meant for a few painted voxels on features from a frozen, pretrained
    network. Features are used as given, without scaling; the kNN vote
    compares L2-normalized rows. Labels follow
    `DataManager.get_training_data`: unannotated voxels are dropped there,
    `fit` also ignores rows labelled ``unlabelled``, and `predict` returns
    the labels seen in `fit`, so the "add one before saving" rule of
    `SemanticSegmentationManager` is unchanged. The MLP
    (``Linear - GELU - Dropout - Linear``) is trained with class-weighted
    cross-entropy and AdamW for a fixed number of minibatch steps, so fitting
    does not slow down as more voxels are painted. The kNN vote compares each
    row with a bank of painted rows (cosine similarity) and weights each vote
    by the inverse of its class's share of the bank.

    With PyTorch installed and a GPU available, fitting and prediction run on
    the GPU; otherwise they run on the CPU with NumPy. The kNN vote dominates
    CPU prediction time; ``knn_weight=0`` skips it.

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
        Maximum number of rows in the kNN bank, a uniform subsample of the
        painted rows in which every class keeps at least ``n_neighbors`` rows.
    knn_weight : float
        Weight of the kNN vote; the MLP gets ``1 - knn_weight``.
    unlabelled : int or None
        Label of rows to ignore in `fit`. None uses every row.
    device : str
        "auto" uses PyTorch on the GPU when both are available and NumPy
        otherwise; "numpy" always uses NumPy; any other value is a PyTorch
        device, such as "cuda:1" or "cpu".
    random_state : int, Generator, RandomState or None
        Seed for initialization, minibatches, dropout and the bank subsample.
        All random numbers are drawn with NumPy, so every device fits the
        same model up to floating-point rounding.

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
        unlabelled: int | None = -1,
        device: str = "auto",
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
        self.unlabelled = unlabelled
        self.device = device
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
        xp = _backend(self.device)
        X, y = np.asarray(X), np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.dtype.kind not in "iu":
            raise ValueError("Labels must be integers")
        if self.unlabelled is not None and X.ndim == 2 and len(X) == len(y):
            X, y = X[y != self.unlabelled], y[y != self.unlabelled]
            if len(y) == 0:
                raise ValueError("No labelled rows to fit; paint at least one class")
        X, y = check_X_y(X, y, dtype=np.float32)

        classes, y = np.unique(y, return_inverse=True)
        rng = _generator(self.random_state)
        mlp = bank = None
        if len(classes) > 1:
            if self.knn_weight < 1:
                mlp = self._fit_mlp(X, y, len(classes), rng, xp)
            if self.knn_weight > 0:
                bank = self._build_bank(X, y, len(classes), rng)
        self.classes_, self.n_features_in_ = classes, X.shape[1]
        self.mlp_, self.bank_ = mlp, bank
        self._on_device = None
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
        X, xp, block = self._check_X(X)
        out = np.empty((len(X), len(self.classes_)), dtype=np.float32)
        if len(self.classes_) < 2:
            out[:] = 1
            return out
        for start in range(0, len(X), block):
            p = self._proba(X[start : start + block], xp)
            out[start : start + block] = xp.get(p)
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
        X, xp, block = self._check_X(X)
        if len(self.classes_) < 2:
            return np.full(len(X), self.classes_[0])
        out = np.empty(len(X), dtype=self.classes_.dtype)
        for start in range(0, len(X), block):
            p = self._proba(X[start : start + block], xp)
            out[start : start + block] = self.classes_[xp.get(p.argmax(axis=1))]
        return out

    def __getstate__(self):
        """Pickle without the arrays moved to a device; they are rebuilt on demand."""
        return {k: v for k, v in self.__dict__.items() if k != "_on_device"}

    def _check_params(self):
        if not 0.0 <= self.knn_weight <= 1.0:
            raise ValueError("knn_weight must be between 0 and 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be at least 0 and below 1")
        for name in ("hidden", "steps", "batch_size", "n_neighbors", "max_bank_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")

    def _check_X(self, X):
        """Validated X, the backend, and the rows to predict at once."""
        check_is_fitted(self, "classes_")
        self._check_params()
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected X with shape (n, {self.n_features_in_}), got {X.shape}"
            )
        if not np.isfinite(X).all():
            raise ValueError("Features contain NaN or infinite values")
        if self.knn_weight < 1 and self.mlp_ is None and len(self.classes_) > 1:
            raise ValueError("knn_weight was 1 at fit time; call fit again")
        if self.knn_weight > 0 and self.bank_ is None and len(self.classes_) > 1:
            raise ValueError("knn_weight was 0 at fit time; call fit again")
        xp = _backend(self.device)
        # Bound the (block, bank) similarity matrix of the kNN vote.
        block = _BLOCK_MAX[xp.name]
        if self.bank_ is not None and self.knn_weight > 0:
            block = min(block, max(256, _BLOCK_CELLS[xp.name] // len(self.bank_[1])))
        return X, xp, block

    def _fitted(self, xp):
        """The MLP parameters and the bank as arrays of ``xp``, cached."""
        key = (xp.name, getattr(xp, "device", None))
        cache = getattr(self, "_on_device", None)
        if cache is None or cache[0] != key:
            mlp = (
                None
                if self.mlp_ is None
                else {k: xp.put(v) for k, v in self.mlp_.items()}
            )
            bank = None if self.bank_ is None else tuple(map(xp.put, self.bank_))
            self._on_device = cache = (key, mlp, bank)
        return cache[1:]

    def _proba(self, X, xp):
        """Probabilities for one block of rows, as an array of ``xp``."""
        X = xp.put(np.asarray(X, dtype=np.float32))
        mlp, bank = self._fitted(xp)
        if self.knn_weight == 1:
            return self._knn_proba(X, bank, xp)
        p = self._mlp_proba(X, mlp, xp)
        if self.knn_weight == 0:
            return p
        w = self.knn_weight
        return (1 - w) * p + w * self._knn_proba(X, bank, xp)

    def _fit_mlp(self, X, y, n_classes, rng, xp):
        """Train the MLP with class-weighted cross-entropy and AdamW."""
        n, d = X.shape
        # PyTorch's nn.Linear initialization: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        params = {}
        for name, fan_in, shape in (
            ("w1", d, (d, self.hidden)),
            ("b1", d, (self.hidden,)),
            ("w2", self.hidden, (self.hidden, n_classes)),
            ("b2", self.hidden, (n_classes,)),
        ):
            bound = 1.0 / math.sqrt(fan_in)
            params[name] = xp.put(rng.uniform(-bound, bound, shape).astype(np.float32))
        class_weight = xp.put(
            compute_class_weight("balanced", classes=np.arange(n_classes), y=y).astype(
                np.float32
            )
        )

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = {k: p * 0 for k, p in params.items()}
        v = {k: p * 0 for k, p in params.items()}
        batch = min(n, self.batch_size)
        rows = xp.arange(batch)
        keep = np.float32(1.0 - self.dropout)
        lr = self.learning_rate
        X, y = xp.put(X), xp.put(y)
        for step in range(1, self.steps + 1):
            idx = xp.put(rng.integers(0, n, batch))
            xb, yb = X[idx], y[idx]
            mask = xp.put(
                (rng.random((batch, self.hidden), dtype=np.float32) < keep) / keep
            )
            pre = xb @ params["w1"] + params["b1"]
            cdf = 0.5 * (1.0 + xp.erf(pre / math.sqrt(2.0)))
            hid = pre * cdf * mask
            prob = xp.softmax(hid @ params["w2"] + params["b2"])

            # Gradient of the weighted mean cross-entropy w.r.t. the logits.
            row_weight = class_weight[yb]
            grad = prob
            grad[rows, yb] -= 1.0
            grad *= (row_weight / row_weight.sum())[:, None]
            grads = {"w2": hid.T @ grad, "b2": grad.sum(axis=0)}
            pdf = xp.exp(-0.5 * pre * pre) / math.sqrt(2.0 * math.pi)
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
                p -= lr * m_hat / (xp.sqrt(v_hat) + eps)
        return {k: xp.get(p) for k, p in params.items()}

    def _mlp_proba(self, X, mlp, xp):
        pre = X @ mlp["w1"] + mlp["b1"]
        hid = pre * 0.5 * (1.0 + xp.erf(pre / math.sqrt(2.0)))
        return xp.softmax(hid @ mlp["w2"] + mlp["b2"])

    def _build_bank(self, X, y, n_classes, rng):
        """Bank rows (L2-normalized), their class indices and vote weights."""
        counts = np.bincount(y, minlength=n_classes)
        if counts.sum() > self.max_bank_size:
            # A uniform subsample, except that every class keeps at least
            # n_neighbors rows (taken from the largest classes) so that a
            # sparsely painted class can still win a vote.
            quota = rng.multivariate_hypergeometric(counts, self.max_bank_size)
            floor = np.minimum(
                counts, min(self.n_neighbors, self.max_bank_size // n_classes)
            )
            short = np.maximum(floor - quota, 0)
            quota, excess = quota + short, short.sum()
            for c in np.argsort(-quota, kind="stable"):
                take = min(excess, quota[c] - floor[c])
                quota[c], excess = quota[c] - take, excess - take
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

    def _knn_proba(self, X, bank, xp):
        """Class-balanced vote of the most similar bank rows (cosine)."""
        rows, bank_y, vote_weight = bank
        k = min(self.n_neighbors, len(rows))
        top = xp.top_k(xp.normalize(X) @ rows.T, k)
        votes = bank_y[top][:, :, None] == xp.arange(len(vote_weight))
        votes = votes.sum(axis=1, dtype=xp.float32) * vote_weight
        return votes / votes.sum(axis=1, keepdims=True).clip(1e-9)
