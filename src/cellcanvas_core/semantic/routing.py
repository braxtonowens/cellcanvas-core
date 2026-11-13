"""Separate heads for groups of classes, each on its own features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from cellcanvas_core.semantic.fewshot import FewShotClassifier

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellcanvas_core.data.data_set import DataSet


@dataclass(frozen=True)
class Route:
    """The classes one head predicts and the feature columns it sees.

    Parameters
    ----------
    classes : tuple of int
        Painted label values (``>= 1``) of the classes of this head.
    columns : tuple of (int, int)
        ``(start, stop)`` column ranges of the feature rows, one per feature
        array, as returned by `feature_columns`.
    """

    classes: tuple[int, ...]
    columns: tuple[tuple[int, int], ...]

    def __post_init__(self):
        """Check the argument types and store them as tuples of int."""
        if isinstance(self.classes, int | np.integer | str):
            raise TypeError("Route classes must be a sequence of painted label values")
        if isinstance(self.columns, str) or any(
            np.ndim(c) != 1 or len(c) != 2 for c in self.columns
        ):
            raise TypeError("Route columns must be a sequence of (start, stop) pairs")
        object.__setattr__(
            self, "classes", tuple(_whole(c, "classes") for c in self.classes)
        )
        object.__setattr__(
            self,
            "columns",
            tuple(
                (_whole(a, "columns"), _whole(b, "columns")) for a, b in self.columns
            ),
        )


def _whole(value, what):
    """``value`` as an int; reject fractions and non-numbers instead of truncating."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise TypeError(f"Route {what} must be integers, got {value!r}")
    if value != int(value):
        raise ValueError(f"Route {what} must be whole numbers, got {value!r}")
    return int(value)


class _OneClass:
    """A head that saw one class: probability one for it."""

    def __init__(self, label):
        self.classes_ = np.array([label])

    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=np.float32)


def feature_columns(dataset: DataSet, names: Sequence[str]):
    """Column ranges of the named feature arrays in the dataset's feature rows.

    Rows from `DataManager.get_training_data` and
    `SemanticSegmentationManager.predict` hold the feature arrays of
    ``dataset.features`` side by side, in dictionary order. The ranges are
    only valid for datasets whose ``features`` have the same arrays, widths
    and order: rows are matched by position, not by name.

    Parameters
    ----------
    dataset : DataSet
        A dataset with the feature arrays every dataset of the model uses, in
        the same order.
    names : sequence of str
        Keys of ``dataset.features``.

    Returns
    -------
    columns : tuple of (int, int)
        ``(start, stop)`` of each named array, in the order of ``names``.
    """
    if isinstance(names, str):
        raise TypeError("names must be a sequence of feature array names")
    ranges, start = {}, 0
    for name, values in dataset.features.items():
        width = values.shape[dataset.feature_axis]
        ranges[name] = (start, start + width)
        start += width
    missing = [n for n in names if n not in ranges]
    if missing:
        raise KeyError(f"No feature arrays {missing}; the dataset has {list(ranges)}")
    if len(set(names)) != len(names):
        raise ValueError("Each feature array can be named once per route")
    return tuple(ranges[n] for n in names)


class RoutedClassifier(ClassifierMixin, BaseEstimator):
    """One head per group of classes, each on its own feature arrays.

    Classes of different kinds can need different features: a feature that
    helps large regions can hurt small particles. Each route names the classes
    of one head and the feature columns it sees. Every head is trained on all
    painted rows, one-vs-rest: the classes of the other routes count as
    background for it. The heads are combined as a product of experts with a
    shared background: for class c of head g,
    ``q_c = P_g(c) * prod(P_h(background) for every other head h)``, and
    ``q_background = prod(P_h(background) for every head h)``, normalized
    over the classes. A voxel is background only if every head says so, and
    a class of one head is suppressed where another head finds its own class,
    so painted regions also remove particle false positives inside them.
    With one route of one feature array this is the head alone. Each head
    predicts every row, so prediction takes about as long as the heads
    together.

    The routes, ``background_label`` and ``balance`` in effect at `fit` are
    kept with the fitted heads (``routes_``, ``background_label_``,
    ``balance_``); changing the parameters
    afterwards needs a new `fit`. Every dataset must have the same feature
    arrays in the same order (see `feature_columns`).

    Labels follow `DataManager.get_training_data` (painted value minus one);
    `predict` and the columns of `predict_proba` use the same zero-based
    labels, so the "add one before saving" rule of
    `SemanticSegmentationManager` is unchanged.

    Parameters
    ----------
    routes : mapping of str to Route
        One head per entry. Classes must not repeat across routes.
    background_label : int
        Painted label value of the background class (``>= 1``), not in any
        route; the same value as a `SeededDataManager`'s. Rows of classes
        that are in no route raise an error in `fit`.
    estimator : estimator or None
        Cloned for each head; must have ``fit``, ``predict_proba`` and
        ``classes_``. None uses `FewShotClassifier` with its defaults. A head
        that sees only one class (its route has nothing painted yet) is not
        fitted and gives that class probability one.
    balance : bool
        Scale each feature array of a head by ``sqrt(D / (n * D_b))``, where
        ``D_b`` is its width, ``D`` the head's total width and ``n`` the number
        of arrays, so that each array carries about the same share of the
        squared norm. This assumes standardized features (per channel); a
        single array is unchanged.

    Attributes
    ----------
    classes_ : np.ndarray
        Zero-based labels seen in `fit`, plus the shared background (every
        head is trained against it), the columns of `predict_proba`.
    heads_ : dict of str to estimator
        The fitted head of each route.
    routes_ : dict of str to Route
        The routes the heads were fitted with.
    background_label_ : int
        The background label value the heads were fitted with.
    balance_ : bool
        Whether the heads were fitted on balanced feature arrays.
    n_features_in_ : int
        Number of features seen in `fit`.
    """

    def __init__(
        self,
        routes: Mapping[str, Route],
        background_label: int,
        estimator=None,
        balance: bool = True,
    ):
        self.routes = routes
        self.background_label = background_label
        self.estimator = estimator
        self.balance = balance

    @classmethod
    def from_dataset(
        cls,
        dataset: DataSet,
        routes: Mapping[str, tuple[Sequence[int], Sequence[str]]],
        background_label: int,
        **kwargs,
    ):
        """Routes given as painted label values and feature array names.

        Parameters
        ----------
        dataset : DataSet
            Supplies the feature columns, as `feature_columns`.
        routes : mapping of str to (class labels, feature names)
            In the order of `Route`, for example
            ``{"region": ([2, 3], ["cryosiam", "sam"]),
            "particle": ([4], ["cryosiam", "tomotwin"])}``.
        background_label : int
            As for the class.
        **kwargs
            ``estimator`` and ``balance``.
        """
        built = {
            name: Route(classes, feature_columns(dataset, names))
            for name, (classes, names) in routes.items()
        }
        return cls(built, background_label=background_label, **kwargs)

    def fit(self, X, y):
        """Fit one head per route on all rows, one-vs-rest.

        Parameters
        ----------
        X : array-like
            (n, d) feature rows.
        y : array-like
            (n,) zero-based labels (painted value minus one); rows labelled
            -1 (unlabelled, as for `FewShotClassifier`) are ignored.

        Returns
        -------
        self : RoutedClassifier
            The fitted classifier.
        """
        X, y = np.asarray(X), np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError("Expected X of shape (n, d) and y of shape (n,)")
        if y.dtype.kind not in "iu":
            raise ValueError("Labels must be integers")
        background, members = self._check_routes(X.shape[1])
        X, y = X[y != -1], y[y != -1]
        if len(y) == 0:
            raise ValueError("No labelled rows to fit; paint at least one class")
        if y.min() < 0:
            raise ValueError("Labels are painted values minus one, at least -1")
        seen = np.unique(y)
        unrouted = sorted(
            int(c) + 1 for c in seen if c != background and c not in members
        )
        if unrouted:
            raise ValueError(
                f"Painted labels {unrouted} are in no route and are not the background"
            )
        estimator = FewShotClassifier() if self.estimator is None else self.estimator
        routes = dict(self.routes)
        heads = {}
        for name, route in routes.items():
            own = np.isin(y, [c - 1 for c in route.classes])
            target = np.where(own, y, background)
            if len(np.unique(target)) == 1:
                heads[name] = _OneClass(target[0])
            else:
                heads[name] = clone(estimator).fit(
                    _columns(X, route, self.balance), target
                )
        self.heads_, self.routes_ = heads, routes
        self.background_label_, self.balance_ = self.background_label, self.balance
        self.classes_ = np.unique(np.concatenate([h.classes_ for h in heads.values()]))
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        """Combined class probabilities, columns ordered as ``classes_``.

        Parameters
        ----------
        X : array-like
            (n, d) feature rows.

        Returns
        -------
        probabilities : np.ndarray
            (n, n_classes) float32 array; each row sums to one. Where every
            head gives the background probability zero, so the product is zero
            for every class, the row is the heads' own-class probabilities,
            normalized.
        """
        check_is_fitted(self, "heads_")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected X with shape (n, {self.n_features_in_}), got {X.shape}"
            )
        background = self.background_label_ - 1
        probs, not_mine = {}, {}
        for name, route in self.routes_.items():
            head = self.heads_[name]
            rows = _columns(X, route, self.balance_)
            p = np.asarray(head.predict_proba(rows), np.float64)
            probs[name] = p
            hit = np.flatnonzero(head.classes_ == background)
            not_mine[name] = p[:, hit[0]] if len(hit) else np.zeros(len(X))
        index = {int(c): i for i, c in enumerate(self.classes_)}
        q = np.zeros((len(X), len(self.classes_)))
        own = np.zeros_like(q)
        if background in index:
            q[:, index[background]] = np.prod(list(not_mine.values()), axis=0)
        for name, p in probs.items():
            others = [v for k, v in not_mine.items() if k != name]
            rest = np.prod(others, axis=0) if others else 1.0
            for j, c in enumerate(self.heads_[name].classes_):
                if c != background:
                    q[:, index[int(c)]] = p[:, j] * rest
                    own[:, index[int(c)]] = p[:, j]
        total = q.sum(axis=1, keepdims=True)
        empty = total[:, 0] == 0
        q[empty], total[empty] = own[empty], own[empty].sum(axis=1, keepdims=True)
        q = np.divide(q, total, out=np.full_like(q, 1 / q.shape[1]), where=total > 0)
        return q.astype(np.float32)

    def predict(self, X):
        """Most probable class per row, from `predict_proba`.

        Parameters
        ----------
        X : array-like
            (n, d) feature rows.

        Returns
        -------
        labels : np.ndarray
            (n,) labels taken from ``classes_``.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _check_routes(self, n_features):
        """The zero-based background label and the zero-based routed labels."""
        if not self.routes:
            raise ValueError("At least one route is required")
        if self.background_label < 1:
            raise ValueError("background_label must be at least 1 (0 is unlabelled)")
        members = []
        for name, route in self.routes.items():
            if not route.classes:
                raise ValueError(f"Route {name} has no classes")
            if not route.columns:
                raise ValueError(f"Route {name} has no feature columns")
            for start, stop in route.columns:
                if not 0 <= start < stop <= n_features:
                    raise ValueError(
                        f"Route {name} columns ({start}, {stop}) are outside "
                        f"the {n_features} features"
                    )
            members += list(route.classes)
        if min(members) < 1:
            raise ValueError("Route classes are painted label values, at least 1")
        if len(set(members)) != len(members):
            raise ValueError("A class can be in one route only")
        if self.background_label in members:
            raise ValueError("The background is shared; it cannot be in a route")
        return self.background_label - 1, {c - 1 for c in members}


def _columns(X, route, balance):
    """The route's feature columns of X, each array scaled if balance.

    Scaled columns are float32; unscaled ones keep the input dtype, so a
    float16 tile reaches the head at half size.
    """
    widths = [stop - start for start, stop in route.columns]
    if not balance or len(widths) == 1:
        if len(widths) == 1:
            start, stop = route.columns[0]
            return X[:, start:stop]
        return np.concatenate([X[:, a:b] for a, b in route.columns], axis=1)
    total = sum(widths)
    out = np.empty((len(X), total), np.float32)
    offset = 0
    for (start, stop), width in zip(route.columns, widths, strict=True):
        scale = np.sqrt(total / (len(widths) * width))
        np.multiply(
            X[:, start:stop], np.float32(scale), out=out[:, offset : offset + width]
        )
        offset += width
    return out
