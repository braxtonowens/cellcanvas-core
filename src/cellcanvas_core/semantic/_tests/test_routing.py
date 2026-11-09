import pickle

import numpy as np
import pytest
import zarr
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.fewshot import FewShotClassifier
from cellcanvas_core.semantic.picking import pick
from cellcanvas_core.semantic.routing import Route, RoutedClassifier, feature_columns
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager

FAST = FewShotClassifier(steps=60, batch_size=256)
# Painted values: 1 background, 2 a region class, 3 a particle class.
BG, REGION, PARTICLE = 1, 2, 3


class Fixed(ClassifierMixin, BaseEstimator):
    """A head that records its training data and returns fixed probabilities.

    ``table`` maps a label set (sorted tuple of the labels seen in fit) to the
    probability row returned for every input.
    """

    def __init__(self, table=None):
        self.table = table

    def fit(self, X, y):
        self.X_, self.y_ = X.copy(), y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        row = self.table[tuple(int(c) for c in self.classes_)]
        return np.tile(np.asarray(row, np.float32), (len(X), 1))


def routes(region_columns=((0, 4),), particle_columns=((4, 6),)):
    return {
        "region": Route((REGION,), region_columns),
        "particle": Route((PARTICLE,), particle_columns),
    }


def labels(*counts):
    """Zero-based labels: counts[i] rows of painted value i + 1."""
    return np.repeat(np.arange(len(counts)), counts)


def test_feature_columns_follow_the_dictionary_order():
    shape = (2, 3, 3)
    features = {
        "a": np.zeros((4, *shape)),
        "b": np.zeros((2, *shape)),
        "c": np.zeros((3, *shape)),
    }
    ds = DataSet(
        np.zeros((1, *shape)), features, np.zeros(shape, "i4"), np.zeros(shape, "i4")
    )
    assert feature_columns(ds, ["c", "a"]) == ((6, 9), (0, 4))
    with pytest.raises(KeyError, match="'d'"):
        feature_columns(ds, ["a", "d"])
    with pytest.raises(ValueError):
        feature_columns(ds, ["a", "a"])
    with pytest.raises(TypeError):  # one name, not its letters
        feature_columns(ds, "a")
    last = {name: np.moveaxis(v, 0, -1) for name, v in features.items()}
    ds_last = DataSet(
        np.zeros((1, *shape)),
        last,
        np.zeros(shape, "i4"),
        np.zeros(shape, "i4"),
        feature_axis=-1,
    )
    assert feature_columns(ds_last, ["c", "a"]) == ((6, 9), (0, 4))
    model = RoutedClassifier.from_dataset(
        ds,
        {"region": ([2], ["a", "b"]), "particle": ([3], ["a", "c"])},
        background_label=1,
    )
    assert model.routes["particle"] == Route((3,), ((0, 4), (6, 9)))


def test_heads_see_only_their_columns_and_other_routes_as_background():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(9, 6)).astype(np.float32)
    y = labels(3, 3, 3)
    table = {(0, 1): [0.5, 0.5], (0, 2): [0.5, 0.5]}
    model = RoutedClassifier(routes(), BG, Fixed(table), balance=False).fit(X, y)
    region, particle = model.heads_["region"], model.heads_["particle"]
    np.testing.assert_array_equal(region.X_, X[:, :4])
    np.testing.assert_array_equal(particle.X_, X[:, 4:6])
    np.testing.assert_array_equal(region.y_, [0, 0, 0, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(particle.y_, [0, 0, 0, 0, 0, 0, 2, 2, 2])
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])


def test_product_of_experts():
    """q_c = P_g(c) * prod of the other heads' background; background = prod of all."""
    X, y = np.zeros((9, 6), np.float32), labels(3, 3, 3)
    table = {(0, 1): [0.2, 0.8], (0, 2): [0.6, 0.4]}
    model = RoutedClassifier(routes(), BG, Fixed(table)).fit(X, y)
    q = np.array([0.2 * 0.6, 0.8 * 0.6, 0.4 * 0.2])
    np.testing.assert_allclose(model.predict_proba(X[:2]), [q / q.sum()] * 2, rtol=1e-6)
    np.testing.assert_array_equal(model.predict(X[:2]), [1, 1])


def test_one_route_is_the_head_alone():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    y = labels(20, 20)
    X[20:, :4] += 3
    single = {"region": Route((REGION,), ((0, 4),))}
    model = RoutedClassifier(single, BG, FAST).fit(X, y)
    head = clone(FAST).fit(X[:, :4], y)
    np.testing.assert_allclose(
        model.predict_proba(X), head.predict_proba(X[:, :4]), atol=1e-6
    )


def test_region_paint_suppresses_particle_false_positives():
    """Region paint removes particle false positives inside the region.

    The particle features alone cannot tell the region's interior from a
    particle; the region head can, and its vote removes the particle there.
    """
    rng = np.random.default_rng(3)
    n = 60
    centres = {"bg": 0.0, "region": 4.0, "particle": 4.0}  # particle features
    region_feature = {"bg": 0.0, "region": 4.0, "particle": 0.0}
    parts = []
    for name in ("bg", "region", "particle"):
        r = rng.normal(size=(n, 4)) + region_feature[name]
        p = rng.normal(size=(n, 2)) + centres[name]
        parts.append(np.hstack([r, p]))
    X = np.vstack(parts).astype(np.float32)
    y = labels(n, n, n)
    # paint only part of the region: the rest is unseen, like most of a tomogram
    painted = np.r_[0:n, n : n + n // 4, 2 * n : 3 * n]
    model = RoutedClassifier(routes(), BG, FAST).fit(X[painted], y[painted])
    unseen_region = X[n + n // 4 : 2 * n]
    particle = list(model.classes_).index(PARTICLE - 1)
    region = list(model.classes_).index(REGION - 1)
    alone = model.heads_["particle"].predict_proba(unseen_region[:, 4:])[:, -1]
    combined = model.predict_proba(unseen_region)
    assert alone.mean() > 0.6  # on its own the particle head fires in the region
    assert combined[:, particle].mean() < alone.mean() / 2
    assert (combined.argmax(1) == region).mean() > 0.85
    on_particles = model.predict_proba(X[2 * n :])
    assert (on_particles.argmax(1) == particle).mean() > 0.9


@pytest.mark.parametrize("only", ["region", "particle"])
def test_one_kind_of_class(only):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    X[20:] += 5
    painted = REGION if only == "region" else PARTICLE
    y = np.r_[np.zeros(20, int), np.full(20, painted - 1)]
    model = RoutedClassifier({only: routes()[only]}, BG, FAST).fit(X, y)
    np.testing.assert_array_equal(model.classes_, [0, painted - 1])
    assert (model.predict(X) == y).mean() > 0.9


def test_a_route_without_painted_classes_predicts_nothing_of_its_own():
    """Only regions painted, but a particle route exists.

    There is no particle column, and the result is the region head's.
    """
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    X[20:, :4] += 3
    y = labels(20, 20)
    model = RoutedClassifier(routes(), BG, FAST).fit(X, y)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_allclose(
        model.predict_proba(X),
        model.heads_["region"].predict_proba(X[:, :4]),
        atol=1e-6,
    )


def test_background_never_painted():
    """Without background paint each head is trained on its rivals' rows."""
    X, y = np.zeros((6, 6), np.float32), np.repeat([1, 2], 3)
    table = {(0, 1): [0.3, 0.7], (0, 2): [0.9, 0.1]}
    model = RoutedClassifier(routes(), BG, Fixed(table)).fit(X, y)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])  # shared background
    q = np.array([0.3 * 0.9, 0.7 * 0.9, 0.1 * 0.3])
    np.testing.assert_allclose(model.predict_proba(X[:1])[0], q / q.sum(), rtol=1e-6)


def test_rows_where_every_head_is_certain_use_the_heads_own_classes():
    """Every product is zero there; the row must not fall to the background."""
    X, y = np.zeros((9, 6), np.float32), labels(3, 3, 3)
    table = {(0, 1): [0.0, 1.0], (0, 2): [0.0, 1.0]}
    model = RoutedClassifier(routes(), BG, Fixed(table)).fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X[:1]), [[0.0, 0.5, 0.5]])
    assert model.predict(X[:1])[0] != BG - 1


@pytest.mark.parametrize("balance", [False, True])
def test_float16_rows(balance):
    """Float16 tiles predict as float32 ones; unscaled columns keep float16."""
    rng = np.random.default_rng(5)
    X = (rng.normal(size=(90, 6)) + np.repeat(np.eye(3, 6) * 3, 30, 0)).astype(
        np.float16
    )
    y = labels(30, 30, 30)
    seen = []

    class Recorder(Fixed):
        def predict_proba(self, X):
            seen.append(X.dtype)
            return super().predict_proba(X)

    table = {(0, 1): [0.5, 0.5], (0, 2): [0.5, 0.5]}
    model = RoutedClassifier(routes(), BG, Recorder(table), balance=balance)
    model.fit(X.astype(np.float32), y)
    np.testing.assert_allclose(
        model.predict_proba(X), model.predict_proba(X.astype(np.float32))
    )
    assert np.float16 in seen
    fewshot = RoutedClassifier(routes(), BG, FAST, balance=balance).fit(X, y)
    np.testing.assert_allclose(
        fewshot.predict_proba(X),
        fewshot.predict_proba(X.astype(np.float32)),
        atol=1e-3,
    )


def test_invalid_routes_raise():
    X, y = np.zeros((9, 6), np.float32), labels(3, 3, 3)
    fixed = Fixed({(0, 1): [0.5, 0.5], (0, 2): [0.5, 0.5]})
    with pytest.raises(ValueError, match=r"\[4\] are in no route"):
        RoutedClassifier(routes(), BG, fixed).fit(X, labels(3, 3, 0, 3))
    with pytest.raises(ValueError, match="one route only"):
        bad = {"a": Route((2,), ((0, 4),)), "b": Route((2, 3), ((4, 6),))}
        RoutedClassifier(bad, BG, fixed).fit(X, y)
    with pytest.raises(ValueError, match="background is shared"):
        RoutedClassifier({"a": Route((1, 2), ((0, 4),))}, BG, fixed).fit(X, y)
    with pytest.raises(ValueError, match="outside"):
        RoutedClassifier(routes(particle_columns=((4, 7),)), BG, fixed).fit(X, y)
    with pytest.raises(ValueError, match="at least 1"):
        RoutedClassifier(routes(), 0, fixed).fit(X, y)
    with pytest.raises(ValueError, match="no classes"):
        RoutedClassifier({"a": Route((), ((0, 4),))}, BG, fixed).fit(X, y)
    with pytest.raises(ValueError, match="At least one route"):
        RoutedClassifier({}, BG, fixed).fit(X, y)
    model = RoutedClassifier(routes(), BG, fixed).fit(X, y)
    with pytest.raises(ValueError, match="shape"):
        model.predict_proba(np.zeros((2, 5), np.float32))


def test_balance_scales_each_array():
    X = np.ones((6, 6), np.float32)
    y = labels(3, 3)
    fixed = Fixed({(0, 1): [0.5, 0.5]})
    model = RoutedClassifier({"r": Route((2,), ((0, 4), (4, 6)))}, BG, fixed).fit(X, y)
    x = model.heads_["r"].X_
    np.testing.assert_allclose(x[0, :4], np.sqrt(6 / (2 * 4)))
    np.testing.assert_allclose(x[0, 4:], np.sqrt(6 / (2 * 2)))
    sq = [(x[0, :4] ** 2).sum(), (x[0, 4:] ** 2).sum()]
    np.testing.assert_allclose(sq[0], sq[1])  # equal shares of the squared norm


def test_clone_and_pickle():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 6)).astype(np.float32)
    y = labels(10, 10, 10)
    X[10:20, :4] += 3
    X[20:, 4:] += 3
    model = RoutedClassifier(routes(), BG, FAST)
    copy = clone(model)
    assert copy.routes == model.routes and copy.estimator is not FAST
    copy.set_params(estimator__steps=7)
    assert copy.estimator.steps == 7 and FAST.steps == 60
    model.fit(X, y)
    assert model.heads_["region"] is not model.heads_["particle"]
    again = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(again.predict_proba(X), model.predict_proba(X))


def volume(seed=0):
    """Three feature arrays over a (6, 24, 24) volume.

    "shared" sees everything weakly; "regional" marks the slab (class 2);
    "particle" marks small spheres (class 3) but ALSO lights up inside the
    slab, as a region-blind particle feature would.
    """
    rng = np.random.default_rng(seed)
    shape = (6, 24, 24)
    z, y, x = np.indices(shape)
    slab = (y >= 14) & (y < 22)
    centres = [(2, 4, 4), (3, 4, 16), (2, 9, 10), (4, 6, 20)]
    sphere = np.zeros(shape, bool)
    for c in centres:
        sphere |= (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= 4
    shared = rng.normal(size=(2, *shape)) + 1.5 * (slab | sphere)
    regional = rng.normal(size=(3, *shape)) + 4 * slab
    particle = rng.normal(size=(2, *shape)) + 4 * (sphere | slab)
    features = {
        "shared": shared.astype("f4"),
        "regional": regional.astype("f4"),
        "particle": particle.astype("f4"),
    }
    painted = np.zeros(shape, "i4")
    painted[3, :12, :][~sphere[3, :12, :]] = BG
    painted[3, 14:17, :] = REGION  # a few rows of the slab only
    painted[sphere & (z == 3)] = PARTICLE
    return features, painted, slab, sphere, centres


def test_end_to_end_with_chunked_zarr_features(tmp_path):
    """Through DataManager, SemanticSegmentationManager and picking.

    Features are chunked Zarr arrays, and a second tomogram is unpainted.
    """
    features, painted, slab, sphere, centres = volume()
    stored = {}
    for name, values in features.items():
        array = zarr.create_array(
            str(tmp_path / f"{name}.zarr"),
            shape=values.shape,
            chunks=(values.shape[0], 2, 12, 12),
            dtype="f4",
        )
        array[:] = values
        stored[name] = array
    shape = painted.shape
    ds = DataSet(np.zeros((1, *shape)), stored, painted, np.zeros(shape, "i4"))
    other = DataSet(
        np.zeros((1, *shape)), stored, np.zeros(shape, "i4"), np.zeros(shape, "i4")
    )
    model = RoutedClassifier.from_dataset(
        ds,
        {
            "region": ([REGION], ["shared", "regional"]),
            "particle": ([PARTICLE], ["shared", "particle"]),
        },
        background_label=BG,
        estimator=FAST,
    )
    manager = SemanticSegmentationManager(DataManager([ds, other]), model)
    manager.fit()
    proba = manager.predict_proba(ds.concatenated_features)
    assert proba.shape == (3, *shape)
    np.testing.assert_allclose(proba.sum(0), 1, rtol=1e-5)
    prediction = manager.predict(ds.concatenated_features)
    assert (prediction[slab] == REGION - 1).mean() > 0.9
    # picks of the particle class: the spheres, nothing in the slab
    points, _ = pick(proba[PARTICLE - 1], radius=2, threshold=0.3, edge=0)
    assert len(points) >= len(centres)
    inside = slab[tuple(np.round(points).astype(int).T)]
    assert not inside.any()
    found = [np.min(np.linalg.norm(points - c, axis=1)) <= 2 for c in centres]
    assert all(found)
    # a single head on the particle-route features fires in the slab
    single = RoutedClassifier.from_dataset(
        ds,
        {"all": ([REGION, PARTICLE], ["shared", "particle"])},
        background_label=BG,
        estimator=FAST,
    )
    single_manager = SemanticSegmentationManager(DataManager([ds]), single)
    single_manager.fit()
    flooded = single_manager.predict_proba(ds.concatenated_features)[PARTICLE - 1]
    assert flooded[slab].mean() > proba[PARTICLE - 1][slab].mean() + 0.2
    manager.predict_to_segmentation(ds, batch_size=500)
    np.testing.assert_array_equal(ds.segmentation, prediction + 1)


def test_heads_need_not_handle_a_single_class():
    """A route with nothing painted yet works with any sklearn classifier."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    X[20:, :4] += 3
    y = labels(20, 20)  # background and region only, no particle yet
    model = RoutedClassifier(routes(), BG, LogisticRegression()).fit(X, y)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    assert (model.predict(X) == y).mean() > 0.9


def test_fitted_state_does_not_follow_later_parameter_changes():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(30, 6)).astype(np.float32)
    X[10:20, :4] += 3
    X[20:, 4:] += 3
    y = labels(10, 10, 10)
    model = RoutedClassifier(routes(), BG, FAST).fit(X, y)
    before = model.predict_proba(X)
    model.set_params(background_label=4, routes={"r": Route((2,), ((0, 6),))})
    model.set_params(balance=False)
    np.testing.assert_array_equal(model.predict_proba(X), before)


def test_unlabelled_rows_are_ignored():
    X, y = np.zeros((9, 6), np.float32), labels(3, 3, 3)
    fixed = Fixed({(0, 1): [0.5, 0.5], (0, 2): [0.5, 0.5]})
    model = RoutedClassifier(routes(), BG, fixed)
    model.fit(np.vstack([X, np.ones((2, 6), np.float32)]), np.r_[y, -1, -1])
    np.testing.assert_array_equal(model.heads_["region"].X_, X[:, :4])
    with pytest.raises(ValueError, match="No labelled rows"):
        model.fit(X[:2], np.array([-1, -1]))


def test_route_arguments_are_checked():
    with pytest.raises(TypeError, match="sequence of painted"):
        Route(2, ((0, 4),))
    with pytest.raises(TypeError, match="start, stop"):
        Route((2,), (0, 4))
    assert Route([2, 3], [[0, 4]]) == Route((2, 3), ((0, 4),))


def test_classes_of_a_route_in_any_order():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 6)).astype(np.float32)
    X[20:40, :4] += 4
    X[40:, :4] -= 4
    y = labels(20, 20, 20)
    one = RoutedClassifier({"r": Route((2, 3), ((0, 4),))}, BG, FAST).fit(X, y)
    other = RoutedClassifier({"r": Route((3, 2), ((0, 4),))}, BG, FAST).fit(X, y)
    np.testing.assert_array_equal(one.classes_, [0, 1, 2])
    np.testing.assert_array_equal(one.predict_proba(X), other.predict_proba(X))


def test_with_seeded_background():
    """Only the foreground is painted; SeededDataManager supplies the background."""
    from cellcanvas_core.data.seeding import SeededDataManager

    features, painted, slab, sphere, _ = volume()
    painted[painted == BG] = 0
    shape = painted.shape
    ds = DataSet(np.zeros((1, *shape)), features, painted, np.zeros(shape, "i4"))
    model = RoutedClassifier.from_dataset(
        ds,
        {
            "region": ([REGION], ["shared", "regional"]),
            "particle": ([PARTICLE], ["shared", "particle"]),
        },
        background_label=BG,
        estimator=FAST,
    )
    manager = SemanticSegmentationManager(
        SeededDataManager([ds], background_label=BG), model
    )
    manager.fit()
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    prediction = manager.predict(ds.concatenated_features)
    background = ~(slab | sphere)
    assert (prediction[background] == BG - 1).mean() > 0.8
    assert (prediction[slab] == REGION - 1).mean() > 0.9
