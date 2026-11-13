import pickle
import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.fewshot import FewShotClassifier
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager

# Small settings keep the tests fast; the algorithm is the same.
FAST = {"steps": 60, "batch_size": 256}


def blobs(n_per_class, n_classes=3, dim=16, spread=0.5, seed=0, noise_seed=1):
    """Gaussian clusters around well separated centres (fixed by seed)."""
    centres = np.random.default_rng(seed).normal(size=(n_classes, dim)) * 3
    y = np.repeat(np.arange(n_classes), n_per_class)
    noise = np.random.default_rng(noise_seed).normal(size=(len(y), dim))
    X = centres[y] + spread * noise
    return X.astype(np.float32), y


def test_fit_predict_shapes():
    X, y = blobs(20)
    model = FewShotClassifier(**FAST).fit(X, y)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    assert model.n_features_in_ == 16
    proba = model.predict_proba(X[:7])
    assert proba.shape == (7, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-5)
    assert model.predict(X[:7]).shape == (7,)
    np.testing.assert_array_equal(model.predict(X[:7].tolist()), model.predict(X[:7]))


def test_column_labels_are_flattened():
    X, y = blobs(10)
    model = FewShotClassifier(**FAST).fit(X, y[:, None])
    assert model.predict(X).shape == (len(X),)


def test_solves_separable_problem_from_few_examples():
    X_train, y_train = blobs(5)
    X_test, y_test = blobs(200, noise_seed=2)
    model = FewShotClassifier(**FAST).fit(X_train, y_train)
    assert np.mean(model.predict(X_test) == y_test) > 0.95


def test_rare_class_is_not_swamped():
    X, y = blobs(500, n_classes=2, seed=1)
    keep = np.concatenate([np.arange(500), np.arange(500, 505)])  # 500 vs 5 rows
    model = FewShotClassifier(**FAST).fit(X[keep], y[keep])
    rare = model.predict(X[505:])
    assert np.mean(rare == 1) > 0.95


def test_vote_counts_neighbours_unless_balanced():
    # A query at angle 0 has 12 class-0 and 4 class-1 rows as its 16 nearest
    # neighbours; 100 more class-0 rows point the other way and only make
    # class 0 common in the bank.
    def at(angles):
        return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

    near0, near1 = at(np.linspace(0.01, 0.12, 12)), at(np.linspace(0.2, 0.23, 4))
    X = np.concatenate([near0, near1, at(np.full(100, np.pi))])
    y = np.repeat([0, 1, 0], [12, 4, 100])
    query = at(np.zeros(1))
    plain = FewShotClassifier(knn_weight=1.0, **FAST).fit(X, y)
    np.testing.assert_array_equal(plain.bank_[2], [1, 1])
    np.testing.assert_allclose(plain.predict_proba(query), [[0.75, 0.25]], rtol=1e-5)
    balanced = clone(plain).set_params(balanced_vote=True).fit(X, y)
    weight = np.array([116 / 112, 116 / 4])  # inverse class shares
    np.testing.assert_allclose(balanced.bank_[2], weight, rtol=1e-5)
    votes = np.array([12, 4]) * weight
    np.testing.assert_allclose(
        balanced.predict_proba(query), [votes / votes.sum()], rtol=1e-5
    )


def test_unlabelled_rows_are_ignored():
    X, y = blobs(20, n_classes=2)
    noise = np.random.default_rng(3).normal(size=(200, X.shape[1]))
    X_all = np.concatenate([X, noise.astype(np.float32)])
    y_all = np.concatenate([y, np.full(200, -1)])
    model = FewShotClassifier(**FAST).fit(X_all, y_all)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    assert len(model.bank_[0]) == len(X)
    assert set(np.unique(model.predict(noise))) <= {0, 1}


def test_unlabelled_value_is_configurable():
    X, y = blobs(10, n_classes=3)
    model = FewShotClassifier(unlabelled=0, **FAST).fit(X, y)
    np.testing.assert_array_equal(model.classes_, [1, 2])
    model = FewShotClassifier(unlabelled=None, **FAST).fit(X, y - 1)
    np.testing.assert_array_equal(model.classes_, [-1, 0, 1])


def test_no_labelled_rows_raises():
    X = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="No labelled rows"):
        FewShotClassifier().fit(X, np.full(4, -1))


def test_invalid_inputs_raise():
    X, y = blobs(5)
    with pytest.raises(ValueError, match="integers"):
        FewShotClassifier().fit(X, y.astype(float))
    with pytest.raises(ValueError, match="inconsistent"):
        FewShotClassifier().fit(X, y[:-1])
    with pytest.raises(ValueError, match="NaN"):
        FewShotClassifier().fit(np.full_like(X, np.nan), y)
    for param in (
        {"knn_weight": 2.0},
        {"dropout": 1.0},
        {"n_neighbors": 0},
        {"max_bank_size": 0},
    ):
        with pytest.raises(ValueError, match=next(iter(param))):
            FewShotClassifier(**param).fit(X, y)
    model = FewShotClassifier(**FAST).fit(X, y)
    with pytest.raises(ValueError, match="shape"):
        model.predict(X[:, :3])
    X_bad = X.copy()
    X_bad[0, 0], X_bad[1, 1] = np.nan, np.inf
    for method in (model.predict, model.predict_proba):
        with pytest.raises(ValueError, match="NaN or infinite"):
            method(X_bad)


def test_single_class_predicts_that_class():
    X, _ = blobs(5)
    model = FewShotClassifier().fit(X, np.full(len(X), 4))
    np.testing.assert_array_equal(model.predict(X), 4)
    np.testing.assert_array_equal(model.predict_proba(X), 1)


def test_unlabelled_rows_may_be_non_finite():
    X, y = blobs(10, n_classes=2)
    X_all = np.concatenate([X, np.full((3, X.shape[1]), np.nan, dtype=np.float32)])
    y_all = np.concatenate([y, np.full(3, -1)])
    model = FewShotClassifier(**FAST).fit(X_all, y_all)
    assert np.mean(model.predict(X) == y) > 0.95


def test_knn_weight_change_after_fit_needs_refit():
    X, y = blobs(10)
    model = FewShotClassifier(knn_weight=0.0, **FAST).fit(X, y)
    with pytest.raises(ValueError, match="call fit again"):
        model.set_params(knn_weight=1.0).predict(X)
    model = FewShotClassifier(knn_weight=1.0, **FAST).fit(X, y)
    with pytest.raises(ValueError, match="call fit again"):
        model.set_params(knn_weight=0.5).predict(X)
    with pytest.raises(ValueError, match="knn_weight"):
        model.set_params(knn_weight=2.0).predict(X)


def test_failed_refit_keeps_the_old_model():
    X, y = blobs(10)
    model = FewShotClassifier(**FAST).fit(X, y)
    with pytest.raises(ValueError):
        model.fit(X, y[:-1])
    assert np.mean(model.predict(X) == y) > 0.95


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        FewShotClassifier().predict(np.zeros((2, 3)))


def test_bank_subsample_is_uniform_with_a_floor_per_class():
    X, y = blobs(300)
    keep = np.concatenate([np.arange(400), np.arange(600, 605)])  # 300, 100, 5
    for seed in range(5):
        model = FewShotClassifier(max_bank_size=100, random_state=seed, **FAST)
        bank_y = model.fit(X[keep], y[keep]).bank_[1]
        counts = np.bincount(bank_y)
        assert counts.sum() == 100 and counts[2] == 5  # 5 rows kept in full
        assert 55 <= counts[0] <= 85 and 15 <= counts[1] <= 35  # about 3 : 1
        bank_y = model.set_params(max_bank_size=30).fit(X[keep], y[keep]).bank_[1]
        counts = np.bincount(bank_y)  # floor is max_bank_size // 3 = 10 rows
        assert counts.sum() == 30 and counts[1] >= 10 and counts[2] == 5


def test_single_predictors():
    X, y = blobs(100)
    model = FewShotClassifier(**FAST)
    for weight in (0.0, 1.0):
        single = clone(model).set_params(knn_weight=weight).fit(X, y)
        assert (single.mlp_ is None) == (weight == 1.0)
        assert (single.bank_ is None) == (weight == 0.0)
        assert np.mean(single.predict(X) == y) > 0.95


def test_predict_matches_predict_proba():
    X, y = blobs(30, spread=3.0)
    model = FewShotClassifier(**FAST).fit(X, y + 1)
    proba = model.predict_proba(X)
    np.testing.assert_array_equal(
        model.predict(X), model.classes_[proba.argmax(axis=1)]
    )


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda"])
def test_transfer_chunks_keep_the_byte_cap_for_wide_features(monkeypatch, device):
    # Wide rows and a large inference block, as with many feature channels on
    # the GPU: each array moved to the device stays within _CHUNK_BYTES.
    import cellcanvas_core.semantic.fewshot as fewshot

    if device != "numpy":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("no CUDA device")
    backend = fewshot._NumPy if device == "numpy" else fewshot._Torch
    X, y = blobs(30, dim=64)
    model = FewShotClassifier(device=device, **FAST).fit(X, y)
    expected = model.predict_proba(X)
    cap = 4 * 64 * 7  # seven rows
    monkeypatch.setattr(fewshot, "_CHUNK_BYTES", cap)
    monkeypatch.setitem(fewshot._BLOCK_MAX, backend.name, 64)
    sent = []
    put = backend.put_float32
    monkeypatch.setattr(
        backend, "put_float32", lambda self, a: sent.append(a.nbytes) or put(self, a)
    )
    np.testing.assert_allclose(model.predict_proba(X), expected, rtol=1e-5, atol=1e-6)
    assert sent and max(sent) <= cap
    assert sum(sent) == X.nbytes
    sent.clear()
    np.testing.assert_array_equal(model.predict(X), expected.argmax(axis=1))
    assert max(sent) <= cap


def test_random_state_makes_fits_repeatable():
    X, y = blobs(30)
    a = FewShotClassifier(**FAST).fit(X, y).predict_proba(X)
    b = FewShotClassifier(**FAST).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    legacy = FewShotClassifier(random_state=np.random.RandomState(0), **FAST)
    assert np.mean(legacy.fit(X, y).predict(X) == y) > 0.95


def test_segmentation_manager_end_to_end():
    """Paint a few voxels of a two-region volume, fit, predict every voxel."""
    rng = np.random.default_rng(0)
    shape = (6, 10, 10)
    truth = np.zeros(shape, dtype=int)
    truth[:, :, 5:] = 1
    features = rng.normal(size=(4, *shape)).astype(np.float32)  # (c, z, y, x)
    features[0] += 6 * truth
    labels = np.zeros(shape, dtype="i4")  # 0 = unannotated
    labels[2:4, 1:4, 1:4] = 1  # stored class 1 -> model class 0
    labels[2:4, 6:9, 6:9] = 2  # stored class 2 -> model class 1
    dataset = DataSet(
        np.zeros((1, *shape)), {"features": features}, labels, np.zeros_like(labels)
    )

    manager = SemanticSegmentationManager(
        DataManager([dataset]), FewShotClassifier(**FAST)
    )
    manager.fit()
    np.testing.assert_array_equal(manager.model.classes_, [0, 1])
    prediction = manager.predict(features)
    assert prediction.shape == shape
    assert np.mean(prediction == truth) > 0.95


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_fits_and_predicts_as_numpy(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")
    X, y = blobs(30)
    query = np.random.default_rng(4).normal(size=(300, 16)).astype(np.float32) * 3
    expected = FewShotClassifier(device="numpy", **FAST).fit(X, y)
    model = FewShotClassifier(device=device, **FAST).fit(X, y)
    for name, value in expected.mlp_.items():
        np.testing.assert_allclose(model.mlp_[name], value, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(
        model.predict_proba(query), expected.predict_proba(query), atol=1e-4
    )
    assert np.mean(model.predict(query) == expected.predict(query)) > 0.99
    # A model fitted with NumPy predicts with PyTorch and vice versa.
    expected.set_params(device=device)
    np.testing.assert_allclose(
        expected.predict_proba(query), model.predict_proba(query), atol=1e-4
    )
    # Pickling leaves the device arrays behind; the copy predicts the same.
    restored = pickle.loads(pickle.dumps(model))
    assert "_on_device" not in restored.__dict__
    np.testing.assert_array_equal(restored.predict(query), model.predict(query))


def test_device_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    X, y = blobs(10)
    model = FewShotClassifier(**FAST).fit(X, y)  # "auto" falls back to NumPy
    assert np.mean(model.predict(X) == y) > 0.95
    with pytest.raises(ImportError, match="PyTorch"):
        model.set_params(device="cuda").fit(X, y)
    assert np.mean(model.set_params(device="numpy").predict(X) == y) > 0.95


@pytest.mark.parametrize("dtype", ["<f8", ">f4", "<f2", "<i4"])
def test_feature_dtypes_predict_alike(dtype):
    X, y = blobs(30)
    model = FewShotClassifier(**FAST).fit(X, y)
    expected = model.predict_proba(X.astype(np.float32))
    np.testing.assert_allclose(
        model.predict_proba(X.astype(dtype)),
        expected,
        atol=2e-3 if dtype == "<f2" else 1e-3,
    )
