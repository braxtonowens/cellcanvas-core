import dask.array as da
import numpy as np
import pytest
import zarr
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.data.seeding import SeededDataManager
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager

SHAPE = (4, 20, 20)


def volume(seed=0):
    """Features (c, z, y, x): typical voxels near one vector, a bright cube far away.

    Labels: class 1 (background) and class 2 (the cube) painted on one slice only.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(3, *SHAPE)).astype(np.float32)
    cube = np.zeros(SHAPE, dtype=bool)
    cube[1:3, 5:12, 5:12] = True
    features[0][cube] += 8.0
    labels = np.zeros(SHAPE, dtype="i4")
    labels[2, 15:18, 15:18] = 1
    labels[2, 6:10, 6:10] = 2
    return features, labels, cube


def dataset(features, labels):
    return DataSet(
        np.zeros((1, *SHAPE)), {"features": features}, labels, np.zeros_like(labels)
    )


def test_seeds_are_unlabelled_typical_voxels():
    features, labels, cube = volume()
    manager = SeededDataManager(
        [dataset(features, labels)], background_label=1, max_seeds=50
    )
    idx = manager.seed_indices(manager.datasets[0])
    assert 0 < len(idx) <= 50
    assert np.all(labels.reshape(-1)[idx] == 0)
    assert not np.any(cube.reshape(-1)[idx])  # the bright cube is never typical


def test_training_data_appends_background_rows():
    features, labels, _ = volume()
    ds = dataset(features, labels)
    X0, y0 = DataManager([ds]).get_training_data()
    manager = SeededDataManager([ds], background_label=1, max_seeds=30)
    X, y = manager.get_training_data()
    X, y, X0, y0 = X.compute(), y.compute(), X0.compute(), y0.compute()
    n_seeds = len(manager.seed_indices(ds))
    assert X.shape == (len(X0) + n_seeds, 3) and y.shape == (len(y0) + n_seeds,)
    np.testing.assert_array_equal(X[: len(X0)], X0)
    np.testing.assert_array_equal(y[: len(y0)], y0)
    np.testing.assert_array_equal(y[len(y0) :], 0)  # background_label 1 -> class 0
    idx = manager.seed_indices(ds)
    np.testing.assert_array_equal(X[len(X0) :], features.reshape(3, -1)[:, idx].T)


def test_fully_labelled_dataset_adds_nothing():
    features, labels, _ = volume()
    ds = dataset(features, np.ones_like(labels))
    manager = SeededDataManager([ds], background_label=1)
    assert manager.seed_rows(ds) is None
    X, y = manager.get_training_data()
    assert X.shape[0] == labels.size and y.shape[0] == labels.size


def test_reproducible_and_validated():
    features, labels, _ = volume()
    ds = dataset(features, labels)
    # the sample must matter here
    few = {"background_label": 1, "n_candidates": 300, "max_seeds": 20}
    a = SeededDataManager([ds], random_state=3, **few).seed_indices(ds)
    b = SeededDataManager([ds], random_state=3, **few).seed_indices(ds)
    np.testing.assert_array_equal(a, b)
    c = SeededDataManager([ds], random_state=4, **few).seed_indices(ds)
    assert not np.array_equal(a, c)
    with pytest.raises(ValueError):
        SeededDataManager([ds], background_label=0)
    with pytest.raises(TypeError):  # required: no default that could collide
        SeededDataManager([ds])
    with pytest.raises(ValueError):
        SeededDataManager([ds], background_label=1, n_candidates=0)


@pytest.mark.parametrize("random_state", [None, np.random.default_rng(5)])
def test_seeds_are_stable_across_calls(random_state):
    """The seeds shown are the seeds trained on, also for a Generator or None."""
    features, labels, _ = volume()
    ds = dataset(features, labels)
    manager = SeededDataManager(
        [ds],
        background_label=1,
        n_candidates=300,
        max_seeds=20,
        random_state=random_state,
    )
    shown = manager.seed_indices(ds)
    np.testing.assert_array_equal(manager.seed_indices(ds), shown)
    X, _ = manager.get_training_data()
    seed_rows = X.compute()[-len(shown) :]
    np.testing.assert_array_equal(seed_rows, features.reshape(3, -1)[:, shown].T)


def test_same_shaped_datasets_get_different_seeds():
    features, labels, _ = volume()
    first, second = dataset(features, labels), dataset(features, labels)
    manager = SeededDataManager(
        [first, second], background_label=1, n_candidates=300, max_seeds=20
    )
    assert not np.array_equal(manager.seed_indices(first), manager.seed_indices(second))


def test_zarr_labels(tmp_path):
    """Labels stored as a Zarr array are read lazily and give the same seeds."""
    features, labels, _ = volume()
    stored = zarr.create_array(
        str(tmp_path / "labels.zarr"), shape=SHAPE, chunks=(1, 20, 20), dtype="i4"
    )
    stored[:] = labels
    few = {"background_label": 1, "n_candidates": 300, "max_seeds": 20}
    ds, ds_zarr = dataset(features, labels), dataset(features, stored)
    np.testing.assert_array_equal(
        SeededDataManager([ds], **few).seed_indices(ds),
        SeededDataManager([ds_zarr], **few).seed_indices(ds_zarr),
    )


def test_seeds_keep_the_cube_from_flooding():
    """Only the cube is painted: without seeds everything is cube; with them, not."""
    features, labels, cube = volume()
    labels[labels == 1] = 0  # no painted background
    ds = dataset(features, labels)
    model = DecisionTreeClassifier(random_state=0)
    flooded = SemanticSegmentationManager(DataManager([ds]), model)
    flooded.fit()
    assert np.all(flooded.predict(features) == 1)  # single class -> cube everywhere
    seeded = SemanticSegmentationManager(
        SeededDataManager([ds], background_label=1), model
    )
    seeded.fit()
    prediction = seeded.predict(features)
    assert np.mean(prediction[cube] == 1) > 0.9
    assert np.mean(prediction[~cube] == 0) > 0.9


class Recorder:
    """An array that records which regions are read."""

    def __init__(self, values):
        self.values, self.reads = values, []
        self.shape, self.dtype, self.ndim = values.shape, values.dtype, values.ndim

    def __getitem__(self, key):
        self.reads.append(key)
        return self.values[key]


def test_only_some_chunks_are_read():
    features, labels, _ = volume()
    recorder = Recorder(features)
    chunked = da.from_array(recorder, chunks=(3, 1, 10, 10), asarray=True)  # 16 chunks
    ds = DataSet(
        np.zeros((1, *SHAPE)), {"features": chunked}, labels, np.zeros_like(labels)
    )
    idx = SeededDataManager(
        [ds], background_label=1, n_candidates=50, max_seeds=10
    ).seed_indices(ds)
    regions = {tuple((k.start, k.stop) for k in key[1:]) for key in recorder.reads}
    assert 0 < len(regions) < 16  # 50 voxels of 100-voxel chunks: two chunks read
    z, y, x = np.unravel_index(idx, SHAPE)
    for point in zip(z, y, x, strict=True):
        assert any(
            all(a <= p < b for p, (a, b) in zip(point, r, strict=True)) for r in regions
        )


def test_non_finite_features_raise():
    features, labels, _ = volume()
    features[:] = np.nan
    ds = dataset(features, labels)
    with pytest.raises(ValueError, match="NaN or infinite"):
        SeededDataManager([ds], background_label=1).get_training_data()
