from unittest.mock import Mock

import dask.array as da
import numpy as np
import pytest
import zarr
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager


@pytest.mark.parametrize("shape", [(2, 3, 7), (1, 3, 7)])
@pytest.mark.parametrize("feature_axis", [0, -1])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_prediction_blocks_preserve_voxels_labels_and_source(
    tmp_path, shape, feature_axis, zarr_format
):
    indices = np.arange(np.prod(shape)).reshape(shape)
    image = np.stack([indices, indices + 1000])
    expected = (indices % 2 * 2 + 1).astype("u1")
    painting = expected.copy()
    painting.ravel()[::5] = 0
    original_painting = painting.copy()
    features = {"identity": np.stack([indices, indices + 100]),
                "parity": (indices % 2)[None]}
    if feature_axis == -1:
        features = {key: np.moveaxis(value, 0, -1) for key, value in features.items()}
    feature_arrays = {}
    for name, values in features.items():
        path = str(tmp_path / name)
        zarr.save_array(path, values, zarr_format=zarr_format)
        feature_arrays[name] = zarr.open_array(path, mode="r")
    output_path = str(tmp_path / "segmentation")
    target = zarr.open_array(output_path, mode="w", shape=shape, dtype="u1",
                             chunks=(1, 2, 3), zarr_format=zarr_format)
    target[:] = 99
    writes = []
    rows_seen = []

    class RecordingTarget:
        dtype = target.dtype
        shape = target.shape

        def __setitem__(self, region, values):
            writes.append(values.size)
            target[region] = values

    class RecordingTree(DecisionTreeClassifier):
        def predict(self, rows):
            rows_seen.append(len(rows))
            assert isinstance(rows, np.ndarray)
            return super().predict(rows)

    dataset = DataSet(image, feature_arrays, painting, RecordingTarget(),
                      feature_axis=feature_axis)
    manager = SemanticSegmentationManager(DataManager([dataset]), RecordingTree(random_state=0))
    manager.fit()
    # The saving path must not call the API that builds a full output volume.
    manager.predict = Mock(side_effect=AssertionError("full-volume prediction called"))
    manager.predict_to_segmentation(dataset, batch_size=5)
    assert max(rows_seen) <= 5
    assert max(writes) <= 5
    assert sum(rows_seen) == sum(writes) == np.prod(shape)
    reopened = zarr.open_array(output_path, mode="r")
    np.testing.assert_array_equal(reopened[:], expected)
    assert reopened.metadata.zarr_format == zarr_format
    np.testing.assert_array_equal(painting, original_painting)
    np.testing.assert_array_equal(image, np.stack([indices, indices + 1000]))
    for name, values in features.items():
        np.testing.assert_array_equal(feature_arrays[name][:], values)


def make_dataset(dtype="u1"):
    return DataSet(np.ones((2, 1, 2, 3)), {"features": np.ones((4, 1, 2, 3))},
                   np.ones((1, 2, 3), dtype="i4"), np.zeros((1, 2, 3), dtype=dtype))


@pytest.mark.parametrize("classes", [
    np.array([-1, 0]), np.array([0.5, 1.5]), np.array([255, 0]), np.array([[0], [1]]),
])
def test_invalid_classes_are_not_written(classes):
    dataset = make_dataset()
    model = Mock()
    model.predict.return_value = classes
    manager = SemanticSegmentationManager(DataManager([dataset]), model)
    with pytest.raises(ValueError):
        manager.predict_to_segmentation(dataset, batch_size=2)
    np.testing.assert_array_equal(dataset.segmentation, 0)


def test_largest_representable_class_does_not_wrap():
    dataset = make_dataset()
    model = Mock()
    model.predict.side_effect = lambda rows: np.full(len(rows), 254, dtype="u1")
    SemanticSegmentationManager(DataManager([dataset]), model).predict_to_segmentation(dataset, batch_size=2)
    np.testing.assert_array_equal(dataset.segmentation, 255)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_does_not_write(batch_size):
    dataset = make_dataset()
    model = Mock()
    with pytest.raises(ValueError, match="batch_size"):
        SemanticSegmentationManager(DataManager([dataset]), model).predict_to_segmentation(dataset, batch_size)
    model.predict.assert_not_called()
    np.testing.assert_array_equal(dataset.segmentation, 0)


def test_noninteger_storage_is_rejected_before_prediction():
    dataset = make_dataset(dtype="f4")
    model = Mock()
    with pytest.raises(ValueError, match="integer dtype"):
        SemanticSegmentationManager(DataManager([dataset]), model).predict_to_segmentation(dataset)
    model.predict.assert_not_called()


def test_untrained_model_does_not_write():
    dataset = make_dataset()
    manager = SemanticSegmentationManager(DataManager([dataset]), DecisionTreeClassifier())
    with pytest.raises(NotFittedError):
        manager.predict_to_segmentation(dataset, batch_size=2)
    np.testing.assert_array_equal(dataset.segmentation, 0)


class ReadRecorder:
    """An array that records which regions are read."""

    def __init__(self, values):
        self.values, self.reads = values, []
        self.shape, self.dtype, self.ndim = values.shape, values.dtype, values.ndim

    def __getitem__(self, key):
        self.reads.append(tuple((k.start, k.stop) for k in key[1:]))
        return self.values[key]


def test_prediction_reads_each_feature_chunk_once():
    shape = (4, 3, 7)
    indices = np.arange(np.prod(shape)).reshape(shape)
    recorder = ReadRecorder(np.stack([indices % 2, indices % 3]).astype("f4"))
    features = da.from_array(recorder, chunks=(2, 2, 3, 4), asarray=True)  # 4 chunks
    painting = (indices % 2 + 1).astype("u1")
    target = np.zeros(shape, dtype="u1")
    dataset = DataSet(np.zeros((1, *shape)), {"features": features}, painting, target)
    manager = SemanticSegmentationManager(
        DataManager([dataset]), DecisionTreeClassifier(random_state=0)
    )
    manager.fit()
    recorder.reads.clear()
    manager.predict_to_segmentation(dataset, batch_size=5)
    assert sorted(recorder.reads) == sorted(set(recorder.reads))  # no chunk twice
    assert len(recorder.reads) == 4
    np.testing.assert_array_equal(target, painting)


def test_prediction_blocks_fall_back_to_batches_for_large_chunks():
    features = da.zeros((3, 4, 3, 7), chunks=(3, 4, 3, 7))
    blocks = list(SemanticSegmentationManager._read_blocks(features, 5, max_bytes=1))
    assert all(np.prod([s.stop - s.start for s in block]) <= 5 for block in blocks)
    covered = np.zeros((4, 3, 7), dtype=int)
    for block in blocks:
        covered[block] += 1
    assert (covered == 1).all()


def test_prediction_blocks_are_planned_lazily():
    # metadata only: a 512^3 grid of 8^3 chunks has 262,144 regions
    import tracemalloc
    from types import SimpleNamespace

    grid = ((8,) * 64,) * 3
    features = SimpleNamespace(
        shape=(2, 512, 512, 512), chunks=((2,), *grid), dtype=np.dtype("f4")
    )
    plan = SemanticSegmentationManager._read_blocks
    tracemalloc.start()
    try:
        chunk_blocks = plan(features, 512)
        first = [next(chunk_blocks) for _ in range(3)]
        next(plan(features, 512, max_bytes=1))  # the batch_size tiles
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2**20
    assert first[0] == (slice(0, 8), slice(0, 8), slice(0, 8))
    assert first[2] == (slice(0, 8), slice(0, 8), slice(16, 24))
