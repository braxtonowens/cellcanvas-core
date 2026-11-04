import dask.array as da
import numpy as np
import pytest
import zarr
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager


def make_dataset(axis=0):
    image = np.arange(24).reshape(2, 3, 4)
    features = {"a": np.stack([image, image + 100]),
                "b": np.stack([image + 200, image + 300, image + 400])}
    if axis == -1:
        features = {key: np.moveaxis(value, 0, -1) for key, value in features.items()}
    labels = (image % 2 + 1).astype("i4")
    labels.flat[0] = 0
    return DataSet(np.stack([image, image + 1000]), features,
                   labels, np.zeros_like(labels), feature_axis=axis)


@pytest.mark.parametrize("axis", [0, -1])
def test_feature_rows_keep_their_voxel_and_channel_order(axis):
    dataset = make_dataset(axis)
    assert isinstance(dataset.concatenated_features, da.Array)
    features, labels = DataManager([dataset]).get_training_data()
    expected = np.arange(1, 24)[:, None] + np.arange(0, 500, 100)[None, :]
    np.testing.assert_array_equal(features.compute(), expected)
    np.testing.assert_array_equal(labels.compute(), np.arange(1, 24) % 2)


def test_reject_mismatched_shapes():
    dataset = make_dataset()
    with pytest.raises(ValueError, match="does not match"):
        DataSet(dataset.image, {"bad": np.zeros((2, 3, 4, 5))},
                dataset.labels, dataset.segmentation)
    with pytest.raises(ValueError, match="must match"):
        DataSet(dataset.image, dataset.features, np.zeros((2, 3)), dataset.segmentation)


def test_no_painted_labels_has_a_useful_error():
    dataset = make_dataset()
    dataset.labels[:] = 0
    with pytest.raises(ValueError, match="Paint"):
        DataManager([dataset]).get_training_data()


def test_image_axes_and_single_slice_2d():
    image = np.zeros((2, 1, 3, 4))
    labels = np.zeros((1, 3, 4), dtype="i4")
    dataset = DataSet(image, {"features": np.ones((5, 1, 3, 4))}, labels, labels.copy())
    assert dataset.image.shape == (2, 1, 3, 4)
    assert dataset.labels.shape == (1, 3, 4)
    assert dataset.image[np.newaxis].shape == (1, 2, 1, 3, 4)
    with pytest.raises(ValueError, match="C, Z, Y, X"):
        DataSet(image[0], dataset.features, labels, labels.copy())


def test_train_predict_save_reopen_with_bounded_batches(tmp_path):
    dataset = make_dataset()
    rows_seen = []

    class RecordingTree(DecisionTreeClassifier):
        def predict(self, features):
            rows_seen.append(len(features))
            assert isinstance(features, np.ndarray)
            return super().predict(features)

    manager = SemanticSegmentationManager(DataManager([dataset]), RecordingTree(random_state=0))
    manager.fit()
    path = str(tmp_path / "features")
    zarr.save_array(path, dataset.concatenated_features.compute())
    predictions = manager.predict(zarr.open(path, mode="r"), batch_size=5)
    assert max(rows_seen) <= 5
    assert sum(rows_seen) == 24
    np.testing.assert_array_equal(predictions.ravel()[1:], dataset.labels.ravel()[1:] - 1)
    stored = predictions + 1
    output = str(tmp_path / "predictions")
    zarr.save_array(output, stored)
    np.testing.assert_array_equal(zarr.open(output, mode="r")[:], stored)
    assert set(np.unique(stored)) == {1, 2}
