import numpy as np
import pytest
import zarr

from cellcanvas_core.data.data_set import DataSet


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("feature_axis", [0, -1])
def test_paths_preserve_inputs_and_reopen_annotations(tmp_path, existing, feature_axis):
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    features = image[np.newaxis] * 2
    if feature_axis == -1:
        features = np.moveaxis(features, 0, -1)
    image = np.stack([image, image + 100])
    paths = {name: str(tmp_path / name) for name in
             ("image", "features", "labels", "segmentation")}
    zarr.save_array(paths["image"], image)
    zarr.save_array(paths["features"], features)
    if existing:
        zarr.save_array(paths["labels"], np.full(image.shape[1:], 2, dtype="i4"))
        zarr.save_array(paths["segmentation"], np.full(image.shape[1:], 3, dtype="i4"))

    def reopen():
        return DataSet.from_paths(
            paths["image"], paths["features"], paths["labels"],
            paths["segmentation"], make_missing_datasets=not existing,
            feature_axis=feature_axis,
        )

    dataset = reopen()
    assert dataset.concatenated_features.shape == (1, 2, 3, 4)
    np.testing.assert_array_equal(dataset.labels[:], 2 if existing else 0)
    np.testing.assert_array_equal(dataset.segmentation[:], 3 if existing else 0)
    dataset.labels[0, 1, 2] = 5
    dataset.segmentation[1, 2, 3] = 6
    reopened = reopen()
    assert reopened.labels[0, 1, 2] == 5
    assert reopened.segmentation[1, 2, 3] == 6
    np.testing.assert_array_equal(reopened.image[:], image)
    np.testing.assert_array_equal(reopened.features[paths["features"]][:], features)


@pytest.mark.parametrize("feature_axis", [0, -1])
def test_stores_create_and_reopen_annotations(tmp_path, feature_axis):
    stores = [zarr.DirectoryStore(str(tmp_path / name)) for name in
              ("image", "features", "labels", "segmentation")]
    image = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
    zarr.group(store=stores[0]).create_dataset("0", data=image)
    features = image if feature_axis == 0 else np.moveaxis(image, 0, -1)
    zarr.array(features, store=stores[1])
    dataset = DataSet.from_stores(*stores, feature_axis=feature_axis)
    np.testing.assert_array_equal(dataset.concatenated_features.compute(), image)
    np.testing.assert_array_equal(dataset.labels[:], 0)
    np.testing.assert_array_equal(dataset.segmentation[:], 0)
    dataset.labels[0, 1, 2] = 5
    dataset.segmentation[1, 2, 3] = 6
    reopened = DataSet.from_stores(*stores, feature_axis=feature_axis)
    assert reopened.labels[0, 1, 2] == 5
    assert reopened.segmentation[1, 2, 3] == 6
    np.testing.assert_array_equal(reopened.image[:], image)
    np.testing.assert_array_equal(reopened.features["features"][:], features)
