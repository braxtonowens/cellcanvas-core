from copy import deepcopy

import numpy as np
import pytest
import zarr
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_core.data._image import open_image
from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager


def make_image(path, zarr_format=3):
    group = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    image = np.arange(48).reshape(2, 1, 4, 6)
    options = {"dimension_names": ("c", "z", "y", "x")} if zarr_format == 3 else {}
    group.create_array("fine/image", data=image, **options)
    group.create_array("coarse/image", data=image[:, :, ::2, ::2], **options)
    multiscales = [{
        "axes": [{"name": "c", "type": "channel"}] + [
            {"name": axis, "type": "space", "unit": "micrometer"}
            for axis in ("z", "y", "x")
        ],
        "datasets": [
            {"path": name, "coordinateTransformations": [
                {"type": "scale", "scale": scale},
                {"type": "translation", "translation": [0, 7, 11, 13]},
            ]}
            for name, scale in [("fine/image", [1, 3, 0.5, 0.5]),
                                ("coarse/image", [1, 3, 1, 1])]
        ],
    }]
    if zarr_format == 3:
        group.attrs["ome"] = {"version": "0.5", "multiscales": multiscales}
    else:
        multiscales[0]["version"] = "0.4"
        group.attrs["multiscales"] = multiscales
    return group


@pytest.mark.parametrize("backend", ["paths", "stores"])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_ngff_train_predict_save_reopen(tmp_path, backend, level, zarr_format):
    group = make_image(tmp_path / "image", zarr_format=zarr_format)
    original_metadata = deepcopy(group.attrs.asdict())
    image_path = ["fine/image", "coarse/image"][level]
    expected = group[image_path][:]
    shape = expected.shape[1:]
    indices = np.arange(np.prod(shape)).reshape(shape)
    features = np.stack([indices, indices % 2])
    feature_path = str(tmp_path / "features")
    zarr.save_array(feature_path, features, zarr_format=zarr_format)
    labels_path = str(tmp_path / "labels")
    predictions_path = str(tmp_path / "predictions")

    def load():
        if backend == "paths":
            return DataSet.from_paths(
                str(tmp_path / "image"), feature_path, labels_path,
                predictions_path, make_missing_datasets=True, image_level=level,
            )
        return DataSet.from_stores(
            group.store, zarr.storage.LocalStore(feature_path),
            zarr.storage.LocalStore(labels_path), zarr.storage.LocalStore(predictions_path),
            image_level=level,
        )

    dataset = load()
    np.testing.assert_array_equal(dataset.image[:], expected)
    assert dataset.labels.shape == shape
    assert dataset.image_metadata == original_metadata
    assert dataset.labels.metadata.zarr_format == 3
    assert dataset.segmentation.metadata.zarr_format == 3
    dataset.labels[:] = indices % 2 + 1
    manager = SemanticSegmentationManager(
        DataManager([dataset]), DecisionTreeClassifier(random_state=0)
    )
    manager.fit()
    dataset.segmentation[:] = manager.predict(dataset.concatenated_features, batch_size=5) + 1
    reopened = load()
    np.testing.assert_array_equal(reopened.segmentation[:], indices % 2 + 1)
    np.testing.assert_array_equal(reopened.labels[:], indices % 2 + 1)
    assert reopened.image_metadata == original_metadata
    assert group.attrs.asdict() == original_metadata
    np.testing.assert_array_equal(group[image_path][:], expected)
    metadata = dataset.image_metadata.get("ome", dataset.image_metadata)
    metadata["multiscales"][0]["axes"][1]["unit"] = "nanometer"
    assert group.attrs.asdict() == original_metadata
    assert group.metadata.zarr_format == zarr_format
    assert group[image_path].metadata.zarr_format == zarr_format


@pytest.mark.parametrize("axes", [
    ["z", "y", "x", "c"], ["t", "c", "z", "y", "x"],
])
def test_wrong_axes_fail_before_creating_annotations(tmp_path, axes):
    group = make_image(tmp_path / "image")
    metadata = group.attrs["ome"]
    metadata["multiscales"][0]["axes"] = [{"name": name} for name in axes]
    group.attrs["ome"] = metadata
    with pytest.raises(ValueError, match="axes must be ordered"):
        DataSet.from_paths(str(tmp_path / "image"), "unused", str(tmp_path / "labels"),
                           str(tmp_path / "predictions"), make_missing_datasets=True)
    assert not (tmp_path / "labels").exists()
    assert not (tmp_path / "predictions").exists()


@pytest.mark.parametrize("level", [-1, 2, True])
def test_invalid_level(tmp_path, level):
    group = make_image(tmp_path / "image")
    with pytest.raises(ValueError, match="image_level"):
        open_image(group.store, level=level)


def test_missing_ngff_array_does_not_fall_back_to_zero(tmp_path):
    group = make_image(tmp_path / "image")
    group.create_array("0", data=np.zeros((2, 1, 4, 6)))
    del group["fine/image"]
    with pytest.raises(ValueError, match="dataset is missing"):
        open_image(group.store)


@pytest.mark.parametrize("dimension_names", [None, ("c", "z", "x", "y")])
def test_ngff_dimension_names_must_match_axes(tmp_path, dimension_names):
    group = make_image(tmp_path / "image")
    values = group["fine/image"][:]
    del group["fine/image"]
    group.create_array("fine/image", data=values, dimension_names=dimension_names)
    with pytest.raises(ValueError, match="dimension_names"):
        open_image(group.store)
