from copy import deepcopy

import numpy as np
import pytest
import zarr
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_core.data._geometry import SpatialTransform, spatial_transform
from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager


def make_image(path, zarr_format=3):
    group = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    image = np.arange(24).reshape(2, 1, 3, 4)
    options = {"dimension_names": ("c", "z", "y", "x")} if zarr_format == 3 else {}
    group.create_array("fine", data=image, **options)
    group.create_array("coarse", data=image[:, :, ::2, ::2], **options)
    multiscale = {
        "axes": [{"name": "c", "type": "channel"}] + [
            {"name": name, "type": "space", "unit": unit}
            for name, unit in zip(("z", "y", "x"), ("nanometer", "micrometer", "micrometer"))
        ],
        "datasets": [
            {"path": name, "coordinateTransformations": [
                {"type": "scale", "scale": scale},
                {"type": "translation", "translation": translation},
            ]}
            for name, scale, translation in [
                ("fine", [1, 2, 3, 5], [0, 7, 11, 13]),
                ("coarse", [1, 4, 6, 10], [0, 17, 19, 23]),
            ]
        ],
        "coordinateTransformations": [
            {"type": "scale", "scale": [1, 10, 20, 30]},
            {"type": "translation", "translation": [0, 100, 200, 300]},
        ],
    }
    if zarr_format == 3:
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}
    else:
        multiscale["version"] = "0.4"
        group.attrs["multiscales"] = [multiscale]
    return group


@pytest.mark.parametrize("backend", ["paths", "stores"])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("level", [0, 1])
def test_coordinates_survive_paint_predict_save_reopen(tmp_path, backend, zarr_format, level):
    group = make_image(tmp_path / "image", zarr_format)
    metadata = deepcopy(group.attrs.asdict())
    shape = group[("fine", "coarse")[level]].shape[1:]
    indices = np.arange(np.prod(shape)).reshape(shape)
    feature_path = str(tmp_path / "features")
    zarr.save_array(feature_path, np.stack([indices, indices % 2]))
    labels_path, output_path = str(tmp_path / "labels"), str(tmp_path / "output")

    def load():
        if backend == "paths":
            return DataSet.from_paths(str(tmp_path / "image"), feature_path,
                                      labels_path, output_path, True, image_level=level)
        return DataSet.from_stores(group.store, zarr.storage.LocalStore(feature_path),
                                   zarr.storage.LocalStore(labels_path),
                                   zarr.storage.LocalStore(output_path), image_level=level)

    dataset = load()
    expected = [
        SpatialTransform((20, 60, 150), (170, 420, 690),
                         ("nanometer", "micrometer", "micrometer")),
        SpatialTransform((40, 120, 300), (270, 580, 990),
                         ("nanometer", "micrometer", "micrometer")),
    ][level]
    assert dataset.spatial_transform == expected
    # Known physical coordinate of voxel (z=0, y=1, x=1).
    world = np.array((0, 1, 1)) * expected.scale + expected.translation
    np.testing.assert_array_equal(world, [(170, 480, 840), (270, 700, 1290)][level])
    dataset.labels[:] = indices % 2 + 1
    manager = SemanticSegmentationManager(DataManager([dataset]), DecisionTreeClassifier(random_state=0))
    manager.fit()
    manager.predict_to_segmentation(dataset, batch_size=3)
    reopened = load()
    assert reopened.spatial_transform == expected
    np.testing.assert_array_equal(reopened.labels[:], indices % 2 + 1)
    np.testing.assert_array_equal(reopened.segmentation[:], indices % 2 + 1)
    assert group.attrs.asdict() == metadata


@pytest.mark.parametrize("transforms", [
    [],
    [{"type": "translation", "translation": [0, 1, 2, 3]}],
    [{"type": "scale", "scale": [1, 2, 3]}],
    [{"type": "scale", "scale": [1, 0, 2, 3]}],
    [{"type": "scale", "scale": [1, -1, 2, 3]}],
    [{"type": "scale", "scale": [1, float("nan"), 2, 3]}],
    [{"type": "scale", "path": "transforms/scale"}],
    [{"type": "rotation", "rotation": [1, 2, 3, 4]}],
])
def test_bad_coordinates_fail_before_creating_annotations(tmp_path, transforms):
    group = make_image(tmp_path / "image")
    metadata = group.attrs["ome"]
    metadata["multiscales"][0]["datasets"][0]["coordinateTransformations"] = transforms
    group.attrs["ome"] = metadata
    with pytest.raises(ValueError, match="NGFF"):
        DataSet.from_paths(str(tmp_path / "image"), "unused", str(tmp_path / "labels"),
                           str(tmp_path / "output"), True)
    assert not (tmp_path / "labels").exists()
    assert not (tmp_path / "output").exists()


def test_bare_image_uses_pixel_coordinates():
    dataset = DataSet(np.zeros((2, 1, 3, 4)), {"features": np.zeros((5, 1, 3, 4))},
                      np.zeros((1, 3, 4), dtype="i4"), np.zeros((1, 3, 4), dtype="i4"))
    assert dataset.spatial_transform == SpatialTransform()


def test_level_only_transforms_and_unspecified_units(tmp_path):
    group = make_image(tmp_path / "image")
    metadata = group.attrs.asdict()
    multiscale = metadata["ome"]["multiscales"][0]
    del multiscale["coordinateTransformations"]
    for axis in multiscale["axes"]:
        axis.pop("unit", None)
    result = spatial_transform(metadata)
    assert result == SpatialTransform((2, 3, 5), (7, 11, 13))
