from copy import deepcopy

import numpy as np
import pytest
import zarr
from sklearn.tree import DecisionTreeClassifier

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager


def make_ngff(path, axes, zarr_format, image=False):
    group = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    shapes = [(1, 3, 4), (1, 2, 2)]
    for level, shape in enumerate(shapes):
        if image:
            shape = (2,) + shape
        options = {"dimension_names": tuple(axes)} if zarr_format == 3 else {}
        group.create_array(f"resolution/{level}", data=np.zeros(shape, dtype="i4"), **options)
    prefix = [1] if image else []
    zero = [0] if image else []
    multiscale = {
        "axes": [{"name": name, "type": "channel" if name == "c" else "space",
                  **({} if name == "c" else {"unit": "micrometer"})} for name in axes],
        "datasets": [{"path": f"resolution/{level}", "coordinateTransformations": [
            {"type": "scale", "scale": prefix + [2, 3 * (level + 1), 5 * (level + 1)]},
            {"type": "translation", "translation": zero + [7, 11, 13]},
        ]} for level in range(2)],
        "coordinateTransformations": [
            {"type": "scale", "scale": prefix + [10, 20, 30]},
            {"type": "translation", "translation": zero + [100, 200, 300]},
        ],
    }
    if zarr_format == 3:
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}
    else:
        multiscale["version"] = "0.4"
        group.attrs["multiscales"] = [multiscale]
    return group


def setup_project(tmp_path, zarr_format=3, level=0):
    image = make_ngff(tmp_path / "image", "czyx", zarr_format, image=True)
    labels = make_ngff(tmp_path / "labels", "zyx", zarr_format)
    output = make_ngff(tmp_path / "output", "zyx", zarr_format)
    shape = labels[f"resolution/{level}"].shape
    indices = np.arange(np.prod(shape)).reshape(shape)
    zarr.save_array(str(tmp_path / "features"), np.stack([indices, indices % 2]))
    return image, labels, output, indices


def load_project(tmp_path, backend, level=0):
    paths = [str(tmp_path / name) for name in ("image", "features", "labels", "output")]
    if backend == "paths":
        return DataSet.from_paths(*paths, image_level=level)
    return DataSet.from_stores(*(zarr.storage.LocalStore(path) for path in paths), image_level=level)


@pytest.mark.parametrize("backend", ["paths", "stores"])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("level", [0, 1])
def test_aligned_ngff_annotations_survive_prediction(tmp_path, backend, zarr_format, level):
    image, labels, output, indices = setup_project(tmp_path, zarr_format, level)
    before = [deepcopy(group.attrs.asdict()) for group in (image, labels, output)]
    dataset = load_project(tmp_path, backend, level)
    dataset.labels[:] = indices % 2 + 1
    manager = SemanticSegmentationManager(DataManager([dataset]), DecisionTreeClassifier(random_state=0))
    manager.fit()
    manager.predict_to_segmentation(dataset, batch_size=3)
    reopened = load_project(tmp_path, backend, level)
    np.testing.assert_array_equal(reopened.segmentation[:], indices % 2 + 1)
    assert reopened.spatial_transform == dataset.spatial_transform
    for group, metadata in zip((image, labels, output), before):
        assert group.attrs.asdict() == metadata
        assert group.metadata.zarr_format == zarr_format
    # No label arrays are invented alongside the declared NGFF datasets.
    assert "labels" not in labels and "labels" not in output


@pytest.mark.parametrize("backend", ["paths", "stores"])
@pytest.mark.parametrize("target", ["labels", "output"])
@pytest.mark.parametrize("mismatch", ["scale", "origin", "units"])
def test_mismatched_annotations_are_rejected_without_writes(tmp_path, backend, target, mismatch):
    image, labels, output, _ = setup_project(tmp_path)
    group = labels if target == "labels" else output
    metadata = group.attrs["ome"]
    multiscale = metadata["multiscales"][0]
    if mismatch == "units":
        multiscale["axes"][1]["unit"] = "nanometer"
    elif mismatch == "scale":
        multiscale["datasets"][0]["coordinateTransformations"][0]["scale"][0] *= 2
    else:
        multiscale["datasets"][0]["coordinateTransformations"][1]["translation"][0] += 1
    group.attrs["ome"] = metadata
    before = [deepcopy(node.attrs.asdict()) for node in (image, labels, output)]
    with pytest.raises(ValueError, match=f"Annotation spatial {mismatch}"):
        load_project(tmp_path, backend)
    for node, metadata in zip((image, labels, output), before):
        assert node.attrs.asdict() == metadata
        assert "labels" not in node
        np.testing.assert_array_equal(node["resolution/0"][:], 0)


@pytest.mark.parametrize("backend", ["paths", "stores"])
def test_bare_array_axis_names_are_checked(tmp_path, backend):
    for name, values, axes in [
        ("image", np.zeros((2, 1, 3, 4)), "czyx"),
        ("features", np.zeros((5, 1, 3, 4)), None),
        ("labels", np.zeros((1, 3, 4), dtype="i4"), "zxy"),
        ("output", np.zeros((1, 3, 4), dtype="i4"), "zyx"),
    ]:
        zarr.create_array(str(tmp_path / name), data=values,
                           dimension_names=tuple(axes) if axes else None)
    with pytest.raises(ValueError, match="Annotation dimension_names"):
        load_project(tmp_path, backend)
