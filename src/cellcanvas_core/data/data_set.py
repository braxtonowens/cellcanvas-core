from dataclasses import dataclass, field
from typing import Dict, List, Union

import dask.array as da
import zarr
from zarr import Array

from cellcanvas_core.data._image import open_image
from cellcanvas_core.data._geometry import SpatialTransform, spatial_transform, validate_alignment


def _open_annotations(path, shape, create, level, expected, grouped=False):
    try:
        node = zarr.open(path, mode="r+")
    except FileNotFoundError:
        if not create:
            raise
        if grouped:
            node = zarr.open_group(path, mode="a", zarr_format=3)
        else:
            return zarr.open_array(path, mode="a", shape=shape, dtype="i4", zarr_format=3,
                                   dimension_names=("z", "y", "x"))
    if isinstance(node, zarr.Group):
        metadata = node.attrs.asdict()
        if "ome" in metadata or "multiscales" in metadata:
            node, metadata = open_image(path, level=level, axes=("z", "y", "x"), mode="r+")
            validate_alignment(metadata, level, expected)
        elif grouped:
            if "labels" in node:
                node = node["labels"]
            elif create:
                options = {"dimension_names": ("z", "y", "x")} if node.metadata.zarr_format == 3 else {}
                node = node.create_array("labels", shape=shape, dtype="i4", **options)
            else:
                raise ValueError("Annotation group has no labels array")
        else:
            node, _ = open_image(path, level=0, axes=("z", "y", "x"), mode="r+")
    if not isinstance(node, zarr.Array) or node.shape != shape:
        raise ValueError("Annotation spatial shape does not match the image")
    names = getattr(node.metadata, "dimension_names", None)
    if names is not None and names != ("z", "y", "x"):
        raise ValueError("Annotation dimension_names must be ordered z, y, x")
    return node

@dataclass
class DataSet:
    """CZYX image, feature-first DZYX arrays and shared ZYX annotations.

    D is the classifier feature width, distinct from image channels C.
    Two-dimensional images retain a singleton Z axis. No time or batch axis
    is stored here; model adapters add a leading batch axis to images.
    """

    image: Array
    features: Dict[str, Array]
    labels: Array
    segmentation: Array
    feature_axis: int = 0
    image_metadata: Dict = field(default_factory=dict)
    image_level: int = 0

    def __post_init__(self):
        if self.image.ndim != 4 or any(size == 0 for size in self.image.shape):
            raise ValueError("Images must have nonempty (C, Z, Y, X) axes")
        spatial_shape = self.image.shape[1:]
        if not self.features:
            raise ValueError("At least one feature array is required")
        for name, values in self.features.items():
            if values.ndim != 4:
                raise ValueError(f"Feature {name} needs one channel axis and the image spatial axes")
            axis = self.feature_axis % values.ndim
            if not -values.ndim <= self.feature_axis < values.ndim:
                raise ValueError("feature_axis is outside the feature array")
            spatial = values.shape[:axis] + values.shape[axis + 1:]
            if spatial != spatial_shape or values.shape[axis] == 0:
                raise ValueError(f"Feature {name} does not match the image shape")
        if self.labels.shape != spatial_shape or self.segmentation.shape != spatial_shape:
            raise ValueError("Labels and segmentation must match the image spatial shape (Z, Y, X)")

    @property
    def spatial_transform(self) -> SpatialTransform:
        """Image scale, origin and units for aligning ZYX annotations."""
        return spatial_transform(self.image_metadata, self.image_level)

    @property
    def concatenated_features(self) -> da.Array:
        """Return lazy channel-first features in dictionary insertion order.

        Set feature_axis=-1 for prototype arrays with channels last. Reading
        this property builds a task graph without loading the whole volume.
        """
        features = [da.moveaxis(da.asarray(value), self.feature_axis, 0)
                    for value in self.features.values()]
        return da.concatenate(features, axis=0)

    def __hash__(self):
        """Simple hash function, should updated in the future.

        Note: this might create issues if files are moved.
        """
        return hash(self.image.path)

    @classmethod
    def from_paths(
        cls,
        image_path: str,
        features_path: Union[List[str], str],
        labels_path: str,
        segmentation_path: str,
        make_missing_datasets: bool = False,
        feature_axis: int = 0,
        image_level: int = 0,
    ):
        """Create a DataSet from a set of paths.

        todo: add ability to create missing labels/segmentations
        """
        # get the image
        image, image_metadata = open_image(image_path, level=image_level)
        geometry = spatial_transform(image_metadata, image_level)

        # get the features
        if isinstance(features_path, str):
            features_path = [features_path]
        features = {path: zarr.open_array(path, mode="r") for path in features_path}

        # get the labels
        labels = _open_annotations(
            labels_path, image.shape[1:], make_missing_datasets, image_level, geometry
        )

        # get the segmentation
        segmentation = _open_annotations(
            segmentation_path, image.shape[1:], make_missing_datasets, image_level, geometry
        )

        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
            feature_axis=feature_axis,
            image_metadata=image_metadata,
            image_level=image_level,
        )

    @classmethod
    def from_stores(
        cls,
        image_store,
        features_store,
        labels_store,
        segmentation_store,
        feature_axis: int = 0,
        image_level: int = 0,
    ):
        """Create a DataSet from a set of paths.

        todo: add ability to create missing labels/segmentations
        """
        # TODO rewrite this to copy everything to be local
        
        # get the image
        image, image_metadata = open_image(image_store, level=image_level)
        geometry = spatial_transform(image_metadata, image_level)

        # get the features
        features = {"features": zarr.open_array(features_store, mode="r")}

        # get the labels
        labels = _open_annotations(labels_store, image.shape[1:], True, image_level,
                                   geometry, grouped=True)

        # get the segmentation
        segmentation = _open_annotations(segmentation_store, image.shape[1:], True,
                                         image_level, geometry, grouped=True)

        # TODO start a background thread that triggers downloads of the zarrs
            
        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
            feature_axis=feature_axis,
            image_metadata=image_metadata,
            image_level=image_level,
        )
