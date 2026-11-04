import os
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import dask.array as da
import zarr
from zarr import Array

from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Multiscales

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
    ):
        """Create a DataSet from a set of paths.

        todo: add ability to create missing labels/segmentations
        """
        # get the image
        image = zarr.open(image_path, "r")
        if image.ndim != 4 or any(size == 0 for size in image.shape):
            raise ValueError("Images must have nonempty (C, Z, Y, X) axes")

        # get the features
        if isinstance(features_path, str):
            features_path = [features_path]
        features = {path: zarr.open(path, "r") for path in features_path}

        # get the labels
        if (not os.path.isdir(labels_path)) and make_missing_datasets:
            labels = zarr.open(
                labels_path,
                mode="a",
                shape=image.shape[1:],
                dtype="i4",
                dimension_separator=".",
            )
        else:
            if Multiscales.matches(ZarrLocation(labels_path)):
                labels = zarr.open(os.path.join(labels_path, "0"),
                                   "a")
            else:
                labels = zarr.open(labels_path, "a")

        # get the segmentation
        if (not os.path.isdir(segmentation_path)) and make_missing_datasets:
            segmentation = zarr.open(
                segmentation_path,
                mode="a",
                shape=image.shape[1:],
                dtype="i4",
                dimension_separator=".",
            )

        else:
            segmentation = zarr.open(segmentation_path, mode="a")

        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
            feature_axis=feature_axis,
        )

    @classmethod
    def from_stores(
        cls,
        image_store,
        features_store,
        labels_store,
        segmentation_store,
        feature_axis: int = 0,
    ):
        """Create a DataSet from a set of paths.

        todo: add ability to create missing labels/segmentations
        """
        # TODO rewrite this to copy everything to be local
        
        # get the image
        # TODO fix hardcoded scale for pickathon
        image = zarr.open(zarr.storage.LRUStoreCache(image_store, None), "r")["0"]
        if image.ndim != 4 or any(size == 0 for size in image.shape):
            raise ValueError("Images must have nonempty (C, Z, Y, X) axes")

        # get the features
        features = {"features": zarr.open(zarr.storage.LRUStoreCache(features_store, None), "r")}

        group_name = "labels"
        
        # get the labels
        labels = zarr.open_group(zarr.storage.LRUStoreCache(labels_store, None),
                                 mode="a")
        if group_name in labels:
            labels = labels[group_name]
        else:
            labels = labels.create_dataset(group_name,
                                           shape=image.shape[1:],
                                           dtype="i4")

        # get the segmentation
        segmentation = zarr.open_group(zarr.storage.LRUStoreCache(segmentation_store, None),
                                       mode="a")
        if group_name in segmentation:
            segmentation = segmentation[group_name]
        else:
            segmentation = segmentation.create_dataset(group_name,
                                                       shape=image.shape[1:],
                                                       dtype="i4")

        # TODO start a background thread that triggers downloads of the zarrs
            
        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
            feature_axis=feature_axis,
        )
