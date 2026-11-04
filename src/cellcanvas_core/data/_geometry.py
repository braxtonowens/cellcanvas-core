"""Spatial coordinates for the selected NGFF image level."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialTransform:
    """ZYX coordinates: world = voxel * scale + translation.

    Missing metadata means pixel coordinates with unspecified units.
    """

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    units: tuple[str | None, str | None, str | None] = (None, None, None)


def spatial_transform(metadata, level=0):
    """Compose level transforms followed by multiscale transforms.

    Supports inline scale and translation vectors from NGFF 0.4/0.5.
    Unsupported transforms raise an error rather than assuming unit spacing.
    """
    image_metadata = metadata.get("ome", metadata)
    if "multiscales" not in image_metadata:
        return SpatialTransform()
    multiscales = image_metadata["multiscales"]
    if not isinstance(multiscales, list) or len(multiscales) != 1:
        raise ValueError("Select a group with exactly one multiscale image")
    multiscale = multiscales[0]
    axes = multiscale.get("axes", [])
    if [axis.get("name") for axis in axes] != ["c", "z", "y", "x"]:
        raise ValueError("Spatial coordinates require CZYX image axes")
    datasets = multiscale.get("datasets", [])
    if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level < len(datasets):
        raise ValueError("image_level is not listed in NGFF datasets")
    scale = np.ones(4)
    translation = np.zeros(4)
    transform_lists = [datasets[level].get("coordinateTransformations")]
    if "coordinateTransformations" in multiscale:
        transform_lists.append(multiscale["coordinateTransformations"])
    for transforms in transform_lists:
        if not isinstance(transforms, list) or not all(
            isinstance(transform, dict) for transform in transforms
        ) or [
            transform.get("type") for transform in transforms
        ] not in (["scale"], ["scale", "translation"]):
            raise ValueError("NGFF transforms require one scale followed by an optional translation")
        for transform in transforms:
            kind = transform["type"]
            try:
                vector = np.asarray(transform[kind], dtype=float)
            except (KeyError, TypeError, ValueError):
                raise ValueError("NGFF transforms require inline numeric vectors") from None
            if vector.shape != (4,) or not np.all(np.isfinite(vector)):
                raise ValueError("NGFF transform vectors need four finite CZYX values")
            if kind == "scale":
                if np.any(vector <= 0):
                    raise ValueError("NGFF scale values must be positive")
                scale *= vector
                translation *= vector
            else:
                translation += vector
    units = tuple(axis.get("unit") for axis in axes[1:])
    return SpatialTransform(tuple(scale[1:]), tuple(translation[1:]), units)
