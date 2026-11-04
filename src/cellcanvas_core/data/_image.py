"""Read image arrays without guessing NGFF axes or resolution paths."""

from copy import deepcopy

import zarr

from cellcanvas_core.data._geometry import spatial_transform


def open_image(source, level=0, axes=("c", "z", "y", "x"), mode="r"):
    """Return a CZYX array and a copy of its source metadata.

    NGFF 0.5 and legacy 0.4 groups use ordered dataset paths from metadata.
    Bare arrays and legacy groups containing ``0`` remain supported.
    Coordinate transformations are preserved, not applied to pixel values.
    """
    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise ValueError("image_level must be a nonnegative integer")
    root = zarr.open(source, mode=mode)
    metadata = deepcopy(root.attrs.asdict())
    require_dimension_names = False
    if isinstance(root, zarr.Array):
        if level != 0:
            raise ValueError("Bare image arrays only support image_level=0")
        image = root
    elif "ome" in metadata or "multiscales" in metadata:
        ome = metadata.get("ome")
        if ome is not None:
            if not isinstance(ome, dict) or ome.get("version") != "0.5":
                raise ValueError("Only NGFF version 0.5 is supported in the ome namespace")
            if root.metadata.zarr_format != 3:
                raise ValueError("NGFF 0.5 requires Zarr format 3")
            require_dimension_names = True
            multiscales = ome.get("multiscales")
        else:
            multiscales = metadata["multiscales"]
        if not isinstance(multiscales, list) or len(multiscales) != 1:
            raise ValueError("Select a group with exactly one multiscale image")
        multiscale = multiscales[0]
        if ome is None and multiscale.get("version") != "0.4":
            raise ValueError("Only NGFF version 0.4 image metadata is supported")
        declared_axes = multiscale.get("axes", [])
        if not all(isinstance(axis, dict) for axis in declared_axes) or [
            axis.get("name") for axis in declared_axes
        ] != list(axes):
            raise ValueError(f"NGFF image axes must be ordered {', '.join(axes)}")
        datasets = multiscale.get("datasets", [])
        if level >= len(datasets):
            raise ValueError("image_level is not listed in NGFF datasets")
        path = datasets[level].get("path")
        if not isinstance(path, str) or not path or any(
            part in ("", ".", "..") for part in path.split("/")
        ):
            raise ValueError("NGFF dataset path must be relative to the image group")
        try:
            image = root[path]
        except KeyError:
            raise ValueError(f"NGFF image dataset is missing: {path}") from None
    else:
        if level != 0 or "0" not in root:
            raise ValueError("Image group needs NGFF metadata or a legacy array named 0")
        image = root["0"]
    if not isinstance(image, zarr.Array) or image.ndim != len(axes) or any(
        size == 0 for size in image.shape
    ):
        raise ValueError(f"Arrays must have nonempty {tuple(axis.upper() for axis in axes)} axes")
    dimension_names = getattr(image.metadata, "dimension_names", None)
    if (require_dimension_names or dimension_names is not None) and dimension_names != tuple(axes):
        raise ValueError("Array dimension_names must match the declared axes")
    if axes == ("c", "z", "y", "x"):
        spatial_transform(metadata, level)
    return image, metadata
