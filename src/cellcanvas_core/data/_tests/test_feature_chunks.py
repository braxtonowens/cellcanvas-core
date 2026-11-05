"""Lazy features keep the storage chunks of Zarr arrays."""

import dask
import numpy as np
import zarr

from cellcanvas_core.data.data_set import DataSet

SHAPE = (4, 12, 12)


def test_zarr_features_keep_their_storage_chunks(tmp_path):
    """Storage chunks above dask's chunk-size limit are not split across blocks."""
    features = zarr.create_array(
        str(tmp_path / "features.zarr"),
        shape=(3, *SHAPE),
        chunks=(3, 2, 12, 12),
        dtype="f4",
    )
    features[:] = np.random.default_rng(0).normal(size=(3, *SHAPE))
    dataset = DataSet(
        np.zeros((1, *SHAPE)),
        {"features": features},
        np.zeros(SHAPE, dtype="i4"),
        np.zeros(SHAPE, dtype="i4"),
    )
    with dask.config.set({"array.chunk-size": "1KiB"}):  # smaller than a chunk
        lazy = dataset.concatenated_features
    assert lazy.chunksize == (3, 2, 12, 12)
    np.testing.assert_array_equal(np.asarray(lazy), features[:])


def test_other_arrays_are_unchanged():
    features = np.random.default_rng(0).normal(size=(3, *SHAPE)).astype("f4")
    dataset = DataSet(
        np.zeros((1, *SHAPE)),
        {"features": features},
        np.zeros(SHAPE, dtype="i4"),
        np.zeros(SHAPE, dtype="i4"),
    )
    np.testing.assert_array_equal(np.asarray(dataset.concatenated_features), features)
