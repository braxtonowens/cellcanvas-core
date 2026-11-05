"""Automatic background labels from the unlabelled voxels that look typical."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np

from cellcanvas_core.data.data_manager import DataManager

if TYPE_CHECKING:
    from cellcanvas_core.data.data_set import DataSet

# Spatial chunks are read until they hold this many times n_candidates voxels.
_OVERSAMPLE = 4


def _sample_voxels(features, labels, n, rng):
    """About n random voxels, reading only some spatial chunks of the features.

    Chunks are visited in random order until they hold ``_OVERSAMPLE * n``
    voxels (or all are visited), then n voxels are drawn uniformly from them.
    Returns sorted flat voxel indices, their (n, c) float32 feature rows and
    their labels.
    """
    features = da.asarray(features)
    spatial = features.shape[1:]
    chunks = features.chunks[1:]
    labels = da.asarray(labels).rechunk(chunks)
    starts = [np.cumsum((0, *c)) for c in chunks]
    blocks = list(product(*(range(len(c)) for c in chunks)))
    chosen, sizes = [], []
    for b in rng.permutation(len(blocks)):
        block = blocks[b]
        chosen.append(block)
        sizes.append(int(np.prod([chunks[a][block[a]] for a in range(3)])))
        if sum(sizes) >= _OVERSAMPLE * n:
            break
    counts = rng.multivariate_hypergeometric(sizes, min(n, sum(sizes)))
    indices, rows, painted = [], [], []
    for block, size, k in zip(chosen, sizes, counts, strict=True):
        if k == 0:
            continue
        region = tuple(
            slice(starts[a][block[a]], starts[a][block[a] + 1]) for a in range(3)
        )
        values, block_labels = da.compute(
            features[(slice(None), *region)], labels[region]
        )
        local = rng.choice(size, k, replace=False)
        rows.append(np.asarray(values).reshape(len(values), -1)[:, local].T)
        painted.append(np.asarray(block_labels).reshape(-1)[local])
        position = np.unravel_index(local, block_labels.shape)
        position = tuple(p + r.start for p, r in zip(position, region, strict=True))
        indices.append(np.ravel_multi_index(position, spatial))
    indices = np.concatenate(indices)
    order = np.argsort(indices)
    rows = np.concatenate(rows).astype(np.float32)[order]
    return indices[order], rows, np.concatenate(painted)[order]


class SeededDataManager(DataManager):
    """A `DataManager` that adds background rows drawn from the unlabelled voxels.

    Painted voxels rarely include the background, so a model trained on them
    alone floods the volume with the painted classes. For each dataset,
    ``n_candidates`` random voxels are sampled from a random subset of its
    feature chunks (about four times as many voxels as are sampled, so a large
    volume is not read whole); their element-wise median is the typical
    feature, and the unlabelled sampled voxels whose distance to it lies below
    the ``percentile``-th percentile of the sample's distances become
    background rows, at most ``max_seeds``.

    This assumes the typical voxel is background, as in a tomogram of a few
    particles or organelles in ice or cytosol. In a crowded volume, such as an
    in situ tomogram where a painted class (cytosol, membrane) fills most of
    it, the seeds land in that class and teach the model that it is
    background; paint the background instead and use `DataManager`.

    Parameters
    ----------
    datasets : list of DataSet, optional
        As `DataManager`.
    background_label : int
        Stored label value of the background class (``>= 1``; 0 is
        unlabelled). Seeds are returned as ``background_label - 1``, the
        label space of `get_training_data`.
    percentile : float
        Distance percentile below which a sampled voxel is a seed.
    max_seeds : int
        Maximum number of seeds per dataset.
    n_candidates : int
        Number of voxels sampled per dataset; about four times as many voxels
        are read, in whole chunks.
    random_state : int, Generator or None
        Seed for the voxel sample.
    """

    def __init__(
        self,
        datasets=None,
        background_label: int = 1,
        percentile: float = 10.0,
        max_seeds: int = 20000,
        n_candidates: int = 200000,
        random_state=0,
    ):
        super().__init__(datasets)
        if background_label < 1:
            raise ValueError("background_label must be at least 1 (0 is unlabelled)")
        self.background_label = background_label
        self.percentile = percentile
        self.max_seeds = max_seeds
        self.n_candidates = n_candidates
        self.random_state = random_state

    def get_training_data(self):
        """Painted rows of every dataset, then its background seeds."""
        features, labels = super().get_training_data()
        seeds = [self.seed_rows(dataset) for dataset in self.datasets]
        seeds = [s for s in seeds if s is not None]
        if not seeds:
            return features, labels
        seed_features, seed_labels = zip(*seeds, strict=True)
        return (
            da.concatenate([features, *seed_features]),
            da.concatenate([labels, *seed_labels]),
        )

    def seeds(self, dataset: DataSet):
        """Flat voxel indices (sorted) and (n, d) feature rows of the seeds."""
        rng = np.random.default_rng(self.random_state)
        sample, rows, painted = _sample_voxels(
            dataset.concatenated_features, dataset.labels, self.n_candidates, rng
        )
        if not np.isfinite(rows).all():
            raise ValueError("Features contain NaN or infinite values")
        distance = np.linalg.norm(rows - np.median(rows, axis=0), axis=1)
        keep = np.flatnonzero(
            (painted == 0) & (distance <= np.percentile(distance, self.percentile))
        )
        if len(keep) > self.max_seeds:
            keep = np.sort(rng.choice(keep, self.max_seeds, replace=False))
        return sample[keep], rows[keep]

    def seed_indices(self, dataset: DataSet):
        """Flat voxel indices of the seeds of one dataset (for display)."""
        return self.seeds(dataset)[0]

    def seed_rows(self, dataset: DataSet):
        """(n, d) features and (n,) labels of the seeds as dask arrays, or None."""
        indices, rows = self.seeds(dataset)
        if len(indices) == 0:
            return None
        labels = da.full(len(indices), self.background_label - 1, dtype=np.int64)
        return da.from_array(rows), labels
