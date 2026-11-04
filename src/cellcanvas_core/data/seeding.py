"""Automatic background labels from the unlabelled voxels that look typical."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np

from cellcanvas_core.data.data_manager import DataManager

if TYPE_CHECKING:
    from cellcanvas_core.data.data_set import DataSet


class SeededDataManager(DataManager):
    """A `DataManager` that adds background rows drawn from the unlabelled voxels.

    Painted voxels rarely include the background, so a model trained on them
    alone floods the volume with the painted classes. For each dataset, a
    random sample of ``n_candidates`` voxels is read once; its element-wise
    median is the typical feature, and the unlabelled sampled voxels whose
    distance to it lies below the ``percentile``-th percentile of the
    sample's distances become background rows, at most ``max_seeds``.

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
        Number of voxels sampled per dataset.
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
        seed_features, seed_labels = zip(*seeds)
        return (
            da.concatenate([features, *seed_features]),
            da.concatenate([labels, *seed_labels]),
        )

    def seeds(self, dataset: DataSet):
        """Flat voxel indices (sorted) and (n, d) feature rows of the seeds."""
        rng = np.random.default_rng(self.random_state)
        features = da.asarray(dataset.concatenated_features)
        labels = da.asarray(dataset.labels).reshape(-1)
        c, n = features.shape[0], labels.shape[0]
        flat = features.reshape(c, n)
        sample = np.sort(rng.choice(n, min(n, self.n_candidates), replace=False))
        rows, painted = da.compute(
            da.stack([da.take(flat[i], sample) for i in range(c)], axis=1),
            da.take(labels, sample),
        )
        rows = np.asarray(rows, dtype=np.float32)
        distance = np.linalg.norm(rows - np.median(rows, axis=0), axis=1)
        keep = np.flatnonzero(
            (np.asarray(painted) == 0)
            & (distance <= np.percentile(distance, self.percentile))
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
