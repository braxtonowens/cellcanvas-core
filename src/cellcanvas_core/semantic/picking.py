"""Ranked particle picks from class probabilities."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def pick(probability, radius, threshold=0.05, edge=0.5, spacing=(1.0, 1.0, 1.0)):
    """Ranked picks of one particle class.

    The probability is smoothed with a Gaussian of ``radius / 2``; local maxima
    of the smoothed probability above ``threshold`` that lie at least
    ``edge * radius`` from every face are candidates, kept in score order with
    greedy non-maximum suppression at ``radius`` and refined to sub-voxel
    positions. The score is the smoothed probability at the pick; since the
    smoothing depends on the radius, compare scores and set thresholds within
    a class, not across classes.

    Meant for particle classes. A large solid region (background, membrane,
    cytosol) has a flat probability and gives a grid of picks spaced about
    ``radius`` apart. The smoothing merges maxima closer than about
    ``1.5 * radius`` into one pick at the stronger; when that is a peak at a
    face, the merged pick falls in the edge margin and is dropped, taking the
    weaker particle with it.

    Parameters
    ----------
    probability : array-like
        (z, y, x) probability of the class.
    radius : float
        Particle radius in voxels of the y/x axes; positive.
    threshold : float
        Minimum smoothed probability of a pick. Keep it low and choose the
        operating point later from the scores.
    edge : float
        Picks closer than ``edge * radius`` to a face are dropped; at least 0.
    spacing : tuple of float
        Voxel spacing (z, y, x) relative to the y/x spacing, for anisotropic z.

    Returns
    -------
    points : np.ndarray
        (n, 3) z, y, x in voxels, best first.
    scores : np.ndarray
        (n,) smoothed probability at each pick.
    """
    prob = np.asarray(probability, dtype=np.float32)
    spacing = np.asarray(spacing, dtype=float)
    if prob.ndim != 3:
        raise ValueError("probability must be a (z, y, x) volume")
    if spacing.shape != (3,) or not np.all(spacing > 0):
        raise ValueError("spacing must be three positive numbers")
    if not radius > 0:
        raise ValueError("radius must be positive")
    if not edge >= 0:
        raise ValueError("edge must be at least 0")
    smooth = ndi.gaussian_filter(prob, sigma=radius / 2 / spacing, mode="nearest")
    peaks = (smooth > threshold) & (
        smooth >= ndi.maximum_filter(smooth, size=3, mode="nearest")
    )
    idx = np.argwhere(peaks)
    margin = edge * radius / spacing
    inside = (idx >= margin).all(1) & (idx <= np.array(prob.shape) - 1 - margin).all(1)
    idx = idx[inside]
    val = smooth[tuple(idx.T)]
    order = np.argsort(-val, kind="stable")
    idx, val = idx[order], val[order]
    keep, alive = [], np.ones(len(idx), dtype=bool)
    tree = cKDTree(idx * spacing) if len(idx) else None
    for j in range(len(idx)):
        if alive[j]:
            keep.append(j)
            alive[tree.query_ball_point(idx[j] * spacing, radius - 1e-6)] = False
    idx, val = idx[keep], val[keep]
    points = idx.astype(float)
    for j, g in enumerate(idx):  # parabola through the peak and its neighbours
        for ax in range(3):
            if 0 < g[ax] < smooth.shape[ax] - 1:
                lo, hi = g.copy(), g.copy()
                lo[ax] -= 1
                hi[ax] += 1
                a, b, c = smooth[tuple(lo)], smooth[tuple(g)], smooth[tuple(hi)]
                den = a - 2 * b + c
                if den < 0:
                    points[j, ax] += np.clip(0.5 * (a - c) / den, -0.5, 0.5)
    return points, val.astype(float)


def pick_classes(probabilities, radii, labels=None, **kwargs):
    """Picks of several particle classes from a probability volume.

    Parameters
    ----------
    probabilities : array-like
        (n_classes, z, y, x), as `SemanticSegmentationManager.predict_proba`.
    radii : dict
        {stored label: radius in voxels} of the particle classes to pick.
    labels : array-like, optional
        Stored label of each probability volume: ``model.classes_ + 1`` for
        `SemanticSegmentationManager.predict_proba`. Defaults to 1, 2, ...
    **kwargs
        Passed to `pick`.

    Returns
    -------
    picks : np.ndarray
        (n, 5) rows of z, y, x, score, stored label; ranked within each label.
    """
    if labels is None:
        labels = np.arange(1, len(probabilities) + 1)
    labels = list(np.asarray(labels).tolist())
    if len(labels) != len(probabilities):
        raise ValueError("labels must give one stored label per probability volume")
    out = []
    for label, radius in radii.items():
        if label not in labels:
            raise ValueError(f"No probability volume for label {label}")
        points, scores = pick(probabilities[labels.index(label)], radius, **kwargs)
        out.append(np.column_stack([points, scores, np.full(len(scores), label)]))
    return np.concatenate(out) if out else np.zeros((0, 5))
