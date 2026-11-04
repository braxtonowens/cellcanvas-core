"""Ranked particle picks from class probabilities."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def predict_proba(manager, feature_image, batch_size=65536):
    """Class probabilities of a feature image with a fitted manager.

    Parameters
    ----------
    manager : SemanticSegmentationManager
        Fitted; its model must have ``predict_proba`` and ``classes_``.
    feature_image : array-like
        (c, z, y, x) features, as `SemanticSegmentationManager.predict`.
    batch_size : int
        Rows predicted at once.

    Returns
    -------
    probabilities : np.ndarray
        (n_classes, z, y, x) float32, classes in ``manager.model.classes_`` order.
    """
    c = feature_image.shape[0]
    shape = tuple(feature_image.shape[1:])
    rows = np.asarray(feature_image).reshape(c, -1).T
    out = np.empty((len(manager.model.classes_), rows.shape[0]), dtype=np.float32)
    for start in range(0, rows.shape[0], batch_size):
        out[:, start : start + batch_size] = manager.model.predict_proba(
            rows[start : start + batch_size]
        ).T
    return out.reshape((-1, *shape))


def pick(probability, radius, threshold=0.05, edge=0.5, spacing=(1.0, 1.0, 1.0)):
    """Ranked picks of one particle class.

    The probability is smoothed with a Gaussian of ``radius / 2``; local maxima
    whose raw probability exceeds ``threshold`` are candidates, kept in score
    order with greedy non-maximum suppression at ``radius``, refined to
    sub-voxel positions, and dropped within ``edge * radius`` of a face.

    Parameters
    ----------
    probability : array-like
        (z, y, x) probability of the class.
    radius : float
        Particle radius in voxels of the y/x axes.
    threshold : float
        Minimum raw probability of a candidate; the score threshold a user
        applies later selects the operating point.
    edge : float
        Picks closer than ``edge * radius`` to a face are dropped.
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
    smooth = ndi.gaussian_filter(prob, sigma=radius / 2 / spacing, mode="nearest")
    peaks = (prob > threshold) & (
        smooth >= ndi.maximum_filter(smooth, size=3, mode="nearest")
    )
    idx = np.argwhere(peaks)
    order = np.argsort(-smooth[peaks], kind="stable")
    idx, val = idx[order], smooth[peaks][order]
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
    margin = edge * radius / spacing
    far = (points >= margin).all(1) & (points <= np.array(prob.shape) - 1 - margin).all(
        1
    )
    return points[far], val[far].astype(float)


def pick_classes(probabilities, radii, **kwargs):
    """Picks of several classes from a probability volume.

    Parameters
    ----------
    probabilities : array-like
        (n_classes, z, y, x), as `predict_proba`.
    radii : dict
        {class index: radius in voxels} of the particle classes to pick.
    **kwargs
        Passed to `pick`.

    Returns
    -------
    picks : np.ndarray
        (n, 5) rows of z, y, x, score, class index; ranked within each class.
    """
    out = []
    for c, radius in radii.items():
        points, scores = pick(probabilities[c], radius, **kwargs)
        out.append(np.column_stack([points, scores, np.full(len(scores), c)]))
    return np.concatenate(out) if out else np.zeros((0, 5))
