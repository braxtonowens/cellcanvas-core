import numpy as np
from sklearn.base import BaseEstimator

from cellcanvas_core.semantic.picking import pick, pick_classes, predict_proba

SHAPE = (24, 48, 48)
R = 4.0


def blobs(centres, heights, sigma=2.0):
    """A probability volume with a Gaussian blob of each height at each centre."""
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij"), -1)
    prob = np.zeros(SHAPE, dtype=np.float32)
    for c, h in zip(centres, heights):
        prob += h * np.exp(-((grid - c) ** 2).sum(-1) / (2 * sigma**2))
    return np.clip(prob, 0, 1)


def test_one_pick_per_blob_ranked_by_score():
    centres = [(12, 12, 12), (12, 30, 30), (6, 36, 12)]
    points, scores = pick(blobs(centres, [0.9, 0.6, 0.8]), R)
    assert points.shape == (3, 3) and scores.shape == (3,)
    assert np.all(np.diff(scores) <= 0)  # best first
    expected = np.array(centres)[[0, 2, 1]]  # by height .9, .8, .6
    assert np.abs(points - expected).max() < 1.0


def test_close_blobs_merge_and_edge_picks_are_dropped():
    # two blobs 3 voxels apart (< R) and one 1 voxel from a face
    prob = blobs([(12, 20, 20), (12, 20, 23), (12, 1, 40)], [0.9, 0.9, 0.9])
    points, _ = pick(prob, R)
    assert len(points) == 1
    assert np.abs(points[0] - (12, 20, 21.5)).max() < 1.5
    points, _ = pick(blobs([(12, 1, 40)], [0.9]), R, edge=0.0)
    assert len(points) == 1  # kept without the edge margin


def test_threshold_and_empty_volume():
    prob = blobs([(12, 24, 24)], [0.3])
    assert len(pick(prob, R)[0]) == 1
    assert len(pick(prob, R, threshold=0.5)[0]) == 0
    points, scores = pick(np.zeros(SHAPE, dtype=np.float32), R)
    assert points.shape == (0, 3) and scores.shape == (0,)


def test_anisotropic_spacing_scales_the_suppression():
    # 5 slices apart: two picks at unit spacing; one when a slice is half a voxel
    prob = blobs([(9, 24, 24), (14, 24, 24)], [0.9, 0.9], sigma=1.0)
    assert len(pick(prob, 3.0)[0]) == 2
    assert len(pick(prob, 3.0, spacing=(0.5, 1.0, 1.0))[0]) == 1


class ProbaModel(BaseEstimator):
    """Two classes; class 1 where the first feature is positive."""

    classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X[:, 0]))
        return np.stack([1 - p, p], 1)


def test_predict_proba_and_pick_classes():
    features = np.zeros((2, *SHAPE), dtype=np.float32)
    features[0] = 8 * blobs([(12, 12, 12), (12, 36, 36)], [1.0, 1.0]) - 4

    class Manager:
        model = ProbaModel()

    proba = predict_proba(Manager(), features, batch_size=1000)
    assert proba.shape == (2, *SHAPE)
    np.testing.assert_allclose(proba.sum(0), 1, rtol=1e-5)
    picks = pick_classes(proba, {1: R})
    assert picks.shape == (2, 5) and np.all(picks[:, 4] == 1)
    assert np.abs(np.sort(picks[:, 1]) - [12, 36]).max() < 1.0
    assert pick_classes(proba, {}).shape == (0, 5)
