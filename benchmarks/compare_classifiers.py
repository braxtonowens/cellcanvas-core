"""Compare pixel classifiers across volumes.

Paints random voxels of every class in k evenly spaced z slices of a
training volume's ground truth, fits each classifier through
SemanticSegmentationManager and scores the prediction against the ground
truth of a test volume (another tomogram), or, without one, against the
unpainted voxels of the training volume. Reports Dice per class, their mean,
and fit and predict time.

    python benchmarks/compare_classifiers.py TRAIN_FEATURES.zarr \
        TRAIN_TRUTH.zarr [TEST_FEATURES.zarr TEST_TRUTH.zarr] \
        [--slices 1 4 16] [--voxels 200] [--seeds 0 1 2] [--background 1]

FEATURES.zarr is a (c, z, y, x) array; TRUTH.zarr a (z, y, x) integer array
with the stored label convention (1, 2, ... are classes; 0 is not scored).
Use --voxels large (e.g. 20000) to paint whole slices.
"""

import argparse
import time

import numpy as np
import zarr
from sklearn.ensemble import RandomForestClassifier

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from cellcanvas_core.semantic.fewshot import FewShotClassifier
from cellcanvas_core.semantic.segmentation_manager import SemanticSegmentationManager

MODELS = {
    "random forest": lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1),
    "few-shot": lambda seed: FewShotClassifier(random_state=seed),
}


def paint(truth, n_slices, n_voxels, rng):
    """Up to n_voxels random voxels of each class in n_slices evenly spaced slices."""
    labels = np.zeros_like(truth)
    step = truth.shape[0] / n_slices
    for z in (rng.uniform(0, step) + step * np.arange(n_slices)).astype(int):
        for c in np.unique(truth[z]):
            if c == 0:
                continue
            y, x = np.nonzero(truth[z] == c)
            pick = rng.choice(len(y), min(n_voxels, len(y)), replace=False)
            labels[z, y[pick], x[pick]] = c
    return labels


def dice(prediction, truth, scored, classes):
    """Dice of each class over the scored voxels."""
    out = {}
    for c in classes:
        p, t = (prediction == c) & scored, (truth == c) & scored
        denominator = p.sum() + t.sum()
        out[c] = 2 * (p & t).sum() / denominator if denominator else np.nan
    return out


def dataset(features, labels):
    """A DataSet holding features and labels, with an empty image."""
    shape = labels.shape
    return DataSet(
        np.zeros((1, *shape), dtype=np.uint8),
        {"features": features},
        labels,
        np.zeros(shape, dtype=labels.dtype),
    )


def main():
    """Fit, predict and score every model."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="train, then test, FEATURES TRUTH")
    parser.add_argument("--slices", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--voxels", type=int, default=200, help="per class and slice")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--background", type=int, help="label left out of the mean Dice"
    )
    args = parser.parse_args()
    if len(args.paths) not in (2, 4):
        parser.error("give TRAIN_FEATURES TRAIN_TRUTH [TEST_FEATURES TEST_TRUTH]")

    train_features = zarr.open_array(args.paths[0], mode="r")
    train_truth = np.asarray(zarr.open_array(args.paths[1], mode="r"))
    if len(args.paths) == 4:
        test_features = zarr.open_array(args.paths[2], mode="r")
        test_truth = np.asarray(zarr.open_array(args.paths[3], mode="r"))
        test = dataset(test_features, np.zeros_like(test_truth))
    else:
        test, test_truth = None, train_truth
    classes = [c for c in np.unique(test_truth) if c > 0]
    shown = [c for c in classes if c != args.background]

    print(
        "| slices | model | mean Dice | "
        + " ".join(f"Dice {c}" for c in shown)
        + " | fit s | predict s |"
    )
    print("|---|---|---|---|---|---|")
    for n_slices in args.slices:
        for name, make in MODELS.items():
            scores, fit_s, predict_s = [], [], []
            for seed in args.seeds:
                labels = paint(
                    train_truth, n_slices, args.voxels, np.random.default_rng(seed)
                )
                train = dataset(train_features, labels)
                manager = SemanticSegmentationManager(DataManager([train]), make(seed))
                manager.logger.setLevel("WARNING")
                start = time.perf_counter()
                manager.fit()
                fit_s.append(time.perf_counter() - start)
                start = time.perf_counter()
                features = (train if test is None else test).concatenated_features
                prediction = manager.predict(features) + 1
                predict_s.append(time.perf_counter() - start)
                scored = test_truth > 0
                if test is None:
                    scored &= labels == 0
                scores.append(dice(prediction, test_truth, scored, classes))
            per_class = [np.nanmean([s[c] for s in scores]) for c in shown]
            print(
                f"| {n_slices} | {name} "
                f"| {np.mean(per_class):.3f} "
                f"| {' '.join(f'{d:.2f}' for d in per_class)} "
                f"| {np.mean(fit_s):.1f} | {np.mean(predict_s):.1f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
