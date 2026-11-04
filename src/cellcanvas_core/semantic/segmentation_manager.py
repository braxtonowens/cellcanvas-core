from typing import Protocol
from itertools import product

import sys
import logging
import numpy as np
import dask.array as da
from dask import delayed
from sklearn.exceptions import NotFittedError

from cellcanvas_core.data.data_manager import DataManager
from cellcanvas_core.data.data_set import DataSet
from tqdm import tqdm

class SegmentationModel(Protocol):
    """Protocol for semantic segmentations models that are
    compatible with the  SemanticSegmentationManager.
    """

    def fit(self, X, y): ...
    def predict(self, X): ...


class SemanticSegmentationManager:
    def __init__(self, data: DataManager, model: SegmentationModel):
        self.data = data
        self.model = model

        self._init_logging()

    def _init_logging(self):
        self.logger = logging.getLogger("cellcanvas")
        self.logger.setLevel(logging.DEBUG)
        streamHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)

    def update_data_manager(self, data: DataManager):
        self.data = data

    def fit(self):
        """Fit using the model using the data in the data manager."""
        self.logger.info("Starting to fit")
        # Get training data from the data manager
        features, labels = self.data.get_training_data()

        features_computed, labels_computed = features.compute(), labels.compute()

        self.logger.info("Starting the actual model fit")

        self.model.fit(features_computed, labels_computed)

    def predict(self, feature_image, batch_size=65536):
        """Predict using the trained model.

        Parameters
        ----------
        feature_image : array-like
            (c, z, y, x) NumPy, Dask or Zarr array.
        batch_size : int
            Maximum number of feature rows passed to the model per call.
            Source chunks and the complete output labels also occupy memory.

        Returns
        -------
        predicted_labels : Array
            Zero-based model classes. Add one before storing as painted labels,
            where zero means unannotated and positive values are classes.
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        feature_image = da.asarray(feature_image)
        if feature_image.ndim != 4 or any(size == 0 for size in feature_image.shape):
            raise ValueError("Features must have nonempty (c, z, y, x) axes")
        c, z, y, x = feature_image.shape
        features = feature_image.transpose(1, 2, 3, 0).reshape(-1, c)

        try:
            predicted_labels = None
            for start in range(0, features.shape[0], batch_size):
                stop = min(start + batch_size, features.shape[0])
                batch = np.asarray(self.model.predict(features[start:stop].compute()))
                if batch.shape != (stop - start,):
                    raise ValueError("Model must return one class per feature row")
                if predicted_labels is None:
                    predicted_labels = np.empty(features.shape[0], dtype=batch.dtype)
                predicted_labels[start:stop] = batch
        except NotFittedError:
            raise NotFittedError(
                "You must train the classifier `clf` first"
                "for example with the `fit_segmenter` function."
            ) from None

        return predicted_labels.reshape(feature_image.shape[1:])

    def predict_to_segmentation(self, dataset: DataSet, batch_size=65536):
        """Write positive class labels to the dataset in spatial blocks.

        Model calls contain at most batch_size voxels. Memory includes the
        feature block, its source chunks and one block of predictions; no
        complete prediction volume is allocated. Model classes start at zero
        and are shifted by one for storage. Existing painting labels are left
        intact. A failure can leave already written blocks in place.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        target = dataset.segmentation
        dtype = np.dtype(target.dtype)
        if dtype.kind not in "iu":
            raise ValueError("Segmentation storage must use an integer dtype")
        features = dataset.concatenated_features
        if features.shape[1:] != target.shape:
            raise ValueError("Feature spatial axes must match segmentation storage")
        z, y, x = target.shape
        block_x = min(x, batch_size)
        block_y = min(y, max(1, batch_size // block_x))
        block_z = min(z, max(1, batch_size // (block_x * block_y)))
        for start_z, start_y, start_x in product(
            range(0, z, block_z), range(0, y, block_y), range(0, x, block_x)
        ):
            region = (slice(start_z, min(z, start_z + block_z)),
                      slice(start_y, min(y, start_y + block_y)),
                      slice(start_x, min(x, start_x + block_x)))
            block = features[(slice(None),) + region].compute()
            rows = np.moveaxis(block, 0, -1).reshape(-1, block.shape[0])
            classes = np.asarray(self.model.predict(rows))
            if classes.shape != (len(rows),):
                raise ValueError("Model must return one class per feature row")
            if classes.dtype.kind not in "iu" or np.any(classes < 0):
                raise ValueError("Model classes must be nonnegative integers")
            if np.any(classes > np.iinfo(dtype).max - 1):
                raise ValueError("Model classes do not fit in segmentation storage after adding one")
            stored = classes.astype(dtype) + 1
            target[region] = stored.reshape(block.shape[1:])
