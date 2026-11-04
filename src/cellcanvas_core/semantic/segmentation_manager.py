from typing import Protocol

import sys
import logging
import numpy as np
import dask.array as da
from dask import delayed
from sklearn.exceptions import NotFittedError

from cellcanvas_core.data.data_manager import DataManager
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
