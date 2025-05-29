"""Module for augmenting EEG data and preparing it for model training.

This script handles loading of sleep EEG data, performs leave-one-subject-out
cross-validation, applies data augmentation techniques (like Gaussian noise,
sign flipping, time reversal) to the training folds, extracts features,
and saves the resulting data splits.
"""

import mne
import numpy as np
from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Data Augmentation Functions
def gaussian_noise(
    epochs: NDArray[np.float64], noise_level: float = 0.16
) -> NDArray[np.float64]:
    """Adds Gaussian noise to the input EEG data.

    :param epochs: 2D numpy array of shape (n_channels, n_times)
                   representing EEG data.
    :param noise_level: Standard deviation of the Gaussian noise to be added.
    :return: 2D numpy array as `epochs` with added Gaussian noise.
    """
    if epochs.ndim != 2:
        raise ValueError(
            "epoch_data must have shape (n_channels, "
            + f"n_times), but got shape {epochs.shape}"
        )
    noise = np.random.normal(0, noise_level, epochs.shape)
    noisy_eeg = epochs + noise
    return noisy_eeg


def sign_flip(epochs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flips the sign of the input EEG data.

    :param epochs: 2D numpy array of shape (n_channels, n_times)
                   representing EEG data.
    :return: 2D numpy array as `epochs` with flipped signs.
    """
    if epochs.ndim != 2:
        raise ValueError(
            "epoch_data must have shape (n_channels, n_times), "
            + f"but got shape {epochs.shape}"
        )
    return epochs * -1


def time_reverse(epochs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reverses the time axis of the input EEG data.

    :param epochs: 2D numpy array of shape (n_channels, n_times)
                   representing EEG data.
    :return: 2D numpy array as `epochs` with the time axis reversed.
    """
    if epochs.ndim != 2:
        raise ValueError(
            "epoch_data must have shape (n_channels, n_times), "
            + f"but got shape {epochs.shape}"
        )
    reversed = np.flip(epochs, axis=1).copy()
    return reversed


def create_epochs_from_numpy(
    data_array: NDArray[np.float64],
    labels_array: NDArray[np.float64],
    info: mne.Info
) -> mne.EpochsArray | None:
    """Helper function to create MNE EpochsArray from numpy arrays.

    :param data_array: 2D numpy array of shape (n_epochs, n_channels, n_times)
    :param labels_array: 1D numpy array of shape (n_epochs,)
    :param info: MNE Info object containing channel information
    :return: MNE EpochsArray object or None if data_array is empty
    """
    if data_array.shape[0] == 0:
        logger.debug("Attempted to create MNE EpochsArray from empty data.")
        return None
    if data_array.ndim != 3 or data_array.shape[1] != len(info.ch_names):
        logger.error(
            f"Data array has incorrect shape {data_array.shape} "
            + f"for {len(info.ch_names)} channels or ndim != 3. "
            + "Cannot create EpochsArray."
        )
        return None

    events = np.column_stack(
        (
            np.arange(len(labels_array)),
            np.zeros(len(labels_array), dtype=int),
            labels_array.astype(int),
        )
    )
    try:
        epochs = mne.EpochsArray(
            data_array, info=info, events=events,
            tmin=0.0, baseline=None,
            verbose=False
        )
        return epochs
    except Exception as e:
        logger.error(f"Failed to create EpochsArray: {e}")
        return None
