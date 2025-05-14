"""Module for feature engineering.

Extracts the features from the preprocessed data.
"""

from typing import List, Tuple

import mne
import numpy as np
from numpy.typing import NDArray

from src.utils.logger import get_logger

# setup logging
logger = get_logger(__name__)

# define standard frequency bands
FREQ_BANDS = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30],
}

# overall PSD computation range
FMIN = 0.5
FMAX = 30.0

# type hint helper
ExtractedFeatures = Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]


class FeatureEngineering:
    """Extracts features from MNE Epochs or EpochsArray objects.

    This class processes EEG, EOG, and EMG data within the epochs to compute
    features relevant for sleep stage classification.
    It calculates relative power spectral densities for EEG, variance for EOG,
    and mean of absolute values for EMG.
    """

    def __init__(self, epochs: mne.Epochs) -> None:
        """Initialize the feature engineering class.

        :param epochs: the preprocessed epochs object.
        """
        logger.info(
            "Starting feature engineering process for " + f"{len(epochs)} epochs/"
        )
        self.epochs = epochs
        self.ch_types = self.epochs.info.get_channel_types(unique=True)

    def feature_extraction(self) -> ExtractedFeatures:
        """Extract features from the epochs.

        :return: Tuple of features, labels, and feature names.
        """
        logger.info("Extracting features from epochs...")

        all_features = []
        feature_names = []
        n_fft = int(self.epochs.info["sfreq"] * 2)

        if "eeg" in self.ch_types:
            eeg_picks = mne.pick_types(self.epochs.info, eeg=True, exclude="bads")
            if len(eeg_picks) > 0:
                eeg_ch_names = [self.epochs.ch_names[i] for i in eeg_picks]
                logger.info(
                    "Extracting EEG features from " + f"channels: {eeg_ch_names}"
                )

                # computen PSD for all channels
                spectrum = self.epochs.compute_psd(
                    method="welch",
                    picks=eeg_picks,
                    fmin=FMIN,
                    fmax=FMAX,
                    n_fft=n_fft,
                    n_overlap=n_fft // 2,
                    average="mean",
                    window="hann",
                    verbose=False,
                )
                psds, freqs = spectrum.get_data(return_freqs=True)

                total_power = np.sum(psds, axis=2, keepdims=True)
                total_power[total_power == 0] = 1e-10

                for band_name, (fmin, fmax) in FREQ_BANDS.items():
                    band_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
                    if len(band_indices) > 0:
                        abs_band_power = np.sum(psds[:, :, band_indices], axis=2)
                        rel_band_power = abs_band_power / total_power[:, :, 0]

                        for i, ch_name in enumerate(eeg_ch_names):
                            all_features.append(rel_band_power[:, i])
                            feature_names.append(f"{ch_name}_{band_name}_RelP")
                    else:
                        logger.warning(
                            "No frequencies found for band "
                            + f"{band_name} in range ({fmin}-{fmax}"
                            + "Hz) within computed PSD frequencies "
                            + f"({freqs.min()}-{freqs.max()} Hz). "
                            + "Skipping this band for EEG."
                        )
            else:
                logger.warning(
                    "No EEG channels found or picked. Skipping "
                    + "EEG feature extraction."
                )

        if "eog" in self.ch_types:
            eog_picks = mne.pick_types(self.epochs.info, eog=True, exclude="bads")
            if len(eog_picks) > 0:
                eog_ch_names = [self.epochs.ch_names[i] for i in eog_picks]
                logger.info(f"Extracting EOG features from channels: {eog_ch_names}")
                eog_data = self.epochs.get_data(picks=eog_picks)

                for i, ch_name in enumerate(eog_ch_names):
                    eog_variance = np.var(eog_data[:, i, :], axis=1)
                    all_features.append(eog_variance)
                    feature_names.append(f"{ch_name}_Var")
            else:
                logger.warning(
                    "No WOG channels found or picked. Skipping "
                    + "EOG feature extraction."
                )

        if "emg" in self.ch_types:
            emg_picks = mne.pick_types(self.epochs.info, emg=True, exclude="bads")
            if len(emg_picks) > 0:
                emg_ch_names = [self.epochs.ch_names[i] for i in emg_picks]
                logger.info(f"Extracting EMG features from channels: {emg_ch_names}")
                emg_data = self.epochs.get_data(picks=emg_picks)

                for i, ch_name in enumerate(emg_ch_names):
                    emg_mean = np.mean(np.abs(emg_data[:, i, :]), axis=1)
                    all_features.append(emg_mean)
                    feature_names.append(f"{ch_name}_Mean")
            else:
                logger.warning(
                    "No EMG channels found or picked. Skipping "
                    + "EMG feature extraction."
                )

        if not all_features:
            logger.error(
                "No features were extracted. Pleach check " + "channel types and data."
            )
            return (
                np.array([]).reshape(len(self.epochs), 0),
                self.epochs.events[:, -1],
                [],
            )

        # combine all features
        X = np.column_stack(all_features)

        # extract labels
        y = self.epochs.events[:, -1]
        logger.info(
            f"Successfully extracted {X.shape[1]} "
            + f"features for {X.shape[0]} epochs."
        )
        logger.info(f"Feature names: {feature_names}")

        return X, y, feature_names


"""
More features ideas to consider:
for EEG:
(frequency domain):
    - we can also extract the band ratios
    - spectral edge frequency (the frequency below which a certain percentage
      (e.g., 50%, 75%, 90%, 95%) of the total power resides)
    - peak frequency (frequency with highest power per band)
    - spectral entropy (for measuring uniformity)

(time domain):
    - statistical measures (variance, skewness, kurtosis, standard deviation)
    - Hjorth parameters (activity, mobility, complexity)
    - zero-crossing rate (number of times the signal crosses zero)
    - mean peak-to-peak amplitude (average difference between consecutive
                                   positive and negative peaks)

non-linear features:
    - fractal dimension (Higuchi Fractal Dimension)
    - entropy measures (sample entropy, approximate entropy)
    - detrended fluctuation analysis (DFA): measures long-range correlations

event-based features (computationally expensive):
    - sleep spindle features
    - K-complex features for N2 target label
    - slow wave features for N3 target label (Percentage of epoch occupied by
                                              delta waves)

for EOG:
    - frequency domain for power in low-frequency bands (for eye movements)

for EMG:
    - amplitude/activity Level for mean of the rectified EMG signal,
      root mean square (RMS) value, and variance or standard deviation
      for distingushing REM from Wake
    - frequency domain features (e.g., median frequency, mean frequency)

IMPORTANT: after adding the features, we need to check which ones are important
for model classification to avoid curse of dimensionality and overfitting.
"""
