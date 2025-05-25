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

    def _calculate_sef(
        self,
        psds: NDArray[np.float64],
        freqs: NDArray[np.float64],
        percentage: float = 0.95,
    ) -> NDArray[np.float64]:
        """Calculate Spectral Edge Frequency (SEF).

        :param psds: power spectral densities.
        :param freqs: frequencies corresponding to PSDs.
        :param percentage: the percentage of total power.
        :return: SEF values for each epoch and channel (n_epochs, n_channels).
        """
        total_power = np.sum(psds, axis=2)
        total_power[total_power == 0] = 1e-10

        cumulative_power = np.cumsum(psds, axis=2)
        sef_indices = np.zeros((psds.shape[0], psds.shape[1]), dtype=int)

        for i in range(psds.shape[0]):
            for j in range(psds.shape[1]):
                power_threshold = total_power[i, j] * percentage
                found_indices = np.where(cumulative_power[i, j, :] >= power_threshold)[
                    0
                ]
                if len(found_indices) > 0:
                    sef_indices[i, j] = found_indices[0]
                else:
                    sef_indices[i, j] = len(freqs) - 1
        return freqs[sef_indices]

    def _calculate_hjorth(
        self, data: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate Hjorth Mobility and Complexity.

        data shape: (n_epochs, n_channels, n_times)
        output shape for mobility/complexity: (n_epochs, n_channels)
        """
        if data.ndim == 2:
            data = data[np.newaxis, np.newaxis, :]
        elif data.ndim == 1:
            data = data[np.newaxis, np.newaxis, :]

        var_x = np.var(data, axis=2)
        var_x[var_x == 0] = 1e-10

        diff_x = np.diff(data, n=1, axis=2)
        var_diff_x = np.var(diff_x, axis=2)
        var_diff_x[var_diff_x == 0] = 1e-10

        mobility = np.sqrt(var_diff_x / var_x)

        diff_dx = np.diff(diff_x, n=1, axis=2)
        var_diff_dx = np.var(diff_dx, axis=2)

        mobility_of_diff_x = np.sqrt(var_diff_dx / var_diff_x)

        complexity = mobility_of_diff_x / mobility
        complexity[mobility == 0] = 0

        return mobility, complexity

    def feature_extraction(self) -> ExtractedFeatures:
        """Extract features from the epochs.

        :return: Tuple of features, labels, and feature names.
        """
        logger.info("Extracting features from epochs...")

        all_features_list: List[NDArray[np.float64]] = []
        feature_names: List[str] = []

        # common PSD parameters
        n_fft = int(self.epochs.info["sfreq"] * 2)
        n_overlap = n_fft // 2

        if "eeg" in self.ch_types:
            eeg_picks = mne.pick_types(self.epochs.info, eeg=True, exclude="bads")
            if len(eeg_picks) > 0:
                eeg_ch_names = [self.epochs.ch_names[i] for i in eeg_picks]
                logger.info(
                    "Extracting EEG features from " + f"channels: {eeg_ch_names}"
                )

                # EEG: PSD-based features (Relative Band Powers, SEF95)
                spectrum_eeg = self.epochs.compute_psd(
                    method="welch",
                    picks=eeg_picks,
                    fmin=FMIN,
                    fmax=FMAX,
                    n_fft=n_fft,
                    n_overlap=n_overlap,
                    average="mean",
                    window="hann",
                    verbose=False,
                )
                psds_eeg, freqs_eeg = spectrum_eeg.get_data(return_freqs=True)

                total_power_eeg = np.sum(psds_eeg, axis=2, keepdims=True)
                total_power_eeg[total_power_eeg == 0] = 1e-10

                for band_name, (fmin_band, fmax_band) in FREQ_BANDS.items():
                    band_indices = np.where(
                        (freqs_eeg >= fmin_band) & (freqs_eeg < fmax_band)
                    )[0]
                    if len(band_indices) > 0:
                        abs_band_power = np.sum(psds_eeg[:, :, band_indices], axis=2)
                        rel_band_power = abs_band_power / total_power_eeg[:, :, 0]

                        for i, ch_name in enumerate(eeg_ch_names):
                            all_features_list.append(rel_band_power[:, i])
                            feature_names.append(f"{ch_name}_{band_name}_RelP")
                    else:
                        logger.warning(
                            f"No frequencies found for EEG band {band_name} in"
                            + f" range ({fmin_band}-{fmax_band} Hz). Skipping."
                        )

                # EEG: Spectral Edge Frequency (SEF95)
                sef95_eeg = self._calculate_sef(psds_eeg, freqs_eeg, percentage=0.95)
                for i, ch_name in enumerate(eeg_ch_names):
                    all_features_list.append(sef95_eeg[:, i])
                    feature_names.append(f"{ch_name}_SEF95")

                # EEG: Time-domain features (Hjorth Parameters)
                eeg_data = self.epochs.get_data(picks=eeg_picks)
                mobility_eeg, complexity_eeg = self._calculate_hjorth(eeg_data)

                for i, ch_name in enumerate(eeg_ch_names):
                    all_features_list.append(mobility_eeg[:, i])
                    feature_names.append(f"{ch_name}_HjorthMobility")
                    all_features_list.append(complexity_eeg[:, i])
                    feature_names.append(f"{ch_name}_HjorthComplexity")
            else:
                logger.warning(
                    "No EEG channels found or picked. Skipping EEG"
                    + " feature extraction."
                )

        if "eog" in self.ch_types:
            eog_picks = mne.pick_types(self.epochs.info, eog=True, exclude="bads")
            if len(eog_picks) > 0:
                eog_ch_names = [self.epochs.ch_names[i] for i in eog_picks]
                logger.info(f"Extracting EOG features from channels: {eog_ch_names}")
                eog_data_time = self.epochs.get_data(picks=eog_picks)

                # EOG: Variance
                for i, ch_name in enumerate(eog_ch_names):
                    eog_variance = np.var(eog_data_time[:, i, :], axis=1)
                    all_features_list.append(eog_variance)
                    feature_names.append(f"{ch_name}_Var")

                # EOG: Relative Delta Power
                spectrum_eog = self.epochs.compute_psd(
                    method="welch",
                    picks=eog_picks,
                    fmin=FMIN,
                    fmax=FMAX,
                    n_fft=n_fft,
                    n_overlap=n_overlap,
                    average="mean",
                    window="hann",
                    verbose=False,
                )
                psds_eog, freqs_eog = spectrum_eog.get_data(
                    picks=eog_ch_names, return_freqs=True
                )
                total_power_eog = np.sum(psds_eog, axis=2, keepdims=True)
                total_power_eog[total_power_eog == 0] = 1e-10

                delta_fmin, delta_fmax = FREQ_BANDS["delta"]
                delta_indices_eog = np.where(
                    (freqs_eog >= delta_fmin) & (freqs_eog < delta_fmax)
                )[0]

                if len(delta_indices_eog) > 0:
                    abs_delta_power_eog = np.sum(
                        psds_eog[:, :, delta_indices_eog], axis=2
                    )
                    rel_delta_power_eog = abs_delta_power_eog / total_power_eog[:, :, 0]
                    for i, ch_name in enumerate(eog_ch_names):
                        all_features_list.append(rel_delta_power_eog[:, i])
                        feature_names.append(f"{ch_name}_Delta_RelP")
                else:
                    logger.warning(
                        "No frequencies found for EOG Delta band."
                        + " Skipping EOG Relative Delta Power."
                    )
            else:
                logger.warning(
                    "No EOG channels found or picked. Skipping EOG"
                    + " feature extraction."
                )

        if "emg" in self.ch_types:
            emg_picks = mne.pick_types(self.epochs.info, emg=True, exclude="bads")
            if len(emg_picks) > 0:
                emg_ch_names = [self.epochs.ch_names[i] for i in emg_picks]
                logger.info(f"Extracting EMG features from channels: {emg_ch_names}")
                emg_data_time = self.epochs.get_data(picks=emg_picks)

                # EMG: Mean of absolute values
                for i, ch_name in enumerate(emg_ch_names):
                    emg_mean_abs = np.mean(np.abs(emg_data_time[:, i, :]), axis=1)
                    all_features_list.append(emg_mean_abs)
                    feature_names.append(f"{ch_name}_MeanAbs")
            else:
                logger.warning(
                    "No EMG channels found or picked. "
                    + "Skipping EMG feature extraction."
                )

        if not all_features_list:
            logger.error(
                "No features were extracted. Please check channel" + "types and data."
            )
            return (
                np.array([]).reshape(len(self.epochs), 0),
                self.epochs.events[:, -1],
                [],
            )

        X = np.column_stack(all_features_list)

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
