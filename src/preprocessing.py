"""Preprocessing functions for EEG data using MNE-Python."""

import mne

from src.utils.logger import get_logger

logger = get_logger(__name__)


def bandpass_filter(
    raw: mne.io.BaseRaw, low_freq: float, high_freq: float, signal: str
) -> mne.io.BaseRaw:
    """Apply a bandpass filter to the raw data for a specific signal type.

    Args:
        raw (mne.io.BaseRaw): The raw object to filter.
        low_freq (float): The low frequency cutoff.
        high_freq (float): The high frequency cutoff.
        signal (str): The type of signal to filter ('EEG', 'EOG', 'EMG').

    Returns:
        mne.io.BaseRaw: The filtered raw object.
    """
    sfreq = raw.info["sfreq"]
    nyquist_freq = sfreq / 2.0
    if high_freq >= nyquist_freq:
        raise ValueError(
            f"Highpass frequency [{high_freq}] must be less than "
            f"Nyquist frequency ({nyquist_freq})"
        )

    picks = []
    if signal == "EEG":
        picks.extend(mne.pick_types(raw.info, eeg=True, exclude="bads"))
        logger.info(
            f"Bandpass filter applied to EEG channels: {low_freq}-{high_freq} Hz"
        )
    if signal == "EOG":
        picks.extend(
            mne.pick_channels(raw.info["ch_names"], include=["EOG horizontal"])
        )
        logger.info(
            f"Bandpass filter applied to EOG channels: {low_freq}-{high_freq} Hz"
        )
    if signal == "EMG":
        picks.extend(mne.pick_channels(raw.info["ch_names"], include=["EMG submental"]))
        logger.info(
            f"Bandpass filter applied to EMG channels: {low_freq}-{high_freq} Hz"
        )

    if picks:
        raw.filter(
            low_freq,
            high_freq,
            picks=picks,
            method="fir",
            fir_design="firwin",
            verbose=False,
        )
    else:
        logger.info("No signal types selected for bandpass filtering.")
    return raw


def notch_filter(raw: mne.io.BaseRaw, freq: float, signal: str) -> mne.io.BaseRaw:
    """Apply a notch filter to the raw data for a specific signal type.

    Args:
        raw (mne.io.BaseRaw): The raw object to filter.
        freq (float): The notch frequency.
        signal (str): The type of signal to filter ('EEG', 'EOG', 'EMG').

    Returns:
        mne.io.BaseRaw: The filtered raw object.
    """
    sfreq = raw.info["sfreq"]
    nyquist = sfreq / 2.0
    if freq >= nyquist:
        raise ValueError(
            f"Notch frequency {freq} is greater than Nyquist frequency {nyquist}."
        )

    picks = []
    if signal == "EEG":
        picks.extend(mne.pick_types(raw.info, eeg=True, exclude="bads"))
        logger.info(f"Notch filter applied to EEG channels at: {freq} Hz")
    if signal == "EOG":
        picks.extend(
            mne.pick_channels(raw.info["ch_names"], include=["EOG horizontal"])
        )
        logger.info(f"Notch filter applied to EOG channels at: {freq} Hz")
    if signal == "EMG":
        picks.extend(mne.pick_channels(raw.info["ch_names"], include=["EMG submental"]))
        logger.info(f"Notch filter applied to EMG channels at: {freq} Hz")

    if picks:
        raw.notch_filter(
            freqs=freq, picks=picks, method="fir", fir_design="firwin", verbose=False
        )
    else:
        logger.info("No signal types selected for notch filtering.")
    return raw
