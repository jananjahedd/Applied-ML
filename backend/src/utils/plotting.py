"""Plotting functions."""

from typing import Any, Optional

import mne


def plot_signals_mne(
    recording: Any,  # it is a recording but to avoid circular imports use Any
    annotations: bool = True,
    raw: Optional[mne.io.BaseRaw] = None,
    channels: Optional[list[str]] = None,
) -> None:
    """Plot the signals of the specified channels using MNE-Python.

    Args:
        recording (Recording): The recording object containing the file path.
        annotations (bool): Whether to plot the annotations.
        raw (mne.io.BaseRaw): The raw object to plot. If None, it is read from
        the file.
        channels (list): List of channel names to plot.
    """
    if raw is None:
        raw = mne.io.read_raw_edf(recording.file_path, preload=True)
    if channels is None:
        channels = raw.ch_names
    if annotations:
        anno = mne.read_annotations(recording.anno_path)
        raw.set_annotations(anno)

    raw.plot(
        picks=channels,
        block=True,
        title=f"PSG Data - Paient {recording.patient_number}, Night {recording.night}",
        show_options=True,
    )
