"""Module for preprocessing of sleep EDF data.

This script includes functions for loading, filtering, epoching,
and artifact rejection of PSG (polysomnography) recordings.
It also provides utility functions for logging and plotting MNE-Python objects.
The goal is to prepare the data for subsequent feature extraction and
model training.
"""

import pathlib
import time
import traceback
from typing import Dict, Optional

import mne
import numpy as np
from autoreject import get_rejection_threshold  # type: ignore

from src.utils.paths import (get_data_dir, get_processed_data_dir,
                             get_repo_root)
from src.utils.logger import get_logger


PROJECT_ROOT = pathlib.Path(get_repo_root())
DATA_SUBFOLDER = "sleep-cassette"
DATA_DIR = pathlib.Path(get_data_dir()) / DATA_SUBFOLDER
PROCESSED_DATA_DIR = pathlib.Path(get_processed_data_dir()) / DATA_SUBFOLDER

N_SUBJECTS_TO_PROCESS = 10
EPOCH_DURATION = 30.0
TARGET_SFREQ = 100.0
NOTCH_FREQ = 50.0

EEG_BANDPASS = (0.3, 35.0)
EOG_BANDPASS = (0.3, 35.0)
EMG_BANDPASS = (10.0, 45.0)

AUTOREJECT_DECIM = 1
AUTOREJECT_RANDOM_STATE = 42
FIXED_EMG_THRESHOLD = 150e-6

ANNOTATION_MAP = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
    "Sleep stage ?": 0,
    "Movement time": 0,
}
EVENT_ID_MAP = {
    "Wake": 1,
    "N1": 2,
    "N2": 3,
    "N3/N4": 4,
    "REM": 5,
    "Unknown/Movement": 0,
}

# setup logger
logger = get_logger("preprocessing")


def bandpass_filter(
    raw: mne.io.BaseRaw, low_freq: float, high_freq: float, signal_type: str
) -> mne.io.BaseRaw:
    """Apply a bandpass FIR filter to specified channel types in raw data.

    :param raw: The MNE Raw object to filter.
    :param low_freq: The lower cutoff frequency of the filter.
    :param high_freq: The upper cutoff frequency of the filter.
    :param signal_type: The type of signal to filter ('eeg', 'eog', 'emg').
    :return: The filtered MNE Raw object.
    """
    sfreq = raw.info["sfreq"]
    nyquist_freq = sfreq / 2.0
    effective_high_freq = high_freq
    if high_freq >= nyquist_freq:
        effective_high_freq = nyquist_freq - 0.5
        if low_freq >= effective_high_freq:
            logger.error(
                f"Cannot filter {signal_type}: Low ({low_freq}) >= Adj. "
                f"High ({effective_high_freq}). Filter skipped."
            )
            return raw
    picks = None
    if signal_type == "eeg":
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
    elif signal_type == "eog":
        picks = mne.pick_types(raw.info, meg=False, eog=True, exclude="bads")
    elif signal_type == "emg":
        picks = mne.pick_types(raw.info, meg=False, emg=True, exclude="bads")
    else:
        logger.warning(
            f"Unknown signal type '{signal_type}' for bandpass filtering."
        )
        return raw
    if picks is not None and len(picks) > 0:
        try:
            raw.filter(
                l_freq=low_freq,
                h_freq=effective_high_freq,
                picks=picks,
                method="fir",
                phase="zero-double",
                fir_design="firwin",
                skip_by_annotation="edge",
                verbose=False,
            )
        except Exception as e:
            logger.error(
                f"Error applying bandpass filter for {signal_type}: {e}"
            )
    return raw


def notch_filter(
        raw: mne.io.BaseRaw,
        freq: float, signal_type: str) -> mne.io.BaseRaw:
    """Apply a notch FIR filter to specified channel types in raw data.

    :param raw: The MNE Raw object to filter.
    :param freq: The frequency to notch out.
    :param signal_type: The type of signal to filter ('eeg', 'eog', 'emg').
    :return: The filtered MNE Raw object.
    """
    sfreq = raw.info["sfreq"]
    nyquist = sfreq / 2.0
    if freq >= nyquist:
        return raw
    picks = None
    if signal_type == "eeg":
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
    elif signal_type == "eog":
        picks = mne.pick_types(raw.info, meg=False, eog=True, exclude="bads")
    elif signal_type == "emg":
        picks = mne.pick_types(raw.info, meg=False, emg=True, exclude="bads")
    else:
        logger.warning(f"Unknown signal '{signal_type}' for notch filtering.")
        return raw
    if picks is not None and len(picks) > 0:
        try:
            raw.notch_filter(
                freqs=freq,
                picks=picks,
                method="fir",
                phase="zero-double",
                fir_design="firwin",
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Error applying notch filter for {signal_type}: {e}")
    return raw


def plot_signals_mne(
    raw: mne.io.BaseRaw,
    title: str = "Raw Signals",
    scalings: Optional[Dict[str, float]] = None,
    show_annotations: bool = True,
) -> None:
    """Plot raw signals using MNE's interactive raw plotter.

    :param raw: The MNE Raw object to plot.
    :param title: The title of the plot, defaults to "Raw Signals".
    :param scalings: Dictionary of channel type scalings for plotting,
                     e.g., {'eeg': 75e-6}. Defaults to MNE's default if None.
    :param show_annotations: Whether to attempt to plot annotations as events,
                             defaults to True.
    """
    if scalings is None:
        scalings = dict(eeg=75e-6, eog=150e-6, emg=100e-6, misc=1e-3)
    plot_kwargs = {
        "block": True,
        "title": title,
        "show_options": True,
        "scalings": scalings,
        "duration": 20.0,
        "n_channels": len(raw.ch_names),
        "remove_dc": False,
    }
    plot_event_color = {
        1: "green",
        2: "yellow",
        3: "orange",
        4: "red",
        5: "purple",
        0: "gray",
    }
    if (show_annotations and raw.annotations is not None and
            len(raw.annotations)) > 0:
        try:
            events_from_annot, event_dict_from_annot_mapping = (
                mne.events_from_annotations(
                    raw, event_id=ANNOTATION_MAP, chunk_duration=None,
                    verbose=False
                )
            )
            if events_from_annot.size > 0:
                plot_kwargs["events"] = events_from_annot
                present_numeric_ids = np.unique(events_from_annot[:, -1])
                current_plot_event_id = {
                    val: key
                    for key, val in EVENT_ID_MAP.items()
                    if val in present_numeric_ids
                }
                plot_kwargs["event_id"] = current_plot_event_id
                plot_kwargs["event_color"] = {
                    val: plot_event_color[val]
                    for val in current_plot_event_id.keys()
                    if val in plot_event_color
                }
        except ValueError as ve:
            if "No matching events found for" in str(ve):
                logger.warning("No annotations in raw matched ANNOTATION_MAP")
            else:
                logger.warning(f"Could not create events : {ve}")
        except Exception as e:
            logger.warning(
                f"Could not create events for plotting signals: {e}"
            )
    raw.plot(**plot_kwargs)


def plot_epochs_mne(
    epochs: mne.Epochs,
    title: str = "Epochs Plot",
    n_epochs: int = 5,
    scalings: Optional[Dict[str, float]] = None,
) -> None:
    """Plot epoched data using MNE's interactive epochs plotter.

    :param epochs: The MNE Epochs object to plot.
    :param title: The title of the plot, defaults to "Epochs Plot".
    :param n_epochs: The number of epochs to display simultaneously.
    :param scalings: Dictionary of channel type scalings for plotting.
                     Defaults to MNE's default if None.
    """
    if scalings is None:
        scalings = dict(eeg=75e-6, eog=150e-6, emg=100e-6, misc=1e-3)
    plot_event_color = {
        1: "green",
        2: "yellow",
        3: "orange",
        4: "red",
        5: "purple",
        0: "gray",
    }
    epochs_event_id_map = epochs.event_id
    filtered_event_color = {
        val: plot_event_color[val]
        for val in epochs_event_id_map.values()
        if val in plot_event_color
    }
    epochs.plot(
        n_epochs=n_epochs,
        n_channels=len(epochs.ch_names),
        scalings=scalings,
        title=title,
        block=True,
        event_id=epochs_event_id_map,
        event_color=filtered_event_color,
    )


def preprocess_pipeline() -> None:
    """Main function to preprocess the raw data."""
    mne.set_log_level("INFO")
    start_time_total = time.time()
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Processed Data Directory: {PROCESSED_DATA_DIR}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_psg_files = []
    try:
        if not DATA_DIR.is_dir():
            raise FileNotFoundError(
                f"Data directory does not exist: {DATA_DIR}")
        all_psg_files = sorted(list(DATA_DIR.glob("SC*-PSG.edf")))
        if not all_psg_files:
            logger.error(
                f"No PSG files matching 'SC*-PSG.edf' found in {DATA_DIR}."
            )
            exit()
        logger.info(f"Found {len(all_psg_files)} PSG files.")
    except FileNotFoundError as e:
        logger.error(str(e))
        exit()
    except Exception as e:
        logger.error(f"Error finding files: {e}")
        exit()

    files_to_process = all_psg_files[:N_SUBJECTS_TO_PROCESS]
    logger.info(
        f"Attempting to process the first {len(files_to_process)} "
        f"subject(s) using Global AR (EEG/EOG) + Fixed (EMG)..."
    )
    if len(all_psg_files) < N_SUBJECTS_TO_PROCESS:
        logger.info(
            f"Found fewer files ({len(all_psg_files)}) than requested ("
            f"{N_SUBJECTS_TO_PROCESS}), processing all found."
        )

    processed_count = 0
    failed_subjects = []

    for i, psg_file_path in enumerate(files_to_process):
        start_time_subject = time.time()
        subject_id = psg_file_path.stem.replace("-PSG", "")
        logger.info(
            f"--- Processing Subject {i+1}/{len(files_to_process)}: "
            f"{subject_id} ---"
        )

        hypno_file_path = None
        annotations_loaded = False
        try:
            base_id = subject_id[:6]
            if not (
                base_id.startswith("SC") and len(base_id) == 6 and
                base_id[2:].isdigit()
            ):
                raise ValueError(
                    "Unexpected subject ID format for " + f"glob: {subject_id}"
                )

            hypno_files_found = list(DATA_DIR.glob(
                f"{base_id}*-Hypnogram.edf"))

            if len(hypno_files_found) == 1:
                hypno_file_path = hypno_files_found[0]
                logger.info(f"Found hypnogram: {hypno_file_path.name}")
            elif len(hypno_files_found) > 1:
                psg_suffix_part = subject_id[6:]
                best_match = None
                for h_file in hypno_files_found:
                    h_stem_parts = h_file.stem.split("-")
                    if len(h_stem_parts) > 0 and \
                            h_stem_parts[0].startswith(base_id):
                        h_suffix_part = h_stem_parts[0][len(base_id):]
                        if psg_suffix_part == h_suffix_part:
                            best_match = h_file
                            break
                if best_match:
                    hypno_file_path = best_match
                    logger.warning(
                        f"Multiple hypnos for {base_id}, used suffix "
                        + f"match: {hypno_file_path.name}"
                    )
                else:
                    hypno_file_path = hypno_files_found[0]
                    logger.warning(
                        f"Multiple hypnos for {base_id}, used first "
                        + f"found: {hypno_file_path.name}"
                    )
            else:
                logger.warning(
                    "No hypnogram file matching pattern "
                    + f"'{base_id}*-Hypnogram.edf' found for {subject_id}."
                )

        except ValueError as ve_glob:
            logger.error(
                f"Error parsing ID for hypnogram search (glob): {ve_glob}"
            )
        except Exception as e_glob:
            logger.error(
                f"Error searching for hypnogram file for {subject_id} "
                + f"(glob): {e_glob}"
            )

        file_name_base = subject_id

        try:
            raw = None
            logger.info(f"Loading: {psg_file_path.name}")
            exclude_ch = ["Event marker", "Marker", "Status"]
            try:
                with mne.utils.use_log_level("WARNING"):
                    raw = mne.io.read_raw_edf(
                        psg_file_path,
                        preload=True,
                        exclude=exclude_ch,
                        infer_types=True,
                    )
                logger.info(
                    f"Data loaded. SFreq: {raw.info['sfreq']:.2f} Hz. "
                    f"Channels ({len(raw.ch_names)}): {raw.ch_names}"
                )
            except Exception as e:
                logger.error(f"Failed to load {psg_file_path.name}: {e}")
                failed_subjects.append(subject_id)
                continue

            eog_channel_name = "horizontal"
            if eog_channel_name in raw.ch_names:
                try:
                    current_type = raw.get_channel_types(
                        picks=[eog_channel_name])[0]
                    if current_type != "eog":
                        raw.set_channel_types({eog_channel_name: "eog"})
                        logger.info(f"Set type '{eog_channel_name}' to 'eog'.")
                except Exception as e:
                    logger.warning(
                        f"Could not set channel type for '{eog_channel_name}"
                        + f"': {e}"
                    )

            if hypno_file_path is not None:
                try:
                    temp_annots = mne.read_annotations(hypno_file_path)
                    raw.set_annotations(temp_annots, emit_warning=False)
                    annotations_loaded = True
                    logger.info(f"Annotations set from {hypno_file_path.name}")
                except Exception as e:
                    logger.warning(
                        f"Could not load or set annotations from "
                        f"{hypno_file_path.name}: {e}"
                    )

            current_sfreq = raw.info["sfreq"]
            if current_sfreq != TARGET_SFREQ:
                logger.info(
                    f"Resampling data from {current_sfreq:.2f} Hz to "
                    f"{TARGET_SFREQ:.2f} Hz..."
                )
                try:
                    raw.resample(sfreq=TARGET_SFREQ, npad="auto",
                                 verbose=False)
                    logger.info("Resampling complete.")
                except Exception as e:
                    logger.error(f"Error during resampling: {e}. Skipping.")
                    failed_subjects.append(subject_id)
                    continue

            logger.info("Applying filters...")
            raw = bandpass_filter(raw, EEG_BANDPASS[0], EEG_BANDPASS[1], "eeg")
            raw = bandpass_filter(raw, EOG_BANDPASS[0], EOG_BANDPASS[1], "eog")
            raw = bandpass_filter(raw, EMG_BANDPASS[0], EMG_BANDPASS[1], "emg")
            nyquist = TARGET_SFREQ / 2.0
            if NOTCH_FREQ < nyquist:
                logger.info(f"Applying {NOTCH_FREQ} Hz notch filter...")
                if NOTCH_FREQ > EEG_BANDPASS[0] and \
                        NOTCH_FREQ < EEG_BANDPASS[1]:
                    raw = notch_filter(raw, NOTCH_FREQ, "eeg")
                if NOTCH_FREQ > EOG_BANDPASS[0] and \
                        NOTCH_FREQ < EOG_BANDPASS[1]:
                    raw = notch_filter(raw, NOTCH_FREQ, "eog")
                if NOTCH_FREQ > EMG_BANDPASS[0] and \
                        NOTCH_FREQ < EMG_BANDPASS[1]:
                    raw = notch_filter(raw, NOTCH_FREQ, "emg")
            else:
                logger.warning(
                    f"Notch frequency {NOTCH_FREQ} Hz >= Nyquist "
                    f"{nyquist} Hz. Skipping notch filter."
                )
            logger.info("Filtering complete.")

            epochs = None
            if annotations_loaded:
                logger.info("Creating epochs based on loaded annotations...")
                try:
                    events, event_ids_from_annot_func = (
                        mne.events_from_annotations(
                            raw,
                            event_id=ANNOTATION_MAP,
                            chunk_duration=EPOCH_DURATION,
                            verbose=False,
                            )
                        )
                    if events.shape[0] > 0:
                        present_numeric_ids = np.unique(events[:, 2])
                        epochs_event_id_map = {
                            stage_name: stage_id
                            for stage_name, stage_id in EVENT_ID_MAP.items()
                            if stage_id in present_numeric_ids
                        }
                        if not epochs_event_id_map:
                            logger.warning("Falling back.")
                            annotations_loaded = False
                        else:
                            epochs = mne.Epochs(
                                raw,
                                events=events,
                                event_id=epochs_event_id_map,
                                tmin=0.0,
                                tmax=EPOCH_DURATION - 1 / raw.info["sfreq"],
                                preload=True,
                                baseline=None,
                                reject_by_annotation=True,
                                verbose=False,
                            )
                            logger.info(
                                f"Created {len(epochs)} labeled epochs "
                                f"from annotations. Event IDs: "
                                f"{epochs.event_id}"
                            )
                    else:
                        logger.warning("Falling back.")
                        annotations_loaded = False
                except Exception as e:
                    logger.error(
                        f"Error during annotation-based epoch creation: {e}"
                    )
                    traceback.print_exc()
                    logger.warning("Falling back to fixed-length epochs.")
                    annotations_loaded = False

            if not annotations_loaded:
                logger.info("Creating fixed-length epochs...")
                try:
                    epochs = mne.make_fixed_length_epochs(
                        raw,
                        duration=EPOCH_DURATION,
                        preload=True,
                        overlap=0.0,
                        verbose=False,
                    )
                    logger.info(f"Created {len(epochs)} unlabeled epochs.")
                except Exception as e:
                    logger.error(f"Failed to create fixed-length epochs: {e}")
                    failed_subjects.append(subject_id)
                    continue

            if epochs is None or len(epochs) == 0:
                logger.warning("Zero epochs created. Skipping subject.")
                failed_subjects.append(subject_id)
                continue
            initial_epoch_count = len(epochs)
            logger.info(
                f"Initial epoch count before rejection: {initial_epoch_count}"
            )

            logger.info("Applying Global Autoreject")
            epochs_clean = epochs.copy()
            final_reject_dict = {}
            applied_method = "preceding steps (AR failed/skipped)"

            try:
                logger.info("Computing global thresholds for EEG/EOG...")
                ar_reject_dict = get_rejection_threshold(
                    epochs,
                    decim=AUTOREJECT_DECIM,
                    random_state=AUTOREJECT_RANDOM_STATE,
                    verbose=False,
                )

                if ar_reject_dict:
                    logger.info(
                        f"Global Autoreject found EEG/EOG thresholds: "
                        f"{ar_reject_dict}"
                    )
                    final_reject_dict.update(ar_reject_dict)
                else:
                    logger.warning(
                        "get_rejection_threshold did not return "
                        + "EEG/EOG thresholds."
                    )
                emg_picks = mne.pick_types(
                    epochs.info,
                    meg=False,
                    eeg=False,
                    eog=False,
                    emg=True,
                    exclude="bads",
                )
                if len(emg_picks) > 0:
                    logger.info(
                        "Adding fixed EMG threshold: "
                        + f"{FIXED_EMG_THRESHOLD:.2e} V"
                    )
                    final_reject_dict["emg"] = FIXED_EMG_THRESHOLD

                if final_reject_dict:
                    logger.info("Applying combined thresholds: "
                                + f"{final_reject_dict}")
                    epochs_clean.drop_bad(
                        reject=final_reject_dict, verbose=False
                    )
                    applied_method = "Global AR (EEG/EOG) + Fixed (EMG)"
                else:
                    logger.warning("No rejection thresholds defined.")

            except Exception as e:
                logger.error(
                    f"Error during combined rejection processing: {e}"
                )
                traceback.print_exc()
                logger.warning("Proceeding with original epochs due to error.")
                epochs_clean = epochs.copy()

            final_epoch_count = len(epochs_clean)
            logger.info(f"Final clean epoch count: {final_epoch_count}")
            dropped_count = initial_epoch_count - final_epoch_count
            if initial_epoch_count > 0:
                drop_percentage = (dropped_count / initial_epoch_count) * 100
                logger.info(
                    f"Epochs dropped by {applied_method}: "
                    + f"{dropped_count} ({drop_percentage:.1f}%)."
                )
            else:
                logger.info("No epochs to drop.")

            if final_epoch_count > 0:
                if (
                    final_reject_dict
                    and dropped_count == initial_epoch_count
                    and applied_method != "preceding steps (AR failed/skipped)"
                ):
                    logger.warning("All epochs were dropped. Nothing saved.")
                    if subject_id not in failed_subjects:
                        failed_subjects.append(subject_id)
                else:
                    try:
                        save_path = (
                            PROCESSED_DATA_DIR / f"{file_name_base}-epo.fif")
                        epochs_clean.save(
                            save_path, overwrite=True, fmt="single",
                            verbose=False
                        )
                        logger.info(f"Cleaned epochs saved to: {save_path}")
                        processed_count += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to save cleaned epochs for "
                            f"{file_name_base}: {e}"
                        )
                        failed_subjects.append(subject_id)
            else:
                logger.warning("No epochs remaining after artifact rejection.")
                if subject_id not in failed_subjects:
                    failed_subjects.append(subject_id)

        except Exception as e:
            logger.error(
                f"!! UNEXPECTED error processing subject {subject_id}: {e} !!"
            )
            traceback.print_exc()
            failed_subjects.append(subject_id)
            continue

        finally:
            del raw, epochs, epochs_clean
            end_time_subject = time.time()
            logger.info(
                f"--- Time taken for {subject_id}: "
                f"{end_time_subject - start_time_subject:.2f} seconds"
            )
            print("-" * 60)

    end_time_total = time.time()
    logger.info("Processing Finished")
    logger.info(
        f"Successfully processed and saved "
        f"{processed_count}/{len(files_to_process)} subjects."
    )
    total_minutes = (end_time_total - start_time_total) / 60
    logger.info(f"Total time elapsed: {total_minutes:.2f} minutes.")
    if failed_subjects:
        logger.warning(
            f"Processing failed for {len(failed_subjects)}"
            f"subject(s): {sorted(list(set(failed_subjects)))}"
        )


if __name__ == "__main__":
    preprocess_pipeline()
