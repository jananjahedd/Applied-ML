"""Module for executing the LOSO CV method on the data.

It takes the preprocessed data, performs data
augmentation on the training splits, and extracts
features for all splits.
"""

import pathlib
from typing import Dict, List, Union, cast

import mne
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import LeaveOneGroupOut, train_test_split  # type: ignore

from src.data.data_augmentation import (
    create_epochs_from_numpy,
    gaussian_noise,
    sign_flip,
    time_reverse,
)
from src.features.feature_engineering import FeatureEngineering
from src.utils.logger import get_logger

# setup logger
logger = get_logger(__name__)


try:
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
except NameError:
    SCRIPT_DIR = pathlib.Path(".").resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    logger.warning(
        "Warning: __file__ not found. " + f"Assuming script dir: {SCRIPT_DIR}"
    )
    logger.warning(f"Derived project root: {PROJECT_ROOT}")


DATA_SUBFOLDER = "sleep-cassette"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data" / DATA_SUBFOLDER
SPLITS_DIR = PROJECT_ROOT / "data_splits" / DATA_SUBFOLDER
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Splits will be saved in: {SPLITS_DIR}")


# Load Preprocessed data

logger.info(f"Loading preprocessed data from: {PROCESSED_DATA_DIR}")

processed_files = sorted(list(PROCESSED_DATA_DIR.glob("SC*-epo.fif")))

if not processed_files:
    logger.error(
        f"No epoch files (*-epo.fif) found in {PROCESSED_DATA_DIR}. "
        "Please run the preprocessing script first."
    )
    exit()

logger.info(f"Found {len(processed_files)} subject files.")

subjects_epochs_list = []
subject_ids = []

for filepath in processed_files:
    try:
        # load epochs for one subject
        epochs_subject = mne.read_epochs(filepath, preload=True, verbose=False)
        if len(epochs_subject) > 0:
            subjects_epochs_list.append(epochs_subject)

            # extract subject ID from filename
            subject_id_str = filepath.stem.replace("-epo", "")
            subject_ids.append(subject_id_str)
            logger.info(
                f"Loaded {len(epochs_subject)} epochs for "
                + f"subject {subject_id_str}"
            )
        else:
            logger.warning(f"Subject {filepath.stem} has zero epochs. Skip.")
    except Exception as e:
        logger.error(f"Failed to load epochs from {filepath}: {e}")

if not subjects_epochs_list:
    logger.error("No epochs loaded successfully from any subject. Exiting.")
    exit()

logger.info(
    "Successfully loaded epochs from "
    + f"{len(subjects_epochs_list)} subjects: {subject_ids}"
)

X: Union[List[NDArray[np.float64]], NDArray[np.float64]] = []
y: Union[List[NDArray[np.int_]], NDArray[np.int_]] = []
groups: Union[List[NDArray[np.int_]], NDArray[np.int_]] = []

for i, epochs_subject in enumerate(subjects_epochs_list):
    subject_id = i
    data: NDArray[np.float64] = epochs_subject.get_data()
    labels: NDArray[np.int_] = epochs_subject.events[:, -1].astype(np.int_)

    if isinstance(X, list):
        X.append(data)
    else:
        logger.error("X was not a list during append phase. Unexpected.")

    if isinstance(y, list):
        y.append(labels)
    else:
        logger.error("y was not a list during append phase. Unexpected.")

    if isinstance(groups, list):
        groups.append(np.full(len(labels), subject_id, dtype=np.int_))
    else:
        logger.error("groups was not a list during append phase. Unexpected.")

# concatenate data from all subjects
if not isinstance(X, list):
    logger.error(
        f"X is not a list before concatenation. Type: {type(X)}. "
        + "Skipping concatenation for X or raising error."
    )
    X_concatenated = X
else:
    X_concatenated = np.concatenate(X, axis=0)
X = cast(NDArray[np.float64], X_concatenated)

if not isinstance(y, list):
    logger.error(
        f"y is not a list before concatenation. Type: {type(y)}. "
        + "Skipping concatenation for y or raising error."
    )
    y_concatenated = y
else:
    y_concatenated = np.concatenate(y, axis=0)
y = cast(NDArray[np.int_], y_concatenated)

if not isinstance(groups, list):
    logger.error(
        f"groups is not a list before concatenation. Type: {type(groups)}. "
        + "Skipping concatenation for groups or raising error."
    )
    groups_concatenated = groups
else:
    groups_concatenated = np.concatenate(groups, axis=0)
groups = cast(NDArray[np.int_], groups_concatenated)

logger.info(
    f"Combined data shapes: X={X.shape}, y={y.shape}, " + f"groups={groups.shape}"
)
logger.info(f"Unique group identifiers: {np.unique(groups)}")
logger.info(f"Number of subjects (groups): {len(subject_ids)}")


# --- LOSO split ---
loso = LeaveOneGroupOut()
n_splits = loso.get_n_splits(X, y, groups)
logger.info(f"Number of LOSO splits to generate: {n_splits}")

if n_splits != len(subject_ids):
    logger.warning(
        f"Mismatch: Number of splits ({n_splits}) "
        + f"differs from number of subjects ({len(subject_ids)})."
    )
if n_splits == 0 and len(subject_ids) > 0:
    logger.error("LeaveOneGroupOut resulted in 0 splits. Exiting.")
    exit()

# prepare template MNE info object
# assuming all subjects have the same channel configuration and sampling rate
if not subjects_epochs_list:
    logger.error("Cannot create base_mne_info as subjects_epochs_list is empty.")
    exit()

base_mne_info = subjects_epochs_list[0].info
all_ch_names = base_mne_info.ch_names
all_ch_types = base_mne_info.get_channel_types(unique=False, picks="all")

logger.info(
    f"Base MNE Info Object created with {len(all_ch_names)} channels, "
    + f"sfreq={base_mne_info['sfreq']}Hz."
)

# fusion configuration
fusion_config: Dict[str, List[str]] = {
    "eeg_only": ["eeg"],
    "eeg_eog": ["eeg", "eog"],
    "eeg_emg": ["eeg", "emg"],
    "eeg_eog_emg": ["eeg", "eog", "emg"],
}

# LOSO CV loop
for i, (train_idx, test_idx) in enumerate(loso.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(test_idx) == 0:
        logger.warning(
            f"Split {i+1}/{n_splits} has an empty test set "
            + "based on indices. Skipping."
        )
        continue

    current_test_subj_idx = groups[test_idx][0]
    try:
        test_subj_id = subject_ids[current_test_subj_idx]
    except IndexError:
        logger.error(
            "Could not find subject string ID for numerical "
            + f"ID {current_test_subj_idx}. Avaliable subject IDs (length "
            + f"{len(subject_ids)}): {subject_ids}"
        )
        continue

    logger.info(f"\n--- Processing Split {i+1}/{n_splits} ---")
    logger.info(
        f"Test Subject ID (Group): {test_subj_id} (Numerical "
        + f"Group Index: {current_test_subj_idx})"
    )
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # split training into train and validation
    val_proportion = 0.2
    if len(np.unique(y_train)) > 1 and y_train.shape[0] >= (1 / val_proportion):
        try:
            X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
                X_train,
                y_train,
                test_size=val_proportion,
                stratify=y_train,
                random_state=42,
            )
        except ValueError as e:
            logger.warning(
                "Stratified split for validation failed in "
                + f"split {i+1}: {e}. Using non-straitifed split."
            )
            X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
                X_train, y_train, test_size=val_proportion, random_state=42
            )
    else:
        logger.info(
            f"Using non-stratified validation split for split {i+1} "
            + "(not enough samples/classes for stratification)."
        )
        X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
            X_train, y_train, test_size=val_proportion, random_state=42
        )

    if X_train_cv.shape[0] == 0:
        logger.warning(
            f"Skipping split {i+1} for test subject {test_subj_id}"
            + " due to empty internal training set."
        )
        continue

    # --- loop for fusion configuration ---
    for config, channel_types in fusion_config.items():
        logger.info(
            f"--- Processing Fusion Configuration: {config} " + f"for Split {i+1} ---"
        )

        selected_ch_indices = [
            idx for idx, ch_type in enumerate(all_ch_types) if ch_type in channel_types
        ]

        if not selected_ch_indices:
            logger.warning(
                f"No channels found for configuration '{config}'"
                + f"(types: {channel_types}). Skipping this "
                + "configuration."
            )
            continue

        logger.info(
            f"Selected {len(selected_ch_indices)} channels" + f" for '{config}'."
        )

        # create a new MNE info object for the current channel subset
        subset_ch_names = [all_ch_names[idx] for idx in selected_ch_indices]
        subset_ch_types = [all_ch_types[idx] for idx in selected_ch_indices]

        template_info: mne.Info = mne.create_info(
            ch_names=subset_ch_names,
            sfreq=base_mne_info["sfreq"],
            ch_types=subset_ch_types,
        )
        original_montage = base_mne_info.get_montage()
        if original_montage is not None:
            try:
                subset_montage = original_montage.copy().pick_channels(
                    subset_ch_names, raise_if_missing=False
                )
                template_info.set_montage(subset_montage)
            except Exception as e_montage:
                logger.warning(
                    f"Could not set montage for subset {config}"
                    + f" with channels {subset_ch_names}: {e_montage}"
                )
        logger.debug(
            f"Created template_info for {config} with "
            + f"channels: {template_info.ch_names}"
        )

        # subset data arrays for current configuration
        X_train_subset: NDArray[np.float64] = X_train_cv[:, selected_ch_indices, :]
        X_val_subset: NDArray[np.float64] = X_val_cv[:, selected_ch_indices, :]
        X_test_subset: NDArray[np.float64] = X_test[:, selected_ch_indices, :]

        logger.info(
            f"Shapes for '{config}': X_train_cv_subset: "
            + f"{X_train_subset.shape}, X_val_cv_subset: "
            + f"{X_val_subset.shape}, X_test_subset: "
            + f"{X_test_subset.shape}"
        )

        # augment the subsetted training data
        X_train_aug = np.array([])
        y_train_aug = np.array([])

        if X_train_subset.shape[0] > 0:
            logger.info(f"Augmenting internal training data ({config})...")
            X_train_aug_list = []
            y_train_aug_list = []
            for epoch_idx in range(X_train_subset.shape[0]):
                original_epoch = X_train_subset[epoch_idx]
                original_label = y_train_cv[epoch_idx]

                X_train_aug_list.append(original_epoch)
                y_train_aug_list.append(original_label)

                X_train_aug_list.append(
                    gaussian_noise(original_epoch.copy(), noise_level=0.16)
                )
                y_train_aug_list.append(original_label)

                X_train_aug_list.append(sign_flip(original_epoch.copy()))
                y_train_aug_list.append(original_label)

                X_train_aug_list.append(time_reverse(original_epoch.copy()))
                y_train_aug_list.append(original_label)

            if X_train_aug_list:
                X_train_aug = np.array(X_train_aug_list)
                y_train_aug = np.array(y_train_aug_list)
            logger.info(
                f"Augmented training set for '{config}' "
                + f"shape: {X_train_aug.shape}"
            )
        else:
            logger.info(
                f"Internal training set (X_train_cv) for {config} was empty. "
                + "No augmentation performed."
            )

        # feature extraction
        current_split_features = []

        X_ft_train, y_ft_train = np.array([]), np.array([])
        if X_train_aug.shape[0] > 0:
            mne_epochs_train = create_epochs_from_numpy(
                X_train_aug, y_train_aug, template_info
            )
            if mne_epochs_train:
                logger.info(
                    f"Extracting TRAIN features for '{config}'"
                    + f" ({len(mne_epochs_train)} epochs)..."
                )
                fe = FeatureEngineering(mne_epochs_train)
                X_ft_train, y_ft_train, feature_names = fe.feature_extraction()
                if feature_names:
                    current_split_features = feature_names
                logger.info(
                    f"Extracted TRAIN features for '{config}': "
                    + f"shape {X_ft_train.shape}"
                )
                if not np.array_equal(y_ft_train, y_train_aug):
                    logger.warning(
                        "y_ft_train label mismatch with y_train_aug"
                        + f" for '{config}'!"
                    )
            else:
                logger.error(
                    "Failed to create MNE Epochs for "
                    + f"augmented training set for '{config}'."
                )
        else:
            logger.info(
                f"Augmented training set for '{config}' is empty."
                + " No TRAIN features extracted."
            )

        # validation set features
        X_ft_val, y_ft_val = np.array([]), np.array([])
        if X_val_subset.shape[0] > 0:
            mne_epochs_val = create_epochs_from_numpy(
                X_val_subset, y_val_cv, template_info
            )
            if mne_epochs_val:
                logger.info(
                    f"Extracting VALIDATION features for '{config}'"
                    + f" ({len(mne_epochs_val)} epochs)..."
                )
                fe = FeatureEngineering(mne_epochs_val)
                X_ft_val, y_ft_val, val_feature_names = fe.feature_extraction()
                if not current_split_features and val_feature_names:
                    current_split_features = val_feature_names
                elif (
                    current_split_features
                    and val_feature_names
                    and current_split_features != val_feature_names
                ):
                    logger.warning(
                        "Feature names mismatch: TRAIN vs "
                        + f"VALIDATION for '{config}'!"
                    )
                logger.info(
                    f"Extracted VALIDATION features for '{config}':"
                    + f" shape {X_ft_val.shape}"
                )
                if not np.array_equal(y_ft_val, y_val_cv):
                    logger.warning(
                        "y_ft_val label mismatch with y_val_cv" f" for '{config}'!"
                    )
            else:
                logger.error(
                    "Failed to create MNE Epochs for "
                    + f"validation set for '{config}'."
                )
        else:
            logger.info(
                f"Validation set (X_val_cv_subset) for '{config}'"
                + " is empty. No VALIDATION features extracted."
            )

        # test set features
        X_ft_test, y_ft_test = np.array([]), np.array([])
        if X_test_subset.shape[0] > 0:
            mne_epochs_test = create_epochs_from_numpy(
                X_test_subset, y_test, template_info
            )
            if mne_epochs_test:
                logger.info(
                    f"Extracting TEST features for '{config}'"
                    + f" ({len(mne_epochs_test)} epochs)..."
                )
                fe = FeatureEngineering(mne_epochs_test)
                X_ft_test, y_ft_test, test_features = fe.feature_extraction()
                if not current_split_features and test_features:
                    current_split_features = test_features
                elif (
                    current_split_features
                    and test_features
                    and current_split_features != test_features
                ):
                    logger.warning(
                        "Feature names mismatch: PREVIOUS" + f" vs TEST for '{config}'!"
                    )
                logger.info(
                    f"Extracted TEST features for '{config}':"
                    + f" shape {X_ft_test.shape}"
                )
                if not np.array_equal(y_ft_test, y_test):
                    logger.warning(
                        "y_ft_test label mismatch with y_test" + f" for '{config}'!"
                    )
            else:
                logger.error(
                    "Failed to create MNE Epochs for " + f"test set for '{config}'."
                )
        else:
            logger.info(
                f"Test set (X_test_subset) for '{config}' is empty."
                + " No TEST features extracted."
            )

        num_active_features = len(current_split_features)
        if X_ft_train.size == 0 and X_train_aug.shape[0] == 0:
            X_ft_train = np.empty((0, num_active_features))
        if y_ft_train.size == 0 and y_train_aug.shape[0] == 0:
            y_ft_train = np.array([])

        if X_ft_val.size == 0 and X_val_cv.shape[0] == 0:
            X_ft_val = np.empty((0, num_active_features))
        if y_ft_val.size == 0 and y_val_cv.shape[0] == 0:
            y_ft_val = np.array([])

        if X_ft_test.size == 0 and X_test.shape[0] == 0:
            X_ft_test = np.empty((0, num_active_features))
        if y_ft_test.size == 0 and y_test.shape[0] == 0:
            y_ft_test = np.array([])

        # save the processed split data
        split_filename = (
            f"split_{str(i).zfill(2)}_test_subj_{test_subj_id}_{config}.npz"
        )
        save_path = SPLITS_DIR / split_filename

        logger.info(
            "Attempting to save processed split " + f"for '{config}' to {save_path}"
        )
        try:
            np.savez_compressed(
                save_path,
                X_train=X_ft_train,
                y_train=y_ft_train,
                X_val=X_ft_val,
                y_val=y_ft_val,
                X_test=X_ft_test,
                y_test=y_ft_test,
                feature_names=np.array(current_split_features, dtype=object),
                test_subject_string_id=test_subj_id,
                test_subject_group_index=current_test_subj_idx,
                fusion_configuration=config,
                used_channel_types=channel_types,
                used_channel_names=subset_ch_names,
            )
            logger.info(
                f"Successfully saved split {i+1} data for '{config}'"
                + f" to {save_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save split {i+1} data for '{config}'"
                + f" to {save_path}: {e}"
            )

    logger.info(f"--- Finished all fusions for LOSO Split {i+1} ---")

logger.info("\n--- All LOSO splits and fusions processed and saved ---")
