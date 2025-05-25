"""Module for executing the LOSO CV method on the data.

It takes the preprocessed data, performs data
augmentation on the training splits, and extracts
features for all splits using a fit/transform pipeline.
"""

import pathlib
from typing import Dict, List

import mne
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.exceptions import NotFittedError

from src.data.data_augmentation import (
    create_epochs_from_numpy,
    gaussian_noise,
    sign_flip,
    time_reverse,
)

from src.features.feature_engineering import FeatureEngineering
from src.utils.logger import get_logger

# setup logger
logger = get_logger("LOSO")


try:
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
except NameError:
    SCRIPT_DIR = pathlib.Path(".").resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    logger.warning(
        "Warning: __file__ not found. "
        + f"Assuming script dir: {SCRIPT_DIR}"
    )
    logger.warning(f"Derived project root: {PROJECT_ROOT}")


DATA_SUBFOLDER = "sleep-cassette"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data" / DATA_SUBFOLDER
SPLITS_DIR = PROJECT_ROOT / "data_splits" / DATA_SUBFOLDER
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Splits will be saved in: {SPLITS_DIR}")

# -------------------------------------------------------------------------
# here we load the processed data, perhaps a function that also includes the
# data paths to the processed files
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
        epochs_subject = mne.read_epochs(filepath, preload=True, verbose=False)
        if len(epochs_subject) > 0:
            subjects_epochs_list.append(epochs_subject)
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

# -------------------------------------------------------------------------
# here we just prepare the data before performing loso, so perhaps another
# function for modularity (optionally it can also go in the load_data)
# arrange the data for preparation
X_list = [e.get_data() for e in subjects_epochs_list]
y_list = [e.events[:, -1].astype(np.int_) for e in subjects_epochs_list]
groups_list = [np.full(len(e), i) for i, e in enumerate(subjects_epochs_list)]

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
groups = np.concatenate(groups_list, axis=0)

logger.info(
    f"Combined data shapes: X={X.shape}, y={y.shape}, groups={groups.shape}"
)
logger.info(f"Unique group identifiers: {np.unique(groups)}")
logger.info(f"Number of subjects (groups): {len(subject_ids)}")

# -------------------------------------------------------------------------
# here begins the loso so perhaps a dedicated function for this would be nice
# --- LOSO split ---
loso = LeaveOneGroupOut()
n_splits = loso.get_n_splits(X, y, groups)
logger.info(f"Number of LOSO splits to generate: {n_splits}")

if n_splits != len(subject_ids):
    logger.warning(
        f"Mismatch: Number of splits ({n_splits}) differs "
        + f"from number of subjects ({len(subject_ids)})."
    )

base_mne_info = subjects_epochs_list[0].info
all_ch_names = base_mne_info.ch_names
all_ch_types = base_mne_info.get_channel_types(unique=False, picks="all")

# fusion configuration
fusion_config: Dict[str, List[str]] = {
    "eeg_only": ["eeg"],
    "eeg_eog": ["eeg", "eog"],
    "eeg_emg": ["eeg", "emg"],
    "eeg_eog_emg": ["eeg", "eog", "emg"],
}

# --------------------------
# this is the big loop for loso, before are intitializations and checks
# LOSO CV loop
for i, (train_idx, test_idx) in enumerate(loso.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    current_test_subj_idx = groups[test_idx][0]
    test_subj_id = subject_ids[current_test_subj_idx]

    logger.info(f"\n--- Processing Split {i+1}/{n_splits} ---")
    logger.info(f"Test Subject ID (Group): {test_subj_id}")

    # split training into train and validation
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # --- loop for fusion configuration ---
    for config, channel_types in fusion_config.items():
        logger.info(f"--- Processing Fusion Configuration: {config} ---")

        selected_ch_indices = [
            idx for idx, ch_type in enumerate(all_ch_types)
            if ch_type in channel_types
        ]

        if not selected_ch_indices:
            logger.warning(f"No channels for '{config}'. Skipping.")
            continue

        template_info: mne.Info = mne.create_info(
            ch_names=[all_ch_names[idx] for idx in selected_ch_indices],
            sfreq=base_mne_info["sfreq"],
            ch_types=[all_ch_types[idx] for idx in selected_ch_indices],
        )

        # subset data arrays for current configuration
        X_train_subset = X_train_cv[:, selected_ch_indices, :]
        X_val_subset = X_val_cv[:, selected_ch_indices, :]
        X_test_subset = X_test[:, selected_ch_indices, :]

        # apply augmentations (functions from data_augmentation.py)
        # I am not so sure if this is performed before or after feature
        # engineering, can you check? Normally, the augmentation is before
        # features but idk atp. Also double check whether augmentation is
        # necessary for all classes in the data, as for instance, wake is
        # high everywhere compared to the other classes.

        # augment the subsetted training data
        logger.info("Augmenting internal training data...")
        X_train_aug_list = []
        y_train_aug_list = []

        if X_train_subset.shape[0] > 0:
            logger.info(f"Augmenting internal training data ({config})...")

            # loop through each original epoch and its corresponding label
            for epoch_idx in range(X_train_subset.shape[0]):
                original_epoch = X_train_subset[epoch_idx]
                original_label = y_train_cv[epoch_idx]

                # append the original data and label
                X_train_aug_list.append(original_epoch)
                y_train_aug_list.append(original_label)

                # append the first augmented version and its label
                X_train_aug_list.append(
                    gaussian_noise(original_epoch.copy(), noise_level=0.16)
                )
                y_train_aug_list.append(original_label)

                # append the second augmented version and its label
                X_train_aug_list.append(sign_flip(original_epoch.copy()))
                y_train_aug_list.append(original_label)

                # append the third augmented version and its label
                X_train_aug_list.append(time_reverse(original_epoch.copy()))
                y_train_aug_list.append(original_label)

            # convert the lists into single, large NumPy arrays
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
            X_train_aug = np.array([])
            y_train_aug = np.array([])

        # --- Feature Engineering Pipeline ---
        # initialize the feature engineering class
        feature_pipeline = FeatureEngineering()

        # FIT on training data and TRANSFORM it
        mne_epochs_train = create_epochs_from_numpy(
            X_train_aug, y_train_aug, template_info)
        logger.info(f"Extracting & fitting TRAIN features for '{config}'...")
        X_ft_train, y_ft_train, feature_names = feature_pipeline.fit(
            mne_epochs_train)
        logger.info(f"Extracted TRAIN features shape: {X_ft_train.shape}")

        # TRANSFORM validation data
        mne_epochs_val = create_epochs_from_numpy(
            X_val_subset, y_val_cv, template_info)
        logger.info(f"Extracting VALIDATION features for '{config}'...")
        try:
            X_ft_val, y_ft_val = feature_pipeline.transform(mne_epochs_val)
            logger.info(
                f"Extracted VALIDATION features shape: {X_ft_val.shape}")
        except NotFittedError as e:
            logger.error(f"Validation transform failed: {e}."
                         + " The pipeline wasn't fitted.")
            continue

        # TRANSFORM test data
        mne_epochs_test = create_epochs_from_numpy(
            X_test_subset, y_test, template_info)
        logger.info(f"Extracting TEST features for '{config}'...")
        try:
            X_ft_test, y_ft_test = feature_pipeline.transform(mne_epochs_test)
            logger.info(f"Extracted TEST features shape: {X_ft_test.shape}")
        except NotFittedError as e:
            logger.error(f"Test transform failed: {e}."
                         + " The pipeline wasn't fitted.")
            continue

        # ---------------------------------------------------------
        # perhaps better to have a function here instead (optionally)
        # Save the processed split data
        split_filename = (
            f"split_{str(i).zfill(2)}_test_subj_{test_subj_id}_{config}.npz"
        )
        save_path = SPLITS_DIR / split_filename

        logger.info(f"Saving processed split for '{config}' to {save_path}")
        np.savez_compressed(
            save_path,
            X_train=X_ft_train,
            y_train=y_ft_train,
            X_val=X_ft_val,
            y_val=y_ft_val,
            X_test=X_ft_test,
            y_test=y_ft_test,
            feature_names=np.array(feature_names, dtype=object),
            test_subject_string_id=test_subj_id,
        )
        logger.info(f"Successfully saved split data for '{config}'.")

    logger.info(f"--- Finished all fusions for LOSO Split {i+1} ---")

logger.info("\n--- All LOSO splits and fusions processed and saved ---")
