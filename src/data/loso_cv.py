"""Module for executing a subject-wise stratified split on the data.

It takes the preprocessed data, splits it into training, validation, and
test sets ensuring that subjects are not shared across splits. It then
separates the data by modality (EEG, EMG, EOG), applies data
augmentation to the training set, and saves each modality's data
separately for future intermediate fusion.
"""

import pathlib
from typing import Dict, List, Tuple

import mne
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.data.data_augmentation import (
    create_epochs_from_numpy,
    gaussian_noise,
    sign_flip,
    time_reverse,
)
from src.features.feature_engineering import FeatureEngineering
from src.utils.logger import get_logger

from src.utils.paths import (get_processed_data_dir,
                             get_repo_root, get_splits_data)


PROJECT_ROOT = pathlib.Path(get_repo_root())
DATA_SUBFOLDER = "sleep-cassette"
PROCESSED_DATA_DIR = pathlib.Path(get_processed_data_dir()) / DATA_SUBFOLDER
SPLITS_DIR = pathlib.Path(get_splits_data())

FUSION_CONFIG: Dict[str, List[str]] = {
    "eeg": ["eeg"],
    "eog": ["eog"],
    "emg": ["emg"],
}

# setup logger
logger = get_logger("splitting")


def load_preprocessed_data() -> Tuple[List[mne.Epochs], List[str]]:
    """Load preprocessed data from the designated folder"""
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
            epochs_subject = mne.read_epochs(
                filepath, preload=True, verbose=False)
            if len(epochs_subject) > 0:
                subjects_epochs_list.append(epochs_subject)
                subject_id_str = filepath.stem.replace("-epo", "")
                subject_ids.append(subject_id_str)
                logger.info(
                    f"Loaded {len(epochs_subject)} epochs for "
                    + f"subject {subject_id_str}"
                )
            else:
                logger.warning(
                    f"Subject {filepath.stem} has zero epochs. Skip.")
        except Exception as e:
            logger.error(f"Failed to load epochs from {filepath}: {e}")

    if not subjects_epochs_list:
        logger.error("No epochs loaded successfully from any subject.")
        exit()

    logger.info(
        "Successfully loaded epochs from "
        + f"{len(subjects_epochs_list)} subjects: {subject_ids}"
    )

    return subjects_epochs_list, subject_ids


def prepare_data(
    subjects_epochs_list: List[mne.Epochs],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the data before performing the split."""
    X_list = [e.get_data() for e in subjects_epochs_list]
    y_list = [e.events[:, -1].astype(np.int64) for e in subjects_epochs_list]
    groups_list = [
        np.full(len(e), i) for i, e in enumerate(subjects_epochs_list)
    ]

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)

    logger.info(
        f"Combined shapes: X={X.shape}, y={y.shape}, groups={groups.shape}"
    )
    logger.info(f"Unique group identifiers: {np.unique(groups)}")

    return X, y, groups


def splitting(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subjects_epochs_list: List[mne.Epochs],
    splits_dir: pathlib.Path
) -> None:
    """
    Performs a subject-wise stratified split, applies augmentation to the
    training set, and then runs feature engineering.

    This function ensures that all data from a single subject belongs to
    exactly one set (train, validation, or test). The splits are
    stratified by the labels 'y'.

    For each data modality (EEG, EMG, EOG), it performs the following:
    1. Augments the training data.
    2. Fits a feature engineering pipeline on the augmented training data.
    3. Transforms the validation and test data using the fitted pipeline.
    4. Saves the engineered features for each split into separate .npz files,
       ready for model training.
    """
    logger.info("--- Starting Subject-Wise Stratified Split ---")

    # split data into training and a temporary set
    train_val_split = GroupShuffleSplit(
        n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_test_idx = next(train_val_split.split(X, y, groups))

    X_train, y_train, groups_train = (X[train_idx], y[train_idx],
                                      groups[train_idx])
    X_val_test, y_val_test, groups_val_test = (
        X[val_test_idx],
        y[val_test_idx],
        groups[val_test_idx],
    )

    # split the temporary set into validation and test sets
    val_test_split = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(val_test_split.split(
        X_val_test, y_val_test, groups_val_test))

    X_val, y_val, groups_val = (
        X_val_test[val_idx],
        y_val_test[val_idx],
        groups_val_test[val_idx],
    )
    X_test, y_test, groups_test = (
        X_val_test[test_idx],
        y_val_test[test_idx],
        groups_val_test[test_idx],
    )

    logger.info(f"Train set shape: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}, y_test: {y_test.shape}")
    logger.info(f"Subjects in train: {np.unique(groups_train)}")
    logger.info(f"Subjects in validation: {np.unique(groups_val)}")
    logger.info(f"Subjects in test: {np.unique(groups_test)}")

    base_mne_info = subjects_epochs_list[0].info
    all_ch_names = base_mne_info.ch_names
    all_ch_types = base_mne_info.get_channel_types(unique=False, picks="all")
    sfreq = base_mne_info["sfreq"]

    # loop over each modality to perform augmentation and feature engineering
    for modality, ch_types_to_include in FUSION_CONFIG.items():
        logger.info(
            f"--- Processing all splits for {modality.upper()} modality ---")

        # select channels and create info object for the current modality
        selected_ch_indices = [
            idx
            for idx, ch_type in enumerate(all_ch_types)
            if ch_type in ch_types_to_include
        ]
        if not selected_ch_indices:
            logger.warning(f"No channels found for '{modality}'. Skipping.")
            continue

        selected_ch_names = [all_ch_names[i] for i in selected_ch_indices]
        modality_ch_types = [all_ch_types[i] for i in selected_ch_indices]
        modality_info = mne.create_info(
            ch_names=selected_ch_names, sfreq=sfreq, ch_types=modality_ch_types
        )

        # slice data for the current modality
        X_train_modality = X_train[:, selected_ch_indices, :]
        X_val_modality = X_val[:, selected_ch_indices, :]
        X_test_modality = X_test[:, selected_ch_indices, :]

        # augment only the training data
        logger.info(f"Augmenting training data for {modality.upper()}...")
        X_train_aug_list, y_train_aug_list, groups_train_aug_list = [], [], []

        if X_train_modality.shape[0] > 0:
            for i in range(X_train_modality.shape[0]):
                epoch, label, group = (X_train_modality[i], y_train[i],
                                       groups_train[i])

                # append original and augmented versions
                X_train_aug_list.extend([
                    epoch,
                    gaussian_noise(epoch.copy(), noise_level=0.1),
                    sign_flip(epoch.copy()),
                    time_reverse(epoch.copy())
                ])
                y_train_aug_list.extend([label] * 4)
                groups_train_aug_list.extend([group] * 4)

            X_train_aug = np.array(X_train_aug_list)
            y_train_aug = np.array(y_train_aug_list)
            groups_train_aug = np.array(groups_train_aug_list)
            logger.info(
                f"Augmented training set for '{modality.upper()}'"
                + f" shape: {X_train_aug.shape}"
            )
        else:
            X_train_aug, y_train_aug, groups_train_aug = (
                np.array([]),
                np.array([]),
                np.array([]),
            )
            logger.info(
                f"Training set for {modality.upper()} is empty."
                + " No augmentation performed."
            )

        # feature Engineering
        logger.info(
            f"--- Applying Feature Engineering for {modality.upper()} ---")
        feature_engineer = FeatureEngineering()

        # fit on augmented training data
        logger.info(
            f"Fitting on augmented training data for {modality.upper()}.")
        epochs_train_aug = create_epochs_from_numpy(
            X_train_aug, y_train_aug, modality_info
        )
        X_train_featured, y_train_featured, feature_names = (
            feature_engineer.fit(epochs_train_aug)
        )

        # transform validation data
        logger.info(f"Transforming validation data for {modality.upper()}.")
        epochs_val = create_epochs_from_numpy(
            X_val_modality, y_val, modality_info)
        X_val_featured, y_val_featured = feature_engineer.transform(epochs_val)

        # transform test data
        logger.info(f"Transforming test data for {modality.upper()}.")
        epochs_test = create_epochs_from_numpy(
            X_test_modality, y_test, modality_info)
        X_test_featured, y_test_featured = feature_engineer.transform(
            epochs_test)

        # save the engineered features
        data_to_save = {
            "train": (X_train_featured, y_train_featured, groups_train_aug),
            "val": (X_val_featured, y_val_featured, groups_val),
            "test": (X_test_featured, y_test_featured, groups_test),
        }

        for split_name, (X_featured, y_featured,
                         groups_split) in data_to_save.items():
            save_filename = f"{split_name}_{modality}_featured.npz"
            save_path = SPLITS_DIR / save_filename
            logger.info(
                f"Saving {split_name} featured data for '{modality}'"
                + f" to {save_path}"
            )
            np.savez_compressed(
                save_path,
                X=X_featured,
                y=y_featured,
                subject_group_ids=groups_split,
                feature_names=np.array(feature_names, dtype=object)
            )
            logger.info(
                f"Successfully saved featured data for '{modality.upper()}'.")

    logger.info("\n--- All data splits processed, engineered, and saved ---")


def main():
    # --- Execution ---
    # Create the output directory if it doesn't exist
    logger.info(f"Creating output directory: {SPLITS_DIR}")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the preprocessed MNE Epochs files
    subjects_epochs_list, _ = load_preprocessed_data()

    # 2. Concatenate data from all subjects into single numpy arrays
    X, y, groups = prepare_data(subjects_epochs_list)

    # 3. Execute the split, augmentation, and feature engineering
    splitting(
        X=X,
        y=y,
        groups=groups,
        subjects_epochs_list=subjects_epochs_list,
        splits_dir=SPLITS_DIR,
    )

    logger.info("✅ Pipeline finished successfully!")


if __name__ == "__main__":
    main()

