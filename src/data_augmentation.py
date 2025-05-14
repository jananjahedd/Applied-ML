"""Module for augmenting EEG data and preparing it for model training.

This script handles loading of sleep EEG data, performs leave-one-subject-out
cross-validation, applies data augmentation techniques (like Gaussian noise,
sign flipping, time reversal) to the training folds, extracts features,
and saves the resulting data splits.
"""

import pathlib
from typing import List, Union, cast

import mne
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import LeaveOneGroupOut, train_test_split  # type: ignore

from src.features.feature_engineering import FeatureEngineering
from src.utils.logger import get_logger

# setup logger
logger = get_logger(__name__)


try:
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent
except NameError:
    SCRIPT_DIR = pathlib.Path(".").resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent
    logger.warning(
        "Warning: __file__ not found. " + f"Assuming script dir: {SCRIPT_DIR}"
    )
    logger.warning(f"Derived project root: {PROJECT_ROOT}")


DATA_SUBFOLDER = "sleep-cassette"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data" / DATA_SUBFOLDER
SPLITS_DIR = PROJECT_ROOT / "data_splits" / DATA_SUBFOLDER


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

# LOSO split
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

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Splits will be saved in: {SPLITS_DIR}")

# prepare template MNE info object
# assuming all subjects have the same channel configuration and sampling rate
base_mne_info = subjects_epochs_list[0].info
ch_names = base_mne_info.ch_names
sfreq = base_mne_info["sfreq"]
ch_types = base_mne_info.get_channel_types(unique=False, picks="all")

template_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
if subjects_epochs_list[0].get_montage() is not None:
    template_info.set_montage(subjects_epochs_list[0].get_montage())
logger.info(
    f"Created template MNE Info object with {len(ch_names)} "
    + f"channels, sfreq={sfreq}Hz."
)


def create_epochs_from_numpy(
    data_array: NDArray[np.float64], labels_array: NDArray[np.float64], info: mne.Info
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
            data_array, info=info, events=events, tmin=0.0, baseline=None, verbose=False
        )
        return epochs
    except Exception as e:
        logger.error(f"Failed to create EpochsArray: {e}")
        return None


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

    test_subj_idx = groups[test_idx][0]
    try:
        test_subj_id = subject_ids[test_subj_idx]
    except IndexError:
        logger.error(
            "Could not find subject string ID for numerical "
            + f"ID {test_subj_idx}. Avaliable subject IDs (length "
            + f"{len(subject_ids)}): {subject_ids}"
        )
        continue

    logger.info(f"\n--- Processing Split {i+1}/{n_splits} ---")
    logger.info(
        f"Test Subject ID (Group): {test_subj_id} (Numerical "
        + f"Group Index: {test_subj_idx})"
    )
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if X_train.shape[0] == 0:
        logger.warning(
            f"Skipping split {i+1} for test subject "
            + f"{test_subj_id} due to empty initial "
            + "combined training/validation set."
        )
        continue

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

    logger.info(
        "Internal training set (X_train_cv) for augmentation: "
        + f"{X_train_cv.shape[0]} epochs."
    )
    logger.info("Internal validation set (X_val_cv): " + f"{X_val_cv.shape[0]} epochs.")

    if X_train_cv.shape[0] == 0:
        logger.warning(
            "Training set (X_train_cv) is emtpy for split "
            + f"{i+1} after validation split. No augmentation "
            + "or training features will be generated. "
        )

    X_train_aug = np.array([])
    y_train_aug = np.array([])

    if X_train_cv.shape[0] > 0:
        logger.info("Augmenting internal training data (X_train_cv)...")
        X_train_aug_list = []
        y_train_aug_list = []
        for epoch_idx in range(X_train_cv.shape[0]):
            original_epoch = X_train_cv[epoch_idx]
            original_label = y_train_cv[epoch_idx]

            X_train_aug_list.append(original_epoch)
            y_train_aug_list.append(original_label)

            X_train_aug_list.append(gaussian_noise(original_epoch, noise_level=0.16))
            y_train_aug_list.append(original_label)

            X_train_aug_list.append(sign_flip(original_epoch))
            y_train_aug_list.append(original_label)

            X_train_aug_list.append(time_reverse(original_epoch))
            y_train_aug_list.append(original_label)

        if X_train_aug_list:
            X_train_aug = np.array(X_train_aug_list)
            y_train_aug = np.array(y_train_aug_list)
        logger.info(f"Augmented training set (X_train_aug) shape: {X_train_aug.shape}")
    else:
        logger.info(
            "Internal training set (X_train_cv) was empty. "
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
                "Extracting features for augmented "
                + f"training set (len{mne_epochs_train} epochs)..."
            )
            fe = FeatureEngineering(mne_epochs_train)
            X_ft_train, y_ft_train, feature_names = fe.feature_extraction()
            if feature_names:
                current_split_features = feature_names
            logger.info(f"Extracted training features: train shape {X_ft_train.shape}")
            if not np.array_equal(y_ft_train, y_train_aug):
                logger.warning("y_ft_train label mismatch with y_train_aug!")
        else:
            logger.error("Failed to create MNE Epochs for augmented training.")
    else:
        logger.info(
            "Augmented training set is empty. " + "No features extracted for training."
        )

    # validation set features
    X_ft_val, y_ft_val = np.array([]), np.array([])
    if X_val_cv.shape[0] > 0:
        mne_epochs_val = create_epochs_from_numpy(X_val_cv, y_val_cv, template_info)
        if mne_epochs_val:
            logger.info(
                "Extracting features for validation "
                + f"set ({len(mne_epochs_val)} epochs)..."
            )
            fe_val = FeatureEngineering(mne_epochs_val)
            X_ft_val, y_ft_val, f_names_val = fe_val.feature_extraction()
            if not current_split_features and f_names_val:
                current_split_features = f_names_val
            elif (
                current_split_features
                and f_names_val
                and current_split_features != f_names_val
            ):
                logger.warning("Feature names mismatch: training vs validation!")
            logger.info(f"Extracted validation features: val shape {X_ft_val.shape}")
            if not np.array_equal(y_ft_val, y_val_cv):
                logger.warning("y_ft_val label mismatch with y_val_cv!")
        else:
            logger.error("Failed to create MNE Epochs for validation set.")
    else:
        logger.info(
            "Validation set (X_val_cv) is empty. "
            + "No features extracted for validation."
        )

    # test set features
    X_ft_test, y_ft_test = np.array([]), np.array([])
    if X_test.shape[0] > 0:
        mne_epochs_test = create_epochs_from_numpy(X_test, y_test, template_info)
        if mne_epochs_test:
            logger.info(
                "Extracting features for test set "
                + f"({len(mne_epochs_test)} epochs)..."
            )
            fe_test = FeatureEngineering(mne_epochs_test)
            X_ft_test, y_ft_test, f_names_test = fe_test.feature_extraction()
            if not current_split_features and f_names_test:
                current_split_features = f_names_test
            elif (
                current_split_features
                and f_names_test
                and current_split_features != f_names_test
            ):
                logger.warning("Feature names mismatch: previous vs test!")
            logger.info(f"Extracted test features: X_ft_test shape {X_ft_test.shape}")
            if not np.array_equal(y_ft_test, y_test):
                logger.warning("y_ft_test label mismatch with y_test_set!")
        else:
            logger.error("Failed to create MNE Epochs for test set.")
    else:
        logger.info(
            "Test set (X_test_set) is empty. " + "No features extracted for test."
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
    split_filename = f"split_{str(i).zfill(2)}_test_subj_{test_subj_id}.npz"
    save_path = SPLITS_DIR / split_filename

    logger.info(f"Attempting to save processed split to {save_path}")
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
            test_subject_group_index=test_subj_idx,
        )
        logger.info(f"Successfully saved split {i+1} data to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save split {i+1} data to {save_path}: {e}")

logger.info("\n--- All LOSO splits processed and saved ---")
