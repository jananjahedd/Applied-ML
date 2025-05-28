"""Module for the Support-Vector Machine model.

It implements the SVM architecture, trains and evaluates
the model.
"""

import glob
import os
import pathlib
from collections import defaultdict
from typing import Any, DefaultDict, List, Optional, Set, Tuple, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from src.utils.logger import get_logger

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore

    imblearn_available = True
    print("INFO: imblearn library found. SMOTE is available.")
except ImportError:
    print(
        "WARNING: imblearn library not found. SMOTE will not be used. "
        "To use SMOTE, please install imbalanced-learn: pip install"
        + " imbalanced-learn"
    )
    ImbPipeline = SklearnPipeline
    SMOTE = None
    imblearn_available = False

# setup logger
logger = get_logger("svm")

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

ALL_FEATURE_NAMES = [
    "Fpz-Cz_delta_RelP",
    "Pz-Oz_delta_RelP",
    "Fpz-Cz_theta_RelP",
    "Pz-Oz_theta_RelP",
    "Fpz-Cz_alpha_RelP",
    "Pz-Oz_alpha_RelP",
    "Fpz-Cz_sigma_RelP",
    "Pz-Oz_sigma_RelP",
    "Fpz-Cz_beta_RelP",
    "Pz-Oz_beta_RelP",
    "horizontal_Var",
    "submental_Mean",
]

SPLITS_DIR = PROJECT_ROOT / "data_splits" / "sleep-cassette"

ProcessedData = Tuple[
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    Optional[List[str]],
    str,
    bool,
]


class ResultMetrics(TypedDict):
    """TypedDict for storing SVM result metrics."""

    accuracy: List[float]
    f1_macro: List[float]
    roc_auc_ovr: List[float]
    count: int


def load_split_data(npz_file_path: str) -> ProcessedData:
    """Loads data and fusion configuration from a single .npz split file.

    :param npz_file_path: path to the split files.
    :return: a tuple containing the following:
        - X_train: training features.
        - y_train: training labels.
        - X_val: validation features.
        - y_val: validation labels.
        - X_test: test features.
        - y_test: test labels.
        - file_feature_names: list of feature names.
        - fusion_config: fusion configuration string.
        - error_occurred: True if an error occurred or data was empty
    """
    X_train, y_train, X_val, y_val, X_test, y_test = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    file_feature_names: Optional[List[str]] = None
    fusion_config: str = "unknown_config_fallback"

    try:
        data = np.load(npz_file_path, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        if "fusion_configuration" in data:
            fusion_config = (
                str(data["fusion_configuration"].item())
                if data["fusion_configuration"].ndim == 0
                else str(data["fusion_configuration"])
            )
        else:
            basename = os.path.basename(npz_file_path)
            parts = basename.replace(".npz", "").split("_")
            subject_id_part_index = -1
            for idx, part_val in enumerate(parts):
                if part_val.startswith("SC") or part_val.startswith("ST"):
                    subject_id_part_index = idx
                    break
            if subject_id_part_index != -1 and len(parts) > subject_id_part_index + 1:
                fusion_config = "_".join(parts[subject_id_part_index + 1 :])
            else:
                fusion_config = "unknown_config_from_filename"

        if "feature_names" in data:
            loaded_features = data["feature_names"]
            if (
                isinstance(loaded_features, np.ndarray)
                and loaded_features.ndim == 0
                and isinstance(loaded_features.item(), list)
            ):
                file_feature_names = loaded_features.item()
            elif isinstance(loaded_features, np.ndarray):
                file_feature_names = list(map(str, loaded_features))
            elif isinstance(loaded_features, list):
                file_feature_names = [str(fn) for fn in loaded_features]
            else:
                logger.warning(
                    f"Warning: 'feature_names' in {npz_file_path} is "
                    + f"of type {type(loaded_features)}."
                    + " Trying list conversion."
                )
                try:
                    file_feature_names = list(map(str, loaded_features))
                except TypeError:
                    logger.error(
                        "Error: Could not convert 'feature_names'." + "Fallback needed."
                    )
                    if (
                        X_train is not None
                        and X_train.ndim == 2
                        and X_train.shape[1] == len(ALL_FEATURE_NAMES)
                    ):
                        file_feature_names = ALL_FEATURE_NAMES
        else:
            logger.warning(
                f"'feature_names' not in {npz_file_path}." + " Fallback attempted."
            )
            if (
                X_train is not None
                and X_train.ndim == 2
                and X_train.shape[1] == len(ALL_FEATURE_NAMES)
            ):
                file_feature_names = ALL_FEATURE_NAMES

        if X_train is None or X_train.size == 0 or y_train is None or y_train.size == 0:
            logger.warning(
                "X_train or y_train is missing or" + f" empty in {npz_file_path}."
            )
            return (None, None, None, None, None, None, None, fusion_config, True)

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            file_feature_names,
            fusion_config,
            False,
        )
    except FileNotFoundError:
        logger.error(f"Error: File not found at {npz_file_path}")
        return None, None, None, None, None, None, None, None, True  # type: ignore
    except KeyError as e:
        logger.error(f"Error: Missing expected key {e} in {npz_file_path}")
        return None, None, None, None, None, None, None, None, True  # type: ignore
    except Exception as e:
        logger.error(f"Error loading {npz_file_path}: {e}", exc_info=True)
        return None, None, None, None, None, None, None, None, True  # type: ignore


def _get_split_files(splits_dir: pathlib.Path) -> List[str]:
    split_files = glob.glob(os.path.join(splits_dir, "split_*.npz"))
    if not split_files:
        logger.error(f"No .npz split files found in {splits_dir}.")
        return []
    return sorted(split_files)


def _prepare_data_for_split(
    X_train_raw: Optional[NDArray[np.float64]],
    y_train_orig: Optional[NDArray[np.int_]],
    X_val_raw: Optional[NDArray[np.float64]],
    y_val_orig: Optional[NDArray[np.int_]],
    X_test_raw: Optional[NDArray[np.float64]],
    y_test_orig: Optional[NDArray[np.int_]],
    file_feature_names: Optional[List[str]],  # Keep for potential future use
    # though not directly used in this simplified version
    split_file_basename: str,
    master_label_set: Set[int],
) -> Tuple[
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    bool,  # should_skip_split
]:
    if (
        X_train_raw is None
        or X_train_raw.size == 0
        or y_train_orig is None
        or y_train_orig.size == 0
    ):
        logger.warning(
            f"X_train or y_train is missing or empty in {split_file_basename}."
        )
        return None, None, None, None, None, None, True

    # Assertions like these can remain or be handled by the return values
    # assert y_train_orig is not None, f"y_train_orig is None for {split_file_basename}"
    y_train_p = y_train_orig.astype(int)
    master_label_set.update(y_train_p)

    y_val_p: Optional[NDArray[np.int_]] = None
    if (
        y_val_orig is not None and X_val_raw is not None and X_val_raw.size > 0
    ):  # Ensure X_val also exists
        y_val_p = y_val_orig.astype(int)
        master_label_set.update(y_val_p)
    elif y_val_orig is not None and (X_val_raw is None or X_val_raw.size == 0):
        logger.warning(
            "y_val_orig present but X_val_raw is missing/empty for"
            f"{split_file_basename}. Discarding y_val."
        )
        X_val_raw = None  # Ensure consistency
        y_val_orig = None

    y_test_p: Optional[NDArray[np.int_]] = None
    if (
        y_test_orig is not None and X_test_raw is not None and X_test_raw.size > 0
    ):  # Ensure X_test also exists
        y_test_p = y_test_orig.astype(int)
        master_label_set.update(y_test_p)
    elif y_test_orig is None or X_test_raw is None or X_test_raw.size == 0:
        logger.error(
            "Critical: y_test data or X_test data is missing/empty"
            f" for {split_file_basename}."
        )
        return (
            X_train_raw,
            y_train_p,
            X_val_raw,
            y_val_p,
            X_test_raw,
            None,
            True,
        )  # Skip if no test data

    # Further checks for X_train_svm, X_test_svm emptiness
    if X_train_raw.size == 0 or (
        X_test_raw is not None and X_test_raw.size == 0
    ):  # X_test_raw already checked basically
        logger.warning(f"Empty train or test feature arrays for {split_file_basename}.")
        return X_train_raw, y_train_p, X_val_raw, y_val_p, X_test_raw, y_test_p, True

    return X_train_raw, y_train_p, X_val_raw, y_val_p, X_test_raw, y_test_p, False


def _build_svm_pipeline(
    X_train_svm: NDArray[np.float64],
    X_val_svm: Optional[NDArray[np.float64]],
    imblearn_available_flag: bool,  # Renamed to avoid conflict with module
    SMOTE_class_ref: Optional[type],  # Renamed
) -> Union[SklearnPipeline, ImbPipeline]:
    n_features = X_train_svm.shape[1]
    use_pca = n_features > 10

    svm_pipeline_steps = [("scaler", StandardScaler())]
    if use_pca:
        num_pca_fit_samples = X_train_svm.shape[0]
        if X_val_svm is not None and X_val_svm.size > 0:
            num_pca_fit_samples += X_val_svm.shape[0]

        max_pca_comps = min(num_pca_fit_samples, n_features)
        pca_n_components: Union[int, float, str] = 0.95
        if max_pca_comps <= 1:
            logger.info(f"Max PCA components ({max_pca_comps}) <= 1. Disabling PCA.")
            use_pca = False  # Update use_pca based on this check
        # Check if 0.95 variance results in < 1 component
        if (
            use_pca
            and isinstance(pca_n_components, float)
            and int(pca_n_components * max_pca_comps) < 1
            and max_pca_comps > 0
        ):
            pca_n_components = max_pca_comps  # Use max available components
            logger.info(
                f"Adjusted PCA components to {pca_n_components} as 0.95 variance "
                "resulted in <1 component."
            )
        if use_pca:  # Re-check use_pca as it might have been disabled
            svm_pipeline_steps.append(
                ("pca", PCA(n_components=pca_n_components, random_state=42))
            )

    if imblearn_available_flag and SMOTE_class_ref is not None:
        svm_pipeline_steps.append(("smote", SMOTE_class_ref(random_state=42)))
    svm_pipeline_steps.append(
        (
            "svm",
            SVC(
                kernel="rbf", probability=True, random_state=42, class_weight="balanced"
            ),
        )
    )
    CurrentPipeline = (
        ImbPipeline if "smote" in dict(svm_pipeline_steps) else SklearnPipeline
    )
    return CurrentPipeline(svm_pipeline_steps)


def _train_svm_model_with_gridsearch(
    pipeline: Union[SklearnPipeline, ImbPipeline],
    param_grid: dict[Any, Any],
    X_hp_train: NDArray[np.float64],
    y_hp_train: NDArray[np.int_],
) -> Optional[SVC]:  # Return type is the estimator from the pipeline
    best_svm_estimator = None
    if X_hp_train.shape[0] < 5 or len(np.unique(y_hp_train)) < 2:
        logger.warning("Combined train+val for SVM too small. Fitting directly.")
        try:
            pipeline.fit(X_hp_train, y_hp_train)
            best_svm_estimator = (
                pipeline  # The whole pipeline is the "best estimator" here
            )
        except Exception as e:
            logger.error(f"Error fitting SVM directly: {e}", exc_info=True)
        return best_svm_estimator

    min_samples_class = 0
    if y_hp_train.size > 0:
        if np.all(y_hp_train >= 0):  # Ensure labels are non-negative for bincount
            counts = np.bincount(y_hp_train)
            if counts.size > 0:  # Check if counts is not empty
                min_samples_class = (
                    np.min(counts[np.nonzero(counts)])
                    if np.any(counts)
                    else 0  # type: ignore
                )
            else:  # Handle case where y_hp_train was non-empty but bincount result
                # is empty (e.g. all labels are 0, then np.nonzero is empty)
                min_samples_class = (
                    0  # type: ignore
                    if len(np.unique(y_hp_train)) <= 1
                    else np.min(np.unique(y_hp_train, return_counts=True)[1])
                )

        else:  # Handle negative labels if they can occur
            logger.warning(
                "y_hp_train_svm contains negative labels, cannot use np.bincount "
                "directly for min_samples_class."
            )
            unique_labels, counts = np.unique(y_hp_train, return_counts=True)
            min_samples_class = np.min(counts) if len(counts) > 0 else 0  # type: ignore

    # Adjust SMOTE k_neighbors if SMOTE is in the pipeline
    if (
        "smote" in pipeline.named_steps and SMOTE is not None
    ):  # Check against global SMOTE
        k_val = min(5, min_samples_class - 1) if min_samples_class > 1 else 1
        if k_val < 1:  # SMOTE k_neighbors must be >= 1
            logger.info(
                f"Calculated k_val for SMOTE is {k_val}. "
                "Removing SMOTE as k_neighbors too small/invalid."
            )
            # Rebuild pipeline without SMOTE
            steps_no_smote = [s for s in pipeline.steps if s[0] != "smote"]
            pipeline = SklearnPipeline(steps_no_smote)  # Fallback to SklearnPipeline
        else:
            try:
                pipeline.set_params(smote__k_neighbors=k_val)
            except (
                ValueError
            ) as e:  # Catch specific error if smote is not in pipeline anymore
                logger.warning(
                    "Could not set smote__k_neighbors, "
                    f"SMOTE might have been removed: {e}"
                )

    n_splits_cv = min(5, min_samples_class) if min_samples_class > 0 else 1
    if n_splits_cv < 2:
        logger.warning(
            f"Warning: Cannot perform {n_splits_cv}-fold CV for SVM "
            f"(min_samples_class: {min_samples_class}). Fitting directly."
        )
        try:
            pipeline.fit(X_hp_train, y_hp_train)
            best_svm_estimator = pipeline
        except Exception as e:
            logger.error(
                f"Error fitting SVM directly after CV check: {e}", exc_info=True
            )
        return best_svm_estimator

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=n_splits_cv,
        scoring="f1_macro",
        n_jobs=-1,
        error_score="raise",  # or np.nan
    )
    try:
        gs.fit(X_hp_train, y_hp_train)
        best_svm_estimator = gs.best_estimator_
        logger.info(f"Best SVM Params: {gs.best_params_}")
    except Exception as e:
        logger.error(f"Error in GridSearchCV for SVM: {e}", exc_info=True)
        logger.info("GridSearchCV failed for SVM. Attempting to fit pipeline directly.")
        try:
            pipeline.fit(X_hp_train, y_hp_train)
            best_svm_estimator = pipeline
        except Exception as e_fit:
            logger.error(
                f"Error fitting SVM directly after GS failure: {e_fit}", exc_info=True
            )

    return best_svm_estimator


def _evaluate_svm_on_test_set(
    best_svm: Union[SklearnPipeline, ImbPipeline],  # best_svm is the pipeline
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int_],
    master_label_set: Set[int],
    fusion_config: str,  # For logging
    split_file_basename: str,  # For logging
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    acc, f1, roc_auc_val = np.nan, np.nan, np.nan
    try:
        y_pred = best_svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        sorted_labels = sorted(list(master_label_set))
        y_proba = best_svm.predict_proba(X_test)
        unique_y_test_labels = np.unique(y_test)

        if len(unique_y_test_labels) > 1 and y_proba.shape[1] >= len(
            unique_y_test_labels
        ):
            # Determine labels for ROC AUC. Prefer master_label_set if dimensions match,
            # otherwise use unique labels from y_test if appropriate.
            roc_auc_labels_to_use = sorted_labels
            if y_proba.shape[1] != len(sorted_labels) and y_proba.shape[1] == len(
                unique_y_test_labels
            ):
                roc_auc_labels_to_use = sorted(list(unique_y_test_labels))
            elif y_proba.shape[1] != len(sorted_labels) and y_proba.shape[1] != len(
                unique_y_test_labels
            ):
                logger.warning(
                    f"predict_proba columns ({y_proba.shape[1]}) match neither "
                    f"master_label_set ({len(sorted_labels)}) nor unique_y_test_labels "
                    f"({len(unique_y_test_labels)}). ROC AUC may be problematic."
                )
                # Try with unique_y_test_labels if it's the only one matching cols
                if y_proba.shape[1] == len(unique_y_test_labels):
                    roc_auc_labels_to_use = sorted(list(unique_y_test_labels))
                # If still no match roc_auc_score might fail or give unexpected results
                # Defaulting to sorted_labels and letting roc_auc_score handle it.

            roc_auc_val = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=roc_auc_labels_to_use,
            )
        else:
            logger.warning(
                f"Cannot compute ROC AUC for SVM (config {fusion_config}, file "
                f"{split_file_basename}): not enough classes in y_test "
                f"({len(unique_y_test_labels)}) or y_proba shape ({y_proba.shape}) "
                "mismatch with available labels."
            )

        logger.info(
            f"SVM Test Performance (Config: {fusion_config}, "
            f"File: {split_file_basename}):"
        )
        report = classification_report(
            y_test,
            y_pred,
            zero_division=0,
            labels=sorted_labels,
            target_names=[f"Class {lab}" for lab in sorted_labels],
        )
        logger.info(f"\n{report}")
        cm = confusion_matrix(y_test, y_pred, labels=sorted_labels)
        logger.info(f"Confusion Matrix:\n{cm}")

    except Exception as e:
        logger.error(
            f"Error during SVM evaluation for {split_file_basename}: {e}", exc_info=True
        )
        # Metrics will remain NaN if an error occurs
    return acc, f1, roc_auc_val


def _log_overall_results(results_by_config: DefaultDict[str, ResultMetrics]) -> None:
    logger.info("\n\n--- Overall Results for SVM Model ---")
    for config_name, results in results_by_config.items():
        if (
            results["count"] > 0
            and results["accuracy"]
            and any(not np.isnan(x) for x in results["accuracy"])
        ):  # Check if any valid accuracy score exists
            logger.info(
                f"\nConfiguration: {config_name} (Processed {results['count']} files)"
            )
            logger.info(
                f"  Mean Accuracy: {np.nanmean(results['accuracy']):.4f} +/- "
                f"{np.nanstd(results['accuracy']):.4f}"
            )
            logger.info(
                f"  Mean Macro F1-score: {np.nanmean(results['f1_macro']):.4f} +/- "
                f"{np.nanstd(results['f1_macro']):.4f}"
            )
            logger.info(
                "  Mean ROC AUC (OVR Macro): "
                f"{np.nanmean(results['roc_auc_ovr']):.4f} +/- "
                f"{np.nanstd(results['roc_auc_ovr']):.4f}"
            )
        else:
            logger.error(
                "\nNo valid results or all results are NaN for "
                f"SVM configuration: {config_name}"
            )


def main_svm() -> None:
    """Main function for training and evaluating the SVM model."""
    split_files = _get_split_files(SPLITS_DIR)
    if not split_files:
        return
    logger.info(f"Found {len(split_files)} split files to process for SVM.")

    results_by_config: DefaultDict[str, ResultMetrics] = defaultdict(
        lambda: {"accuracy": [], "f1_macro": [], "roc_auc_ovr": [], "count": 0}
    )
    master_label_set: Set[int] = set()

    for i, split_file in enumerate(sorted(split_files)):
        logger.info(
            f"\n--- Processing Split File {i + 1}/{len(split_files)}:"
            f" {os.path.basename(split_file)} ---"
        )
        (
            X_train_raw,
            y_train_orig,
            X_val_raw,
            y_val_orig,
            X_test_raw,
            y_test_orig,
            file_feature_names,
            fusion_config,
            error_occurred,
        ) = load_split_data(split_file)

        if (
            error_occurred or X_train_raw is None or y_train_orig is None
        ):  # Basic check from load_split_data
            logger.warning(
                f"Skipping file {split_file} due to loading error "
                "or missing train data."
            )
            continue
        if file_feature_names is None:
            logger.error(f"Feature names undetermined for {split_file}. Skipping.")
            continue

        (
            X_train_p,
            y_train_p,
            X_val_p,
            y_val_p,
            X_test_p,
            y_test_p,
            should_skip_split,
        ) = _prepare_data_for_split(
            X_train_raw,
            y_train_orig,
            X_val_raw,
            y_val_orig,
            X_test_raw,
            y_test_orig,
            file_feature_names,
            os.path.basename(split_file),
            master_label_set,
        )

        if (
            should_skip_split
            or y_test_p is None
            or X_test_p is None
            or X_train_p is None
        ):  # y_test_p check is crucial
            logger.warning(
                f"Skipping processing for {os.path.basename(split_file)} "
                "due to data preparation issues."
            )
            # Ensure metrics are recorded as NaN if processing is skipped
            current_config_results = results_by_config[fusion_config]
            current_config_results["count"] += 1  # Count attempt
            current_config_results["accuracy"].append(np.nan)
            current_config_results["f1_macro"].append(np.nan)
            current_config_results["roc_auc_ovr"].append(np.nan)
            continue

        current_config_results = results_by_config[fusion_config]
        current_config_results["count"] += 1
        logger.info(f"Processing for Configuration: {fusion_config}")
        logger.info("Building SVM Pipeline and Training Model...")

        # X_train_p, X_val_p, X_test_p are already assigned
        # y_train_p, y_val_p, y_test_p are assigned

        svm_pipeline = _build_svm_pipeline(
            X_train_p, X_val_p, imblearn_available, SMOTE
        )

        # Prepare data for hyperparameter tuning
        X_hp_train_svm: NDArray[np.float64] = X_train_p
        y_hp_train_svm: NDArray[np.int_] = y_train_p  # type: ignore
        if (
            X_val_p is not None
            and X_val_p.size > 0
            and y_val_p is not None
            and y_val_p.size > 0
        ):
            X_hp_train_svm = np.vstack((X_train_p, X_val_p))
            y_hp_train_svm = np.concatenate((y_train_p, y_val_p))  # type: ignore
        X_hp_train_svm, y_hp_train_svm = shuffle(
            X_hp_train_svm, y_hp_train_svm, random_state=42
        )

        # Define svm_param_grid (could be a global or passed)
        svm_param_grid = {
            "svm__C": [0.1, 1, 10, 50],
            "svm__gamma": [1e-4, 1e-3, 1e-2, 0.1, "scale"],
        }

        best_svm = _train_svm_model_with_gridsearch(
            svm_pipeline, svm_param_grid, X_hp_train_svm, y_hp_train_svm
        )

        if (
            best_svm
            and X_test_p is not None
            and X_test_p.size > 0
            and y_test_p is not None
        ):
            acc, f1, roc_auc = _evaluate_svm_on_test_set(
                best_svm,
                X_test_p,
                y_test_p,
                master_label_set,
                fusion_config,
                os.path.basename(split_file),
            )
            current_config_results["accuracy"].append(
                acc if acc is not None else np.nan
            )
            current_config_results["f1_macro"].append(f1 if f1 is not None else np.nan)
            current_config_results["roc_auc_ovr"].append(
                roc_auc if roc_auc is not None else np.nan
            )
        else:
            logger.warning(
                "No best_svm model or test data missing for SVM (config "
                f"{fusion_config}, file {os.path.basename(split_file)}). "
                "Appending NaNs."
            )
            current_config_results["accuracy"].append(np.nan)
            current_config_results["f1_macro"].append(np.nan)
            current_config_results["roc_auc_ovr"].append(np.nan)

    _log_overall_results(results_by_config)
