"""Module for the Support-Vector Machine model.

It implements the SVM architecture, trains and evaluates
the model.
"""

import glob
import os
import pathlib
from collections import defaultdict
from typing import DefaultDict, List, Optional, Set, Tuple, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline as SklearnPipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.utils import shuffle  # type: ignore

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
                        "Error: Could not convert 'feature_names'."
                        + "Fallback needed."
                    )
                    if (
                        X_train is not None
                        and X_train.ndim == 2
                        and X_train.shape[1] == len(ALL_FEATURE_NAMES)
                    ):
                        file_feature_names = ALL_FEATURE_NAMES
        else:
            logger.warning(
                f"'feature_names' not in {npz_file_path}."
                + " Fallback attempted."
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
        return None, None, None, None, None, None, None, None, True
    except KeyError as e:
        logger.error(f"Error: Missing expected key {e} in {npz_file_path}")
        return None, None, None, None, None, None, None, None, True
    except Exception as e:
        logger.error(f"Error loading {npz_file_path}: {e}", exc_info=True)
        return None, None, None, None, None, None, None, None, True


def main_svm() -> None:
    """Main function for training and evaluating the SVM model."""
    split_files = glob.glob(os.path.join(SPLITS_DIR, "split_*.npz"))
    if not split_files:
        logger.error(f"No .npz split files found in {SPLITS_DIR}.")
        return
    logger.info(f"Found {len(split_files)} split files to process for SVM.")

    results_by_config: DefaultDict[str, ResultMetrics] = defaultdict(
        lambda: {"accuracy": [], "f1_macro": [], "roc_auc_ovr": [], "count": 0}
    )
    master_label_set: Set[int] = set()

    for i, split_file in enumerate(sorted(split_files)):
        logger.info(
            f"\n--- Processing Split File {i+1}/{len(split_files)}:"
            + f" {os.path.basename(split_file)} ---"
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
            is_empty,
        ) = load_split_data(split_file)

        if is_empty:
            logger.warning(f"Skipping file {split_file}.")
            continue
        if file_feature_names is None:
            logger.error(f"Feature names undetermined for {split_file}." + " Skipping.")
            continue

        assert (
            y_train_orig is not None
        ), f"y_train_orig is None for {split_file} despite is_empty=False"
        y_train: NDArray[np.int_] = y_train_orig.astype(int)
        master_label_set.update(y_train)

        y_val: Optional[NDArray[np.int_]] = None
        if y_val_orig is not None:
            y_val = y_val_orig.astype(int)
            master_label_set.update(y_val)

        y_test: Optional[NDArray[np.int_]] = None
        if y_test_orig is not None:
            y_test = y_test_orig.astype(int)
            master_label_set.update(y_test)
        else:
            logger.error(
                f"y_test data is missing for {split_file}." + "Skipping this split."
            )
            continue

        current_config_results = results_by_config[fusion_config]
        current_config_results["count"] += 1
        logger.info(f"Processing for Configuration: {fusion_config}")
        logger.info("Training SVM Model (All features from this file's config)...")

        X_train_svm, X_val_svm, X_test_svm = X_train_raw, X_val_raw, X_test_raw

        if X_train_svm is None or X_test_svm is None:
            logger.warning(
                f"Skipping SVM for this fold (config: {fusion_config}):"
                + " X_train_svm or X_test_svm is None."
            )
            current_config_results["accuracy"].append(np.nan)
            current_config_results["f1_macro"].append(np.nan)
            current_config_results["roc_auc_ovr"].append(np.nan)
            continue

        if X_train_svm.size == 0 or X_test_svm.size == 0:
            logger.warning(
                f"Skipping SVM for this fold (config: {fusion_config}):"
                + " Empty train/test feature arrays."
            )
            current_config_results["accuracy"].append(np.nan)
            current_config_results["f1_macro"].append(np.nan)
            current_config_results["roc_auc_ovr"].append(np.nan)
            continue

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
                logger.info(
                    f"Max PCA components ({max_pca_comps}) <= 1." + " Disabling PCA."
                )
                use_pca = False
            if (
                isinstance(pca_n_components, float)
                and int(pca_n_components * max_pca_comps) < 1
                and max_pca_comps
            ) > 0:
                pca_n_components = max_pca_comps
                logger.info(
                    f"Adjusted PCA components to {pca_n_components}"
                    + " as 0.95 variance resulted in <1 component."
                )

            if use_pca:
                svm_pipeline_steps.append(
                    ("pca", PCA(n_components=pca_n_components, random_state=42))
                )

        if imblearn_available and SMOTE is not None:
            svm_pipeline_steps.append(("smote", SMOTE(random_state=42)))  # type: ignore
        svm_pipeline_steps.append(
            (
                "svm",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=42,
                    class_weight="balanced",
                ),
            )
        )

        CurrentPipeline = (
            ImbPipeline if "smote" in dict(svm_pipeline_steps) else SklearnPipeline
        )
        svm_pipeline = CurrentPipeline(svm_pipeline_steps)

        svm_param_grid = {
            "svm__C": [0.1, 1, 10, 50],
            "svm__gamma": [1e-4, 1e-3, 1e-2, 0.1, "scale"],
        }

        X_hp_train_svm: NDArray[np.float64] = X_train_svm
        y_hp_train_svm: NDArray[np.int_] = y_train
        if (
            X_val_svm is not None
            and X_val_svm.size > 0
            and y_val is not None
            and y_val.size > 0
        ):
            X_hp_train_svm = np.vstack((X_train_svm, X_val_svm))
            y_hp_train_svm = np.concatenate((y_train, y_val))
        X_hp_train_svm, y_hp_train_svm = shuffle(
            X_hp_train_svm, y_hp_train_svm, random_state=42
        )

        best_svm = None
        if X_hp_train_svm.shape[0] < 5 or len(np.unique(y_hp_train_svm)) < 2:
            logger.warning(
                "Combined train+val for SVM too small. " + "Fitting directly."
            )
            try:
                svm_pipeline.fit(X_hp_train_svm, y_hp_train_svm)
                best_svm = svm_pipeline
            except Exception as e:
                logger.error(f"Error fitting SVM directly: {e}", exc_info=True)
        else:
            min_samples_class = 0
            if y_hp_train_svm.size > 0:
                if np.all(y_hp_train_svm >= 0):
                    counts = np.bincount(y_hp_train_svm)
                    if counts.size > 0:
                        min_samples_class = (
                            np.min(counts[np.nonzero(counts)]) if np.any(counts) else 0
                        )
                    else:
                        min_samples_class = 0
                else:
                    logger.warning(
                        "y_hp_train_svm contains negative labels,"
                        + " cannot use np.bincount directly."
                    )
                    unique_labels, counts = np.unique(
                        y_hp_train_svm, return_counts=True
                    )
                    min_samples_class = np.min(counts) if len(counts) > 0 else 0

            if ("smote" in svm_pipeline.named_steps and
                    SMOTE is not None):  # type: ignore
                k_val = min(5, min_samples_class - 1) if min_samples_class > 1 else 1
                if k_val < 1:
                    logger.info(
                        f"Calculated k_val for SMOTE is {k_val}."
                        + " Removing SMOTE as k_neighbors too small/invalid."
                    )
                    svm_pipeline_steps_no_smote = [
                        s for s in svm_pipeline_steps if s[0] != "smote"
                    ]
                    svm_pipeline = SklearnPipeline(svm_pipeline_steps_no_smote)
                else:
                    svm_pipeline.set_params(smote__k_neighbors=k_val)  # type: ignore

            n_splits_cv = min(5, min_samples_class) if min_samples_class > 0 else 1
            if n_splits_cv < 2:  # GridSearchCV requires at least 2 splits
                logger.warning(
                    f"Warning: Cannot perform {n_splits_cv}-fold CV for SVM"
                    + f" (min_samples_class: {min_samples_class}). "
                    + "Fitting directly."
                )
                try:
                    svm_pipeline.fit(X_hp_train_svm, y_hp_train_svm)
                    best_svm = svm_pipeline
                except Exception as e:
                    logger.error(f"Error fitting SVM directly: {e}", exc_info=True)
            else:
                gs = GridSearchCV(
                    svm_pipeline,
                    svm_param_grid,
                    cv=n_splits_cv,
                    scoring="f1_macro",
                    n_jobs=-1,
                    error_score="raise",
                )
                try:
                    gs.fit(X_hp_train_svm, y_hp_train_svm)
                    best_svm = gs.best_estimator_
                    logger.info(f"Best SVM Params: {gs.best_params_}")
                except Exception as e:
                    logger.error(f"Error in GridSearchCV for SVM: {e}", exc_info=True)
                    # Fallback
                    try:
                        logger.info(
                            "GridSearchCV failed for SVM. "
                            + "Attempting to fit pipeline directly."
                        )
                        svm_pipeline.fit(X_hp_train_svm, y_hp_train_svm)
                        best_svm = svm_pipeline
                    except Exception as e_fit:
                        logger.error(
                            "Error fitting SVM directly after GS"
                            + f" failure: {e_fit}",
                            exc_info=True,
                        )
                        best_svm = None

        if (
            best_svm
            and X_test_svm is not None
            and X_test_svm.size > 0
            and y_test is not None
        ):
            y_pred = best_svm.predict(X_test_svm)
            current_config_results["accuracy"].append(accuracy_score(y_test, y_pred))
            current_config_results["f1_macro"].append(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            )

            sorted_labels = sorted(list(master_label_set))
            try:
                y_proba = best_svm.predict_proba(X_test_svm)
                unique_y_test_labels = np.unique(y_test)
                if len(unique_y_test_labels) > 1 and y_proba.shape[1] >= len(
                    unique_y_test_labels
                ):
                    roc_auc_labels = sorted(list(unique_y_test_labels))
                    if y_proba.shape[1] == len(sorted_labels):
                        roc_auc_labels = sorted_labels

                    roc_auc = roc_auc_score(
                        y_test,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=roc_auc_labels,
                    )
                    current_config_results["roc_auc_ovr"].append(roc_auc)
                else:
                    logger.warning(
                        "Cannot compute ROC AUC for SVM (config "
                        + f"{fusion_config}, fold {i}): not enough classes in"
                        + f" y_test ({len(unique_y_test_labels)}) or y_proba "
                        + f"shape ({y_proba.shape}) mismatch."
                    )
                    current_config_results["roc_auc_ovr"].append(np.nan)
            except Exception as e:
                current_config_results["roc_auc_ovr"].append(np.nan)
                logger.error(f"SVM ROC AUC error: {e}", exc_info=True)

            logger.info(
                f"SVM Test Performance (Config: {fusion_config}, "
                + f"Fold: {os.path.basename(split_file)}):"
            )
            try:
                report = classification_report(
                    y_test,
                    y_pred,
                    zero_division=0,
                    labels=sorted_labels,
                    target_names=[f"Class {lab}" for lab in sorted_labels],
                )
                logger.info(report)
                cm = confusion_matrix(y_test, y_pred, labels=sorted_labels)
                logger.info(f"Confusion Matrix:\n {cm}")
            except Exception as e_report:
                logger.error(
                    "Error generating classification report/CM for"
                    + f" SVM: {e_report}",
                    exc_info=True,
                )

        else:
            logger.warning(
                "No best_svm model or X_test_svm/y_test is empty"
                + f"/None for SVM (config {fusion_config}, fold {i}"
                + "). Appending NaNs."
            )
            current_config_results["accuracy"].append(np.nan)
            current_config_results["f1_macro"].append(np.nan)
            current_config_results["roc_auc_ovr"].append(np.nan)

    logger.info("\n\n--- Overall Results for SVM Model ---")
    for config_name, results in results_by_config.items():
        if (
            results["count"] > 0
            and results["accuracy"]
            and any(not np.isnan(x) for x in results["accuracy"])
        ):
            logger.info(
                f"\nConfiguration: {config_name} (Processed "
                + f"{results['count']} files)"
            )
            logger.info(
                f"  Mean Accuracy: {np.nanmean(results['accuracy']):.4f} +/- "
                + f"{np.nanstd(results['accuracy']):.4f}"
            )
            logger.info(
                f"  Mean Macro F1-score: {np.nanmean(results['f1_macro']):.4f}"
                + f" +/- {np.nanstd(results['f1_macro']):.4f}"
            )
            logger.info(
                "  Mean ROC AUC (OVR Macro): "
                + f"{np.nanmean(results['roc_auc_ovr']):.4f} +/- "
                + f"{np.nanstd(results['roc_auc_ovr']):.4f}"
            )
        else:
            logger.error(
                "\nNo valid results or all results are NaN for "
                + f"SVM configuration: {config_name}"
            )


if __name__ == "__main__":
    main_svm()
