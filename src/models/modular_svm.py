"""Module for the Support-Vector Machine model.

It implements the SVM architecture, trains and evaluates
the model, and generates relevant plots.
"""

import pathlib
import joblib  # type: ignore
import json
from collections import defaultdict
from typing import (Any, DefaultDict,
                    Dict, List, Optional,
                    Set, Tuple, TypedDict, Union)

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from numpy.typing import NDArray  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.metrics import (  # type: ignore
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    make_scorer,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (GridSearchCV,  # type: ignore
                                     learning_curve,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline as SklearnPipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.utils import shuffle  # type: ignore

from src.utils.paths import (get_splits_data,
                             get_repo_root,
                             get_results_dir)
from src.utils.logger import get_logger
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.metrics import roc_curve, auc  # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif  # type: ignore

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore

    imblearn_available = True
    print("INFO: imblearn library found. SMOTE is available.")
except ImportError:
    print(
        "WARNING: imblearn library not found. SMOTE will not be used. "
        "To use SMOTE, please install imbalanced-learn"
        "command: pip install imbalanced-learn"
    )
    ImbPipeline = SklearnPipeline  # type: ignore
    SMOTE = None  # type: ignore
    imblearn_available = False


logger = get_logger("svm")

PROJECT_ROOT = pathlib.Path(get_repo_root())
SPLITS_DIR = pathlib.Path(get_splits_data())
RESULTS_DIR = pathlib.Path(get_results_dir())

# Define configurations to run SVM on
CONFIGS_TO_RUN = ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]

# global stratified CV
stratified_kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


class ResultMetrics(TypedDict):
    """TypedDict for storing SVM result metrics."""
    test_accuracy: List[float]
    test_f1_macro: List[float]
    test_roc_auc_ovr: List[float]
    test_classification_report: List[str]

    # Training set metrics
    train_accuracy: List[float]
    train_f1_macro: List[float]
    train_roc_auc_ovr: List[float]
    train_classification_report: List[str]

    # Validation set metrics
    val_accuracy: List[float]
    val_f1_macro: List[float]
    val_roc_auc_ovr: List[float]
    val_classification_report: List[str]

    count: int


# Data structure for loaded data for a fusion configuration
LoadedFusionData = Tuple[
    Optional[NDArray[np.float64]], Optional[NDArray[np.int_]],
    Optional[NDArray[np.float64]], Optional[NDArray[np.int_]],
    Optional[NDArray[np.float64]], Optional[NDArray[np.int_]],
    Optional[List[str]],  # List of feature names
    bool
]


def load_data(
    config_key: str,
    splits_dir: pathlib.Path
) -> LoadedFusionData:
    """Loads and combines data for a given fusion configuration."""
    modalities_to_load = config_key.split('_')
    logger.info(
        f"Loading data for modalities: {modalities_to_load} for "
        f"config '{config_key}'"
    )

    X_train_parts, X_val_parts, X_test_parts = [], [], []
    combined_features: List[str] = []

    first_y_train, first_y_val, first_y_test = None, None, None

    for modality in modalities_to_load:
        try:
            train_path = splits_dir / f"train_{modality}_featured.npz"
            val_path = splits_dir / f"val_{modality}_featured.npz"
            test_path = splits_dir / f"test_{modality}_featured.npz"

            if not train_path.exists() or not val_path.exists() or \
                    not test_path.exists():
                logger.error(
                    f"Data files for modality '{modality}' not found."
                    " Skipping config '{config_key}'."
                )
                return None, None, None, None, None, None, None, True

            train_data = np.load(train_path, allow_pickle=True)
            val_data = np.load(val_path, allow_pickle=True)
            test_data = np.load(test_path, allow_pickle=True)

            X_train_parts.append(train_data['X_train'])
            X_val_parts.append(val_data['X_val'])
            X_test_parts.append(test_data['X_test'])

            if 'feature_names' in train_data:
                combined_features.extend(list(train_data['feature_names']))
            else:
                num_feats = train_data['X_train'].shape[1]
                combined_features.extend(
                    [f"{modality}_feat{j+1}" for j in range(num_feats)]
                )

            # load once and verify
            y_train_mod, y_val_mod, y_test_mod = (train_data['y_train'],
                                                  val_data['y_val'],
                                                  test_data['y_test'])

            if first_y_train is None:
                # store the labels
                first_y_train = y_train_mod
                first_y_val = y_val_mod
                first_y_test = y_test_mod
            else:
                if not np.array_equal(first_y_train, y_train_mod):
                    logger.error(
                        f"Training label mismatch for modality '{modality}'."
                    )
                    return None, None, None, None, None, None, None, True
                if not np.array_equal(first_y_val, y_val_mod):
                    logger.error(
                        f"Validation label mismatch for modality '{modality}'."
                    )
                    return None, None, None, None, None, None, None, True
                if not np.array_equal(first_y_test, y_test_mod):
                    logger.error(
                        f"Test label mismatch for modality '{modality}'."
                    )
                    return None, None, None, None, None, None, None, True

        except Exception as e:
            logger.error(
                f"Error loading data for modality '{modality}': {e}",
                exc_info=True
            )
            return None, None, None, None, None, None, None, True

    if not X_train_parts or first_y_train is None:
        logger.error(
            f"No data parts loaded or labels missing for config"
            f" '{config_key}'."
        )
        return None, None, None, None, None, None, None, True

    final_X_train = np.hstack(X_train_parts) if X_train_parts else np.array([])
    final_X_val = np.hstack(X_val_parts) if X_val_parts else np.array([])
    final_X_test = np.hstack(X_test_parts) if X_test_parts else np.array([])

    # ensure features are unique
    if len(combined_features) != final_X_train.shape[1] and \
            final_X_train.size > 0:
        logger.warning(
            "Feature name count mismatch. Generating generic feature names."
        )
        combined_features = [
            f"feat_{j+1}" for j in range(final_X_train.shape[1])
        ]

    logger.info(f"Data loaded successfully for config '{config_key}'")
    logger.info(
        f"Shapes: X_train: {final_X_train.shape}, y_train: "
        f"{first_y_train.shape if first_y_train is not None else 'None'}"
    )
    if final_X_val.size > 0:
        logger.info(
            f"Shapes: X_val: {final_X_val.shape}, y_val: "
            f"{first_y_val.shape if first_y_val is not None else 'None'}"
        )
    if final_X_test.size > 0:
        logger.info(
            f"Shapes: X_test: {final_X_test.shape}, y_test: "
            f"{first_y_test.shape if first_y_test is not None else 'None'}"
        )

    return (final_X_train, first_y_train, final_X_val, first_y_val,
            final_X_test, first_y_test, combined_features, False)


def _build_svm_pipeline(
    X_train_svm: NDArray[np.float64],
    X_val_svm: Optional[NDArray[np.float64]],
    imblearn_available_flag: bool,
    SMOTE_class_ref: Optional[type],
) -> Union[SklearnPipeline, ImbPipeline]:  # type: ignore
    n_features = X_train_svm.shape[1]
    if n_features == 0:
        logger.error("Cannot build SVM pipeline: X_train_svm has 0 features.")
        raise ValueError("X_train_svm for pipeline building has 0 features.")

    svm_pipeline_steps: List[Tuple[str, Any]] = [("scaler", StandardScaler())]
    if n_features > 10:
        svm_pipeline_steps.append(
            ("pca", PCA(n_components=0.95, random_state=42))
        )

    if imblearn_available_flag and SMOTE_class_ref is not None:
        svm_pipeline_steps.append(("smote", SMOTE_class_ref(random_state=42)))

    svm_pipeline_steps.append(
        (
            'feature_selection',
            SelectKBest(score_func=f_classif,
                        k=20 if n_features > 20 else 'all')
        )
    )
    calibrated_svc = CalibratedClassifierCV(
        estimator=SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        cv=stratified_kfold,
        method='sigmoid'
    )
    svm_pipeline_steps.append(('svc', calibrated_svc))

    pipeline = ImbPipeline(svm_pipeline_steps)
    return pipeline


def _train_svm_model_with_gridsearch(
    pipeline: Union[SklearnPipeline, ImbPipeline],  # type: ignore
    param_grid: Dict[str, Any],
    X_hp_train: NDArray[np.float64],
    y_hp_train: NDArray[np.int_],
) -> Tuple[Optional[BaseEstimator], Optional[Dict[str, Any]]]:
    """Trains the SVM with GridSearch.

    :param pipeline: the built pipeline.
    :param param_grid: dictionary containing the parameters.
    :param X_hp_train: the training data.
    :param y_hp_train: the labels of the training data.
    :return: tuple containing the best model and the results from
             grid search.
    """
    best_estimator: Optional[BaseEstimator] = None
    cv_results_data: Optional[Dict[str, Any]] = None

    unique_y_classes, y_counts = np.unique(y_hp_train, return_counts=True)
    if X_hp_train.shape[0] < 5 or len(unique_y_classes) < 2:
        logger.warning(
            f"Combined train+val for SVM too small (shape {X_hp_train.shape},"
            f" {len(unique_y_classes)} classes) for GridSearchCV. "
            "Fitting pipeline directly."
        )
        try:
            pipeline.fit(X_hp_train, y_hp_train)
            best_estimator = pipeline
        except Exception as e:
            logger.error(
                f"Error fitting SVM pipeline directly: {e}",
                exc_info=True
            )
        return best_estimator, None

    min_samples_class = np.min(y_counts) if y_counts.size > 0 else 0
    logger.info(
        f"Min samples per class in HP tuning data: {min_samples_class}"
    )

    current_pipeline_dict = dict(pipeline.steps)
    if "smote" in current_pipeline_dict and SMOTE is not None:
        k_val = min(5, min_samples_class - 1) if min_samples_class > 1 else 1
        if k_val == 0 and min_samples_class == 1:
            logger.warning(
                "Min samples per class is 1. SMOTE cannot be applied with "
                "k_neighbors=0. Removing SMOTE."
            )
            original_steps = pipeline.steps
            steps_no_smote = [(name, step) for name, step in original_steps
                              if name != "smote"]
            # determine if ImbPipeline is still needed
            is_still_imblearn = any(
                isinstance(step, (SMOTE.__class__ if SMOTE else object))
                for name, step in steps_no_smote if name != 'smote'
            )
            NewPipelineType = (ImbPipeline if is_still_imblearn and
                               imblearn_available else SklearnPipeline)
            pipeline = NewPipelineType(steps_no_smote)
        elif k_val >= 1:
            try:
                if 'smote__k_neighbors' in pipeline.get_params().keys():
                    pipeline.set_params(smote__k_neighbors=k_val)
                    logger.info(f"SMOTE k_neighbors set to: {k_val}")
                else:
                    logger.warning(
                        "Could not find 'smote__k_neighbors' in pipeline "
                        "params. SMOTE k_val not set."
                    )
            except ValueError as e:
                logger.warning(
                    f"Could not set smote__k_neighbors (k_val={k_val}), "
                    f"SMOTE might have issues: {e}")
        else:
            logger.warning(
                f"Calculated k_val for SMOTE is {k_val}. "
                "SMOTE might be ineffective or error."
            )

    n_splits_cv = min(5, min_samples_class) if min_samples_class >= 2 else 2
    if n_splits_cv < 2:
        logger.warning(
            f"Min samples per class ({min_samples_class}) is less than 2. "
            "GridSearchCV might be unstable or fail. Attempting direct fit."
        )
        try:
            pipeline.fit(X_hp_train, y_hp_train)
            best_estimator = pipeline
        except Exception as e:
            logger.error(
                "Error fitting SVM pipeline directly due to low "
                + f"min_samples_class: {e}",
                exc_info=True
            )
        return best_estimator, None

    scoring = {
        "accuracy": "accuracy",
        'roc_auc_ovr': make_scorer(
            roc_auc_score,
            multi_class='ovr',
            average='macro',
            needs_proba=True
        )
    }
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit='roc_auc_ovr',
        cv=stratified_kfold,
        return_train_score=True,
        n_jobs=-1,
    )
    try:
        logger.info(f"Starting GridSearchCV with cv={n_splits_cv}...")
        gs.fit(X_hp_train, y_hp_train)
        best_estimator = gs.best_estimator_
        cv_results_data = gs.cv_results_
        logger.info(f"Best SVM Params from GridSearchCV: {gs.best_params_}")
        logger.info(f"Best F1 Macro (CV): {gs.best_score_:.4f}")
    except Exception as e:
        logger.error(f"Error in GridSearchCV for SVM: {e}", exc_info=True)
        logger.info(
            "Attempting to fit pipeline directly after GridSearchCV failure."
        )
        try:
            # fit the passed pipeline
            pipeline.fit(X_hp_train, y_hp_train)
            best_estimator = pipeline
        except Exception as e_fit:
            logger.error(
                "Error fitting SVM pipeline directly after GS "
                + f"failure: {e_fit}", exc_info=True
            )

    return best_estimator, cv_results_data


def _evaluate_model(
    estimator: BaseEstimator,
    X_set: Optional[NDArray[np.float64]],
    y_set: Optional[NDArray[np.int_]],
    master_label_set: Set[int],
    set_name: str,
    fusion_config_key: str,
    results_dir: pathlib.Path,
) -> Tuple[float, float, float, str]:
    """Evaluates the model on a given data set and logs/saves results.

    :param estimator: the trained model.
    :param X_set: the input test set.
    :param y_set: the target test set.
    :param master_label_set: all unique labels across all data encountered.
    :param set_name: a string for the set name.
    :param fusion_config_key: the fusion strategy key.
    :param results_dir: the path to the results directory.
    :return: a tuple containing the accuracy, f1 score, ROC AUC score,
             and report
    """
    acc, f1, roc_auc_val = np.nan, np.nan, np.nan
    report_str = "Evaluation not performed or failed."
    cm_data = None

    logger.info(
        f"--- Evaluating on {set_name} set for config "
        + f"'{fusion_config_key}' ---"
    )

    if X_set is None or y_set is None or X_set.size == 0 or y_set.size == 0:
        logger.warning(
            f"Data for {set_name} set is empty or not provided. "
            "Skipping evaluation."
        )
        report_str = f"{set_name} set empty. Evaluation skipped."
        return acc, f1, roc_auc_val, report_str

    if not master_label_set:
        logger.error(
            f"Master label set is empty for {set_name} set evaluation."
            " Cannot proceed."
        )
        return acc, f1, roc_auc_val, "Master label set empty."

    sorted_labels = sorted(list(master_label_set))
    display_labels = [f"Class {lab}" for lab in sorted_labels]

    try:
        y_pred = estimator.predict(X_set)
        acc = accuracy_score(y_set, y_pred)
        f1 = f1_score(y_set, y_pred, average="macro", zero_division=0)
        cm_data = confusion_matrix(y_set, y_pred, labels=sorted_labels)

        unique_y_set_labels = np.unique(y_set)
        if len(unique_y_set_labels) > 1 and \
                hasattr(estimator, "predict_proba"):
            try:
                y_proba = estimator.predict_proba(X_set)

                # use estimator's known classes for ROC AUC labels
                roc_auc_calc_labels = (
                    estimator.classes_ if hasattr(estimator, 'classes_') and
                    len(estimator.classes_) == y_proba.shape[1]
                    else sorted_labels
                )

                # check if y_proba columns match roc_auc_calc_labels length
                if y_proba.shape[1] != len(roc_auc_calc_labels):
                    logger.warning(
                        f"({set_name} set, config {fusion_config_key})"
                        f" predict_proba columns ({y_proba.shape[1]}) "
                        "do not match effective labels for ROC AUC "
                        f"({len(roc_auc_calc_labels)}). Adjusting."
                    )
                    # fallback
                    if y_proba.shape[1] == len(unique_y_set_labels):
                        roc_auc_calc_labels = unique_y_set_labels

                roc_auc_val = roc_auc_score(
                    y_set, y_proba, multi_class="ovr", average="macro",
                    labels=roc_auc_calc_labels
                )
                roc_plot_path = (
                    results_dir /
                    f"roc_{fusion_config_key}_{set_name.lower()}.png"
                )
                plot_roc_curves(
                    estimator, X_set, y_set, sorted_labels, roc_plot_path
                )
                logger.info(f"ROC curves saved to: {roc_plot_path}")

            except ValueError as roc_e:
                logger.warning(
                    f"({set_name} set, config {fusion_config_key}) "
                    f"Could not compute ROC AUC (ValueError): {roc_e}."
                )
                roc_auc_val = np.nan
            except Exception as e_proba:
                logger.error(
                    f"({set_name} set, config {fusion_config_key}) Error "
                    f"during predict_proba or ROC AUC calculation: {e_proba}",
                    exc_info=True
                )
                roc_auc_val = np.nan
        elif not hasattr(estimator, "predict_proba"):
            logger.warning(
                "Estimator does not have predict_proba method. "
                f"ROC AUC not computed for {set_name} set."
            )
            roc_auc_val = np.nan
        else:
            logger.warning(
                f"({set_name} set, config {fusion_config_key})"
                " ROC AUC not computed: y_set has only"
                f" {len(unique_y_set_labels)} unique class(es)."
            )
            roc_auc_val = np.nan

        report_target_names = [f"Class {lab}" for lab in sorted_labels]
        report_str = classification_report(
            y_set, y_pred, labels=sorted_labels,
            target_names=report_target_names, zero_division=0
        )
        logger.info(
            f"SVM {set_name} Performance (Config: {fusion_config_key}):"
        )
        logger.info(f"\n{report_str}")
        logger.info(
            f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, ROC AUC (OVR Macro):"
            f" {roc_auc_val if not np.isnan(roc_auc_val) else 'N/A'}"
        )

        if cm_data is not None:
            logger.info(
                f"Confusion Matrix ({set_name} - Config: "
                f"{fusion_config_key}):\n{cm_data}"
            )
            cm_save_path = (
                results_dir /
                f"cm_{fusion_config_key}_{set_name.lower().replace(' ', '_')}.png"
            )
            plot_confusion_matrix(
                cm_data, display_labels,
                f"CM - {fusion_config_key} - {set_name}",
                cm_save_path
            )

    except Exception as e:
        logger.error(
            f"Error during SVM {set_name} evaluation for config "
            + f"{fusion_config_key}: {e}",
            exc_info=True
        )
        report_str = f"Evaluation failed for {set_name} set: {e}"

    return acc, f1, roc_auc_val, report_str


def _log_overall_results(
        results_by_config: DefaultDict[str, ResultMetrics]) -> None:
    """Logs the overall results of the SVM model for each configuration.

    :param results_by_config: dictionary containing the results.
    """
    logger.info("\n\n--- Overall Summary of SVM Model Runs ---")
    for config_name, results in results_by_config.items():
        if results["count"] == 0:
            logger.info(
                f"\nConfiguration: {config_name} - "
                "No successful runs completed."
            )
            continue

        logger.info(
            f"\nConfiguration: {config_name} (Processed "
            f"{results['count']} time(s))"
        )

        for set_key_prefix, set_name_display in [
            ("train_", "Training"),
            ("val_", "Validation"),
            ("test_", "Test")
        ]:
            acc_key = f"{set_key_prefix}accuracy"
            f1_key = f"{set_key_prefix}f1_macro"
            roc_key = f"{set_key_prefix}roc_auc_ovr"
            # report_key = f"{set_key_prefix}classification_report"

            # check whether there are validation data
            has_valid_data_for_set = False
            if results.get(acc_key) and \
                    not all(np.isnan(v) for v in results[acc_key]):
                has_valid_data_for_set = True

            if not has_valid_data_for_set:
                if set_key_prefix == "val_" and results.get(acc_key) and \
                        all(np.isnan(v) for v in results[acc_key]):
                    logger.info(
                        f"  --- {set_name_display} Set Metrics: "
                        "Skipped or No Data ---"
                    )
                continue

            logger.info(f"  --- {set_name_display} Set Metrics ---")

            metric_map = [
                ("Accuracy", acc_key), ("Macro F1-score", f1_key),
                ("ROC AUC (OVR Macro)", roc_key)
            ]
            for metric_name, data_key in metric_map:
                metric_values = results.get(data_key, [])
                mean_val = (
                    np.nanmean(metric_values) if metric_values else np.nan
                )

                valid_metric_values = [
                    v for v in metric_values if not np.isnan(v)
                ]
                std_val = (
                    np.std(valid_metric_values)
                    if len(valid_metric_values) > 1 else 0.0
                )

                mean_str = (
                    f"{mean_val:.4f}" if not np.isnan(mean_val) else "nan"
                )
                if len(valid_metric_values) > 1:
                    std_str = f"{std_val:.4f}"
                elif len(valid_metric_values) == 1:
                    std_str = "0.0000 (1 run)"
                else:
                    std_str = "nan"

                logger.info(f"  Mean {metric_name}: {mean_str} +/- {std_str}")

    logger.info("--- End of Overall Summary ---")


# plotting functions
def plot_confusion_matrix(
    cm_data: NDArray[np.int_],
    display_labels: List[str],
    title: str,
    save_path: pathlib.Path,
) -> None:
    """Plots and saves the confusion matrix.

    :param cm_data: the confusion matrix generated.
    :param display_labels: the labels to display.
    :param title: title to appear on the saved image.
    :param save_path: the path to the directory to save.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_data, display_labels=display_labels
        )
        disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Confusion matrix plot saved to: {save_path}")
    except Exception as e:
        logger.error(
            f"Error plotting confusion matrix for {title}: {e}",
            exc_info=True
        )


def plot_learning_curve(
    estimator: BaseEstimator,
    title: str,
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    save_path: pathlib.Path,
    cv: Union[int, Any] = 5,
    n_jobs: Optional[int] = -1,
    train_sizes: NDArray[np.float64] = np.linspace(0.1, 1.0, 5),
) -> None:
    """Plots and saves the learning curve.

    :param estimator: the best trained model.
    :param title: the title of the learning curve plot.
    :param X: the input features.
    :param y: the target variable.
    :param save_path: the path to save the plot.
    :param cv: the cross-validation strategy.
    :param n_jobs: the number of jobs to run in parallel.
    :param train_sizes: the relative or absolute numbers of training examples
                        that will be used to generate the learning curve.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=stratified_kfold, n_jobs=n_jobs,
            train_sizes=train_sizes,
            scoring=make_scorer(
                roc_auc_score,
                multi_class="ovr",
                average="macro",
                needs_proba=True
            ),
            error_score=np.nan
        )
        train_scores_mean = np.nanmean(train_scores, axis=1)
        train_scores_std = np.nanstd(train_scores, axis=1)
        val_scores_mean = np.nanmean(val_scores, axis=1)
        val_scores_std = np.nanstd(val_scores, axis=1)

        ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes_abs, train_scores_mean, 'o-',
                color="r", label="Training score")
        ax.plot(train_sizes_abs, val_scores_mean, 'o-',
                color="g", label="Cross-validation score")

        ax.set_title(title)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score (Accuracy)")
        ax.legend(loc="best")
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Learning curve plot saved to: {save_path}")
    except Exception as e:
        logger.error(
            f"Error plotting learning curve for {title}: {e}",
            exc_info=True
        )


def plot_grid_search_results(
    cv_results: Dict[str, Any],
    param_C: str,
    param_gamma: str,
    title: str,
    save_path: pathlib.Path,
) -> None:
    """Plots GridSearchCV results as a heatmap if two params.

    Defaults to line plot if not.
    :param cv_results: dictionary containing the results.
    :param param_C: the C parameter in the grid search.
    :param param_gamma: the gamma parameter in the grid search.
    :param title: title of the plot.
    :param save_path: path to save the plot.
    """
    try:
        scores = cv_results["mean_test_score"]

        # ensure params exist in cv_results
        param_C_key = f"param_{param_C}"
        param_gamma_key = f"param_{param_gamma}"

        if param_C_key not in cv_results or param_gamma_key not in cv_results:
            logger.warning(
                f"One or both params ({param_C_key}, {param_gamma_key}) "
                "not in cv_results. Skipping GS plot."
            )
            # plot for single params if available
            if param_C_key in cv_results and \
                    len(np.unique(cv_results[param_C_key].compressed())) > 1:
                c_values = cv_results[param_C_key].compressed()
                unique_Cs = sorted(np.unique(c_values.astype(float)))
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(unique_Cs, scores[:len(unique_Cs)], 'o-')
                ax.set_xlabel(param_C.split('__')[-1])
                ax.set_ylabel("Mean F1 Macro (Validation)")
                ax.set_title(title)
                ax.set_xscale('log')
                plt.tight_layout()
                fig.savefig(save_path)
                plt.close(fig)
                logger.info(
                    "GridSearchCV results plot (1D for C) saved "
                    f"to: {save_path}"
                )
                return
            elif (
                param_gamma_key in cv_results and 
                len(np.unique(cv_results[param_gamma_key].compressed())) > 1
            ):
                gamma_values = cv_results[param_gamma_key].compressed()
                unique_gammas = sorted(np.unique(gamma_values.astype(str)))
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(unique_gammas, scores[:len(unique_gammas)], 'o-')
                ax.set_xlabel(param_gamma.split('__')[-1])
                ax.set_ylabel("Mean F1 Macro (Validation)")
                ax.set_title(title)
                plt.tight_layout()
                fig.savefig(save_path)
                plt.close(fig)
                logger.info(
                    "GridSearchCV results plot (1D for gamma) saved "
                    f"to: {save_path}"
                )
                return
            else:
                logger.warning(
                    "Not enough varying parameters to plot GridSearchCV"
                    " results meaningfully."
                )
                return

        c_values = cv_results[param_C_key]
        gamma_values = cv_results[param_gamma_key]

        if isinstance(c_values, np.ma.MaskedArray):
            c_values = c_values.compressed()
        if isinstance(gamma_values, np.ma.MaskedArray):
            gamma_values = gamma_values.compressed()

        unique_Cs = sorted(np.unique(c_values.astype(float)))
        unique_gammas_str = sorted(np.unique(gamma_values.astype(str)))

        if len(unique_Cs) > 1 and len(unique_gammas_str) > 1:
            try:
                scores_reshaped = scores.reshape(
                    len(unique_Cs), len(unique_gammas_str)
                )
            except ValueError:
                logger.error(
                    f"Could not reshape scores for GS plot. Check cv_results "
                    f"structure. Number of C values: {len(unique_Cs)}, "
                    f"Gamma values: {len(unique_gammas_str)}, "
                    f"Scores length: {len(scores)}"
                )
                return

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(
                scores_reshaped, interpolation="nearest",
                cmap=plt.cm.viridis, aspect='auto'
            )
            ax.figure.colorbar(im, ax=ax, label="Mean F1 Macro (Validation)")
            ax.set_xlabel(param_gamma.split('__')[-1])
            ax.set_ylabel(param_C.split('__')[-1])
            ax.set_xticks(
                np.arange(len(unique_gammas_str)), labels=unique_gammas_str,
                rotation=45, ha="right"
            )
            ax.set_yticks(
                np.arange(len(unique_Cs)),
                labels=[f"{c:.4f}" for c in unique_Cs]
            )
            ax.set_title(f"{title}")

            for i in range(len(unique_Cs)):
                for j in range(len(unique_gammas_str)):
                    if scores_reshaped[i, j] < (scores_reshaped.max() * 0.6):
                        text_color = "w" 
                    else:
                        text_color = "black"
                    ax.text(
                        j, i, f"{scores_reshaped[i, j]:.3f}", ha="center",
                        va="center", color=text_color
                    )
        elif len(unique_Cs) > 1:
            pass
        elif len(unique_gammas_str) > 1:
            pass
        else:
            logger.warning(
                "Not enough varying parameters for a 2D heatmap in "
                "GridSearchCV results."
            )
            return

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"GridSearchCV results plot saved to: {save_path}")

    except Exception as e:
        logger.error(
            f"Error plotting GridSearchCV results for {title}: {e}",
            exc_info=True
        )


def plot_roc_curves(
    estimator: BaseEstimator,
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    labels: List[int],
    save_path: pathlib.Path
) -> None:
    """One-Vs-Rest ROC curves and macro-avg."""
    ovr = OneVsRestClassifier(estimator)
    y_score = ovr.fit(X, y).predict_proba(X)

    # compute per-class ROC
    fpr, tpr, roc_auc = {}, {}, {}
    for i, lab in enumerate(labels):
        fpr[i], tpr[i], _ = roc_curve((y == lab).astype(int), y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in labels]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in labels:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(labels)
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lab in enumerate(labels):
        ax.plot(fpr[i], tpr[i], lw=1,
                label=f"Class {lab} (AUC = {roc_auc[i]:.2f})")
    ax.plot(fpr["macro"], tpr["macro"], 'k--',
            label=f"Macro-AUC = {roc_auc['macro']:.2f}", lw=2)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _convert_to_json_serializable(item: Any) -> Any:
    """Recursively converts items in a dictionary or list for JSON.

    Converts np.nan to None, np.ndarray to list, and numpy scalars
    to Python types.
    """
    if isinstance(item, dict):
        return {k: _convert_to_json_serializable(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_convert_to_json_serializable(i) for i in item]
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, (np.generic,)):
        return item.item()
    elif isinstance(item, float) and np.isnan(item):
        return None
    return item


def main_svm() -> None:
    logger.info(f"Plotting outputs will be saved to: {RESULTS_DIR}")

    results_by_config: DefaultDict[str, ResultMetrics] = defaultdict(
        lambda: { # Ensures all keys are present with empty lists
            "test_accuracy": [], "test_f1_macro": [], "test_roc_auc_ovr": [],
            "test_classification_report": [], "train_accuracy": [],
            "train_f1_macro": [], "train_roc_auc_ovr": [],
            "train_classification_report": [],
            "val_accuracy": [], "val_f1_macro": [], "val_roc_auc_ovr": [],
            "val_classification_report": [],
            "count": 0
        }
    )
    master_label_set: Set[int] = set()

    for config_idx, fusion_config_key in enumerate(CONFIGS_TO_RUN):
        logger.info(
            f"\n--- Processing Fusion Configuration {config_idx + 1}/"
            f"{len(CONFIGS_TO_RUN)}: {fusion_config_key} ---"
        )

        # get current config results
        current_config_results = results_by_config[fusion_config_key]
        current_config_results["count"] = 1

        def _populate_nan_results(results_dict_ref: ResultMetrics, reason: str):
            for prefix in ["train_", "val_", "test_"]:
                results_dict_ref[f"{prefix}accuracy"].append(np.nan)
                results_dict_ref[f"{prefix}f1_macro"].append(np.nan)
                results_dict_ref[f"{prefix}roc_auc_ovr"].append(np.nan)
                results_dict_ref[f"{prefix}classification_report"].append(
                    reason
                )

        (
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, error_occurred
        ) = load_data(fusion_config_key, SPLITS_DIR)

        # initial data integrity check
        data_load_successful = not error_occurred and \
            X_train is not None and y_train is not None and \
            X_test is not None and y_test is not None and \
            X_train.size > 0 and y_train.size > 0 and \
            X_test.size > 0 and y_test.size > 0

        if not data_load_successful:
            error_msg = (
                f"Data loading failed or data is empty for config "
                f"'{fusion_config_key}'."
            )
            logger.error(error_msg)
            _populate_nan_results(current_config_results, error_msg)
        else:
            master_label_set.update(y_train)
            if y_val is not None and y_val.size > 0:
                master_label_set.update(y_val)
            master_label_set.update(y_test)

            if not master_label_set:
                error_msg = (
                    f"Master label set is empty for config {fusion_config_key}"
                    " after data load."
                )
                logger.error(error_msg)
                _populate_nan_results(current_config_results, error_msg)
            else:
                logger.info(
                    f"Building SVM Pipeline and Training Model for "
                    f"'{fusion_config_key}'..."
                )
                try:
                    svm_pipeline = _build_svm_pipeline(
                        X_train, X_val, imblearn_available, SMOTE
                    )

                    X_hp_train_svm = X_train
                    y_hp_train_svm = y_train
                    if X_val is not None and y_val is not None and \
                            X_val.size > 0 and y_val.size > 0:
                        logger.info(
                            "Combining X_train and X_val for GridSearchCV"
                            " hyperparameter tuning."
                        )
                        X_hp_train_svm = np.vstack((X_train, X_val))
                        y_hp_train_svm = np.concatenate((y_train, y_val))

                    if X_hp_train_svm.size == 0 or y_hp_train_svm.size == 0:
                        error_msg = (
                            f"HP tuning data empty for config "
                            f"'{fusion_config_key}'."
                        )
                        logger.error(error_msg)
                        _populate_nan_results(
                            current_config_results, error_msg
                        )
                    else:
                        X_hp_train_svm, y_hp_train_svm = shuffle(
                            X_hp_train_svm, y_hp_train_svm, random_state=42
                        )
                        svm_param_grid = {
                            "svm__C": [0.1, 1, 10, 50, 100],
                            "svm__gamma": [1e-5, 1e-4, 1e-3, 1e-2, 0.1,
                                           "scale", "auto"],
                        }
                        best_svm_estimator, cv_results = (
                            _train_svm_model_with_gridsearch(
                                svm_pipeline, svm_param_grid,
                                X_hp_train_svm, y_hp_train_svm
                            )
                        )

                        gs_plot_save_path = (
                            RESULTS_DIR / f"gs_results_{fusion_config_key}.png"
                        )
                        if cv_results:
                            plot_grid_search_results(
                                cv_results, param_C="svm__C",
                                param_gamma="svm__gamma",
                                title=f"GridSearchCV Results - {fusion_config_key}",
                                save_path=gs_plot_save_path
                            )

                        if best_svm_estimator:
                            # evaluate on Training set
                            train_acc, train_f1, train_roc, train_report = _evaluate_model(
                                best_svm_estimator, X_train, y_train,
                                master_label_set, "Training",
                                fusion_config_key, RESULTS_DIR
                            )
                            current_config_results["train_accuracy"].append(
                                train_acc
                            )
                            current_config_results["train_f1_macro"].append(
                                train_f1
                            )
                            current_config_results["train_roc_auc_ovr"].append(
                                train_roc
                            )
                            current_config_results[
                                "train_classification_report"].append(
                                    train_report
                                )

                            # evaluate on Validation set
                            if X_val is not None and y_val is not None and X_val.size > 0:
                                val_acc, val_f1, val_roc, val_report = _evaluate_model(
                                    best_svm_estimator, X_val, y_val,
                                    master_label_set, "Validation",
                                    fusion_config_key, RESULTS_DIR
                                )
                                current_config_results["val_accuracy"].append(
                                    val_acc
                                )
                                current_config_results["val_f1_macro"].append(
                                    val_f1
                                )
                                current_config_results[
                                    "val_roc_auc_ovr"].append(val_roc)
                                current_config_results[
                                    "val_classification_report"].append(
                                        val_report
                                    )
                            else:
                                current_config_results["val_accuracy"].append(np.nan)
                                current_config_results["val_f1_macro"].append(np.nan)
                                current_config_results["val_roc_auc_ovr"].append(np.nan)
                                current_config_results["val_classification_report"].append("Validation set not available or empty.")

                            # evaluate on Test set
                            test_acc, test_f1, test_roc, test_report = _evaluate_model(
                                best_svm_estimator, X_test, y_test, master_label_set, "Test", fusion_config_key, RESULTS_DIR)
                            current_config_results["test_accuracy"].append(test_acc)
                            current_config_results["test_f1_macro"].append(test_f1)
                            current_config_results["test_roc_auc_ovr"].append(test_roc)
                            current_config_results["test_classification_report"].append(test_report)

                            model_path = (
                                RESULTS_DIR / f"svm_model_{fusion_config_key}.joblib"
                            )
                            try:
                                joblib.dump(best_svm_estimator, model_path)
                                logger.info(
                                    f"Successfully saved best model for "
                                    f"'{fusion_config_key}' to: {model_path}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to save model for "
                                    f"'{fusion_config_key}'. Error: {e}",
                                    exc_info=True
                                )

                            final_model_for_lc = best_svm_estimator
                            lc_save_path = (
                                RESULTS_DIR / f"lc_{fusion_config_key}.png"
                            )
                            lc_cv_splits = 3
                            if y_hp_train_svm.size > 0:
                                unique_classes_lc, counts_lc = np.unique(
                                    y_hp_train_svm, return_counts=True
                                )
                                if len(unique_classes_lc) > 1 and \
                                        counts_lc.size > 0 and \
                                        np.min(counts_lc) >= 2:
                                    lc_cv_splits = min(5, np.min(counts_lc))

                            plot_learning_curve(
                                final_model_for_lc,
                                title=f"Learning Curve - {fusion_config_key}",
                                X=X_hp_train_svm, y=y_hp_train_svm,
                                save_path=lc_save_path, cv=lc_cv_splits)
                        else:
                            error_msg = (
                                f"No best SVM model found after training for "
                                f"config '{fusion_config_key}'."
                            )
                            logger.warning(error_msg)
                            _populate_nan_results(
                                current_config_results, error_msg)

                except Exception as pipeline_error:
                    error_msg = f"Critical error during pipeline/training for '{fusion_config_key}': {pipeline_error}"
                    logger.error(error_msg, exc_info=True)
                    _populate_nan_results(current_config_results, error_msg)

        # save metrics
        json_save_path = RESULTS_DIR / f"metrics_{fusion_config_key}.json"
        try:
            serializable_results = _convert_to_json_serializable(
                dict(current_config_results)
            )
            with open(json_save_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            logger.info(
                f"Metrics for '{fusion_config_key}' saved to: {json_save_path}"
            )
        except Exception as e_json:
            logger.error(
                f"Failed to save metrics to JSON for '{fusion_config_key}':"
                + f" {e_json}",
                exc_info=True
            )

    _log_overall_results(results_by_config)


if __name__ == "__main__":
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Splits Directory used by SVM: {SPLITS_DIR}")
    logger.info(f"Results Directory: {RESULTS_DIR}")
    if not SPLITS_DIR.exists():
        logger.error(
            f"SPLITS_DIR does not exist: {SPLITS_DIR}. "
            "Please create it or check path. "
            "Ensure that feature extraction and splitting (e.g., loso_cv.py) "
            "has run and saved its output into this directory."
        )
    else:
        main_svm()
