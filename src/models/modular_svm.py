"""Module for the Support-Vector Machine model.

It implements the SVM architecture, trains and evaluates
the model, and generates relevant plots.
"""

import pathlib
from collections import defaultdict
from typing import (Any, DefaultDict,
                    Dict, List, Optional,
                    Set, Tuple, TypedDict, Union)

import numpy as np
import matplotlib.pyplot as plt  # For plotting
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from src.utils.paths import (get_splits_data,
                             get_repo_root,
                             get_results_dir)
from src.utils.logger import get_logger

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


# The primary source of feature names should be the loaded .npz files.
ALL_FEATURE_NAMES_REFERENCE = [
    "Fpz-Cz_delta_RelP", "Pz-Oz_delta_RelP",
    "Fpz-Cz_theta_RelP", "Pz-Oz_theta_RelP",
    "Fpz-Cz_alpha_RelP", "Pz-Oz_alpha_RelP",
    "Fpz-Cz_sigma_RelP", "Pz-Oz_sigma_RelP",
    "Fpz-Cz_beta_RelP", "Pz-Oz_beta_RelP",
    "horizontal_Var", "submental_Mean",
]

SPLITS_DIR = PROJECT_ROOT / "data_splits"

# Define configurations to run SVM on
CONFIGS_TO_RUN = ["eeg", "emg", "eog", "eeg_emg", "eeg_eog", "eeg_emg_eog"]


class ResultMetrics(TypedDict):
    """TypedDict for storing SVM result metrics."""
    accuracy: List[float]
    f1_macro: List[float]
    roc_auc_ovr: List[float]
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
                    [f"{modality}_feat{j}" for j in range(num_feats)]
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

    if not X_train_parts:
        return None, None, None, None, None, None, None, True

    final_X_train = np.hstack(X_train_parts)
    final_X_val = np.hstack(X_val_parts)
    final_X_test = np.hstack(X_test_parts)

    logger.info(f"Data loaded successfully for config '{config_key}'")

    return (final_X_train, first_y_train, final_X_val, first_y_val,
            final_X_test, first_y_test, combined_features, False)


def _build_svm_pipeline(
    X_train_svm: NDArray[np.float64],
    X_val_svm: Optional[NDArray[np.float64]],
    imblearn_available_flag: bool,
    SMOTE_class_ref: Optional[type],
) -> Union[SklearnPipeline, ImbPipeline]:  # type: ignore
    n_features = X_train_svm.shape[1]
    use_pca = n_features > 10

    svm_pipeline_steps: List[Tuple[str, Any]] = [("scaler", StandardScaler())]
    if use_pca:
        num_pca_fit_samples = X_train_svm.shape[0]
        if X_val_svm is not None and X_val_svm.size > 0:
            num_pca_fit_samples += X_val_svm.shape[0]

        max_pca_comps = min(num_pca_fit_samples, n_features)
        pca_n_components: Union[int, float, str] = 0.95

        if max_pca_comps <= 1:
            logger.info(
                f"Max PCA components ({max_pca_comps}) or n_features" +
                f"({n_features}) too small. Disabling PCA."
            )
            use_pca = False
        elif isinstance(pca_n_components, float) and \
                int(pca_n_components * max_pca_comps) < 1:
            pca_n_components = max_pca_comps
            logger.info(
                f"Adjusted PCA n_components to {pca_n_components}"
                "as 0.95 variance resulted in <1 component."
            )

        if use_pca:
            svm_pipeline_steps.append(
                ("pca", PCA(n_components=pca_n_components, random_state=42))
            )

    if imblearn_available_flag and SMOTE_class_ref is not None:
        svm_pipeline_steps.append(("smote", SMOTE_class_ref(random_state=42)))

    svm_pipeline_steps.append(
        ("svm", SVC(
            kernel="rbf", probability=True,
            random_state=42, class_weight="balanced"
            ))
    )

    CurrentPipeline = (ImbPipeline if "smote" in dict(svm_pipeline_steps)
                       else SklearnPipeline)  # type: ignore
    return CurrentPipeline(svm_pipeline_steps)


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

    if X_hp_train.shape[0] < 5 or len(np.unique(y_hp_train)) < 2:
        logger.warning(
            "Combined train+val for SVM too small for GridSearchCV."
            " Fitting pipeline directly."
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

    min_samples_class = 0
    if y_hp_train.size > 0:
        unique_labels, counts = np.unique(y_hp_train, return_counts=True)
        if counts.size > 0:
            min_samples_class = np.min(counts)

    current_pipeline_dict = dict(pipeline.steps)
    if "smote" in current_pipeline_dict and SMOTE is not None:
        k_val = min(5, min_samples_class - 1) if min_samples_class > 1 else 1
        if k_val < 1:
            logger.info(
                f"Calculated k_val for SMOTE is {k_val} (<1). "
                "Removing SMOTE from pipeline."
            )
            steps_no_smote = [s for s in pipeline.steps if s[0] != "smote"]
            pipeline = SklearnPipeline(steps_no_smote)
        else:
            try:
                pipeline.set_params(smote__k_neighbors=k_val)
            except ValueError as e:
                logger.warning(
                    f"Could not set smote__k_neighbors (k_val={k_val}), "
                    f"SMOTE might have issues or been removed: {e}"
                )

    n_splits_cv = min(5, min_samples_class) if min_samples_class > 0 else 1
    if n_splits_cv < 2:
        logger.warning(
            f"Cannot perform {n_splits_cv}-fold CV (min_samples_class:"
            f" {min_samples_class}). Fitting pipeline directly."
        )
        try:
            pipeline.fit(X_hp_train, y_hp_train)
            best_estimator = pipeline
        except Exception as e:
            logger.error(
                f"Error fitting SVM pipeline directly after CV check: {e}",
                exc_info=True
            )
        return best_estimator, None

    gs = GridSearchCV(
        pipeline, param_grid, cv=n_splits_cv, scoring="f1_macro",
        n_jobs=-1, error_score="raise", refit=True
    )
    try:
        gs.fit(X_hp_train, y_hp_train)
        best_estimator = gs.best_estimator_
        cv_results_data = gs.cv_results_
        logger.info(f"Best SVM Params from GridSearchCV: {gs.best_params_}")
    except Exception as e:
        logger.error(f"Error in GridSearchCV for SVM: {e}", exc_info=True)
        logger.info(
            "Attempting to fit pipeline directly after GridSearchCV failure."
        )
        try:
            pipeline.fit(X_hp_train, y_hp_train)
            best_estimator = pipeline
        except Exception as e_fit:
            logger.error(
                "Error fitting SVM pipeline directly after GS "
                + f"failure: {e_fit}", exc_info=True
            )

    return best_estimator, cv_results_data


def _evaluate_svm_on_test_set(
    best_svm_estimator: BaseEstimator,
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int_],
    master_label_set: Set[int],
    fusion_config_key: str,
) -> Tuple[float, float, float, Optional[NDArray[np.int_]]]:
    """Evaluates the model on the test data.

    :param best_svm_estimator: best loaded model after training.
    :param X_test: the testing data.
    :param y_test: the labels of the testing data.
    :param master_label_set: all unique labels across all data encountered.
    :param fusion_config_key: the fusion strategy key.
    :return: a tuple containing the accuracy, f1 score, ROC AUC score,
             confusion matrix
    """
    acc, f1, roc_auc_val = np.nan, np.nan, np.nan
    cm_data = None
    sorted_labels = sorted(list(master_label_set))

    try:
        y_pred = best_svm_estimator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm_data = confusion_matrix(y_test, y_pred, labels=sorted_labels)

        unique_y_test_labels = np.unique(y_test)
        if len(unique_y_test_labels) > 1:
            try:
                y_proba = best_svm_estimator.predict_proba(X_test)
                roc_auc_labels = sorted_labels
                # check for mismtach
                if y_proba.shape[1] == len(unique_y_test_labels) and \
                        len(unique_y_test_labels) < len(sorted_labels):
                    roc_auc_labels = sorted(list(unique_y_test_labels))
                elif y_proba.shape[1] != len(roc_auc_labels):
                    logger.warning(
                        f"predict_proba columns ({y_proba.shape[1]}) do not "
                        f"match number of labels ({len(roc_auc_labels)}) for "
                        "ROC AUC. ROC AUC may be unreliable or fail."
                    )

                roc_auc_val = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro",
                    labels=roc_auc_labels
                )
            except ValueError as roc_e:
                logger.warning(
                    f"Could not compute ROC AUC (ValueError): {roc_e}."
                    f" Config: {fusion_config_key}"
                )
            except Exception as e_proba:
                logger.error(
                    f"Error during predict or ROC AUC calculation: {e_proba}",
                    exc_info=True
                )
        else:
            logger.warning(
                f"Cannot compute ROC AUC for config {fusion_config_key}: "
                f"y_test has only {len(unique_y_test_labels)} unique classes)."
            )

        logger.info(f"SVM Test Performance (Config: {fusion_config_key}):")
        # target_names must match the number of labels used in the report
        report_target_names = [f"Class {lab}" for lab in sorted_labels]
        if len(np.unique(y_pred)) < len(sorted_labels) or \
                len(unique_y_test_labels) < len(sorted_labels):
            # classification report must warn if y_pred and y_test contain
            # different labels
            pass
        report = classification_report(
            y_test, y_pred, zero_division=0, labels=sorted_labels,
            target_names=report_target_names,
        )
        logger.info(f"\n{report}")
        logger.info(
            f"Confusion Matrix (Config: {fusion_config_key}):\n{cm_data}"
        )

    except Exception as e:
        logger.error(
            f"Error during SVM evaluation for config {fusion_config_key}: {e}",
            exc_info=True
        )

    return acc, f1, roc_auc_val, cm_data


def _log_overall_results(
        results_by_config: DefaultDict[str, ResultMetrics]) -> None:
    """Logs the overall results of the SVM model for each configuration.

    :param results_by_config: a dictionary where keys are configuration
                             names (strings) and values are ResultMetrics
                             objects containing the results of the SVM model
                             for that configuration.
    """
    logger.info("\n\n--- Overall Results for SVM Model ---")
    for config_name, results in results_by_config.items():
        if results["count"] > 0 and \
                any(not np.isnan(x) for x in results["accuracy"]):
            logger.info(
                f"\nConfiguration: {config_name} (Processed {results['count']}"
                + " times - typically 1 per config)"
            )
            logger.info(
                f"Mean Accuracy: {np.nanmean(results['accuracy']):.4f} +/-"
                f" {np.nanstd(results['accuracy']):.4f}"
            )
            logger.info(
                f"Mean Macro F1-score: {np.nanmean(results['f1_macro']):.4f}"
                + f" +/- {np.nanstd(results['f1_macro']):.4f}"
            )
            logger.info(
                "Mean ROC AUC (OVR Macro): "
                + f"{np.nanmean(results['roc_auc_ovr']):.4f} +/- "
                + f"{np.nanstd(results['roc_auc_ovr']):.4f}"
            )
        else:
            logger.error(
                "No valid results or all results are NaN for SVM "
                f"configuration: {config_name}"
            )


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
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
            scoring="accuracy"
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

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

        c_values = cv_results[f"param_{param_C}"]
        gamma_values = cv_results[f"param_{param_gamma}"]

        # check if they are masked arrays and convert if necessary
        if isinstance(c_values, np.ma.MaskedArray):
            c_values = c_values.compressed()
        if isinstance(gamma_values, np.ma.MaskedArray):
            gamma_values = gamma_values.compressed()

        unique_Cs = sorted(np.unique(c_values.astype(float)))
        unique_gammas = sorted(np.unique(gamma_values.astype(str)))

        # 2D heatmap
        if len(unique_Cs) > 1 and len(unique_gammas) > 1:
            scores_reshaped = (
                scores.reshape(len(unique_Cs), len(unique_gammas))
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(scores_reshaped,
                           interpolation="nearest",
                           cmap=plt.cm.viridis)
            ax.figure.colorbar(im, ax=ax)
            ax.set_xlabel(param_gamma)
            ax.set_ylabel(param_C)
            ax.set_xticks(
                np.arange(len(unique_gammas)),
                labels=unique_gammas,
                rotation=45,
                ha="right")
            ax.set_yticks(
                np.arange(len(unique_Cs)),
                labels=[f"{c:.4f}" for c in unique_Cs]
            )
            ax.set_title(f"{title}\nMean F1 Macro (Validation)")
            # loop over data dimensions and create text annotations.
            for i in range(len(unique_Cs)):
                for j in range(len(unique_gammas)):
                    ax.text(
                        j, i,
                        f"{scores_reshaped[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color=(
                            "w"
                            if scores_reshaped[i, j]
                            < (scores_reshaped.max() * 0.6)
                            else "black"
                        ),
                    )
        elif len(unique_Cs) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(unique_Cs, scores[:len(unique_Cs)], 'o-')
            ax.set_xlabel(param_C)
            ax.set_ylabel("Mean F1 Macro (Validation)")
            ax.set_title(title)
            ax.set_xscale('log')
        elif len(unique_gammas) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            str_gammas = [str(g) for g in unique_gammas]
            ax.plot(str_gammas, scores[:len(unique_gammas)], 'o-')
            ax.set_xlabel(param_gamma)
            ax.set_ylabel("Mean F1 Macro (Validation)")
            ax.set_title(title)
        else:
            logger.warning(
                "Not enough varying parameters to plot GridSearchCV"
                " results meaningfully."
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


def main_svm() -> None:
    """Main function for training and evaluating the SVM model.

    The model train on various fusion configurations.
    """
    logger.info(f"Plotting outputs will be saved to: {RESULTS_DIR}")

    results_by_config: DefaultDict[str, ResultMetrics] = defaultdict(
        lambda: {"accuracy": [], "f1_macro": [], "roc_auc_ovr": [], "count": 0}
    )
    master_label_set: Set[int] = set()

    for config_idx, fusion_config_key in enumerate(CONFIGS_TO_RUN):
        logger.info(
            f"\n--- Processing Fusion Configuration {config_idx + 1}/"
            + f"{len(CONFIGS_TO_RUN)}: {fusion_config_key} ---"
        )

        # load data for the current fusion configuration
        (
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, error_occurred
        ) = load_data(fusion_config_key, SPLITS_DIR)

        current_config_results = results_by_config[fusion_config_key]
        current_config_results["count"] += 1

        if error_occurred or X_train is None or y_train is None or \
                X_test is None or y_test is None:
            logger.error(
                "Failed to load or prepare data for config "
                f"'{fusion_config_key}'. Skipping."
            )
            current_config_results["accuracy"].append(np.nan)
            current_config_results["f1_macro"].append(np.nan)
            current_config_results["roc_auc_ovr"].append(np.nan)
            continue

        # update master label set with labels from this configuration
        master_label_set.update(y_train)
        if y_val is not None:
            master_label_set.update(y_val)
        master_label_set.update(y_test)

        logger.info(
            "Building SVM Pipeline and Training Model for "
            f"'{fusion_config_key}'..."
        )

        svm_pipeline = _build_svm_pipeline(
            X_train, X_val, imblearn_available, SMOTE
        )

        # prepare data for hyperparameter tuning
        X_hp_train_svm: NDArray[np.float64] = X_train
        y_hp_train_svm: NDArray[np.int_] = y_train
        if X_val is not None and y_val is not None and \
                X_val.size > 0 and y_val.size > 0:
            X_hp_train_svm = np.vstack((X_train, X_val))
            y_hp_train_svm = np.concatenate((y_train, y_val))
        X_hp_train_svm, y_hp_train_svm = shuffle(
            X_hp_train_svm, y_hp_train_svm, random_state=42
        )

        svm_param_grid = {
            "svm__C": [0.1, 1, 10, 50, 100],
            "svm__gamma": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, "scale", "auto"],
        }

        best_svm_estimator, cv_results = _train_svm_model_with_gridsearch(
            svm_pipeline, svm_param_grid, X_hp_train_svm, y_hp_train_svm
        )

        save_path = RESULTS_DIR / f"gs_results_{fusion_config_key}.png"
        if cv_results:
            plot_grid_search_results(
                cv_results, param_C="svm__C", param_gamma="svm__gamma",
                title=f"GridSearchCV Results - {fusion_config_key}",
                save_path=save_path
            )

        if best_svm_estimator:
            acc, f1, roc_auc, cm_data = _evaluate_svm_on_test_set(
                best_svm_estimator, X_test, y_test, master_label_set,
                fusion_config_key
            )
            current_config_results["accuracy"].append(acc)
            current_config_results["f1_macro"].append(f1)
            current_config_results["roc_auc_ovr"].append(roc_auc)

            display_labels = [f"Class {lab}"
                              for lab in sorted(list(master_label_set))]
            save_path = RESULTS_DIR / f"cm_{fusion_config_key}.png"
            if cm_data is not None:
                plot_confusion_matrix(
                    cm_data,
                    display_labels=display_labels,
                    title=f"Confusion Matrix - {fusion_config_key}",
                    save_path=save_path
                )

            final_svc_model = None
            if hasattr(best_svm_estimator, 'named_steps') and \
                    'svm' in best_svm_estimator.named_steps:
                final_svc_model = best_svm_estimator.named_steps['svm']
            elif isinstance(best_svm_estimator, SVC):
                final_svc_model = best_svm_estimator

            save_path = RESULTS_DIR / f"lc_{fusion_config_key}.png"
            if final_svc_model:
                plot_learning_curve(
                    final_svc_model,
                    title=f"Learning Curve SVM - {fusion_config_key}",
                    X=X_hp_train_svm, y=y_hp_train_svm,
                    save_path=save_path
                )
            else:
                logger.warning(
                    "Could not extract final SVC model for learning curve"
                    + f" plotting for config {fusion_config_key}"
                )

        else:
            logger.warning(
                f"No best SVM model found for config '{fusion_config_key}'."
                + " Appending NaNs."
            )
            current_config_results["accuracy"].append(np.nan)
            current_config_results["f1_macro"].append(np.nan)
            current_config_results["roc_auc_ovr"].append(np.nan)

    _log_overall_results(results_by_config)


if __name__ == "__main__":
    """Function for testing the SVM model."""
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Splits Directory used by SVM: {SPLITS_DIR}")
    if not SPLITS_DIR.exists():
        logger.error(
            f"SPLITS_DIR does not exist: {SPLITS_DIR}."
            "Please create it or check path."
        )
        logger.error(
            "Ensure that `split.py` has run and saved its output "
            "(e.g., train_eeg_featured.npz) into this directory."
        )
    else:
        main_svm()
