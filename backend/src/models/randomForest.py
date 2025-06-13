"""Module for training and evaluating the RandomForestClassifier.

The task involves sleep stage classification, the model takes
the processed files.
"""

import json
import pathlib
from typing import Any, List, Optional, Set, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_curve,
)
from sklearn.model_selection import (  # type: ignore
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils import shuffle  # type: ignore
from src.utils.logger import get_logger
from src.utils.paths import get_repo_root, get_results_dir, get_splits_data

# setup logger
logger = get_logger("final_optimized_classifier")

PROJECT_ROOT = pathlib.Path(get_repo_root())
SPLITS_DIR = pathlib.Path(get_splits_data())
RESULTS_DIR = pathlib.Path(get_results_dir()) / "final_final_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

CONFIGS_TO_RUN = ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# type hint
LoadedFusionData = Tuple[
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    Optional[NDArray[np.float64]],
    Optional[NDArray[np.int_]],
    Optional[List[str]],
    bool,
]


def load_data(config_key: str, splits_dir: pathlib.Path) -> LoadedFusionData:
    """Loads and combines data for a given fusion configuration."""
    modalities_to_load = config_key.split("_")
    logger.info(f"Loading data for modalities: {modalities_to_load}" f" for config '{config_key}'")
    X_train_parts, X_val_parts, X_test_parts = [], [], []
    first_y_train, first_y_val, first_y_test = None, None, None

    for modality in modalities_to_load:
        train_path = splits_dir / f"train_{modality}_featured.npz"
        val_path = splits_dir / f"val_{modality}_featured.npz"
        test_path = splits_dir / f"test_{modality}_featured.npz"

        if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
            return None, None, None, None, None, None, None, True

        train_data, val_data, test_data = (
            np.load(train_path, allow_pickle=True),
            np.load(val_path, allow_pickle=True),
            np.load(test_path, allow_pickle=True),
        )
        X_train_parts.append(train_data["X_train"])
        X_val_parts.append(val_data["X_val"])
        X_test_parts.append(test_data["X_test"])

        y_train_mod, y_val_mod, y_test_mod = (train_data["y_train"], val_data["y_val"], test_data["y_test"])
        if first_y_train is None:
            first_y_train, first_y_val, first_y_test = (y_train_mod, y_val_mod, y_test_mod)
    final_X_train, final_X_val, final_X_test = (
        np.hstack(X_train_parts),
        np.hstack(X_val_parts),
        np.hstack(X_test_parts),
    )

    logger.info(
        f"Data loaded for '{config_key}'. Shapes: Train={final_X_train.shape},"
        f" Val={final_X_val.shape}, Test={final_X_test.shape}"
    )
    return (final_X_train, first_y_train, final_X_val, first_y_val, final_X_test, first_y_test, None, False)


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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=display_labels)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Confusion matrix plot saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix for {title}: {e}", exc_info=True)


def plot_learning_curve(
    estimator: BaseEstimator, title: str, X: NDArray[np.float64], y: NDArray[np.int_], save_path: pathlib.Path
) -> None:
    """Plots and saves the learning curves.

    :param estimator: the best trained model.
    :param title: the title of the learning curve plot.
    :param X: the input features.
    :param y: the target variable.
    :param save_path: the path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=stratified_kfold,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring=make_scorer(f1_score, average="macro", zero_division=0),
    )
    train_scores_mean, train_scores_std = (np.mean(train_scores, axis=1), np.std(train_scores, axis=1))
    val_scores_mean, val_scores_std = (np.mean(val_scores, axis=1), np.std(val_scores, axis=1))
    ax.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes_abs, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g"
    )
    ax.plot(train_sizes_abs, train_scores_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes_abs, val_scores_mean, "o-", color="g", label="Cross-validation score")
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score (Macro F1)")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Learning curve plot saved to: {save_path}")


def calculate_and_plot_roc_auc(
    model: BaseEstimator,
    X: NDArray[np.float64],
    y_true: NDArray[np.int_],
    all_classes: list[int],
    title: str,
    save_path: pathlib.Path,
) -> float:
    """Calculates and plots the ROC curves and AUC scores.

    :param model: the trained model.
    :param X: feature matrix for prediction.
    :param y_true: true class labels.
    :param all_classes: list of all possible class labels.
    :param title: title for the ROC plot.
    :param save_path: path to save the generated ROC plot.
    :returns: macro-average AUC score across all present classes.
    """
    y_proba = model.predict_proba(X)
    present_classes = [c for c in all_classes if c in y_true]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fig, ax = plt.subplots(figsize=(10, 8))

    for class_label in all_classes:
        if class_label in present_classes:
            class_idx = list(model.classes_).index(class_label)
            fpr[class_label], tpr[class_label], _ = roc_curve(y_true == class_label, y_proba[:, class_idx])
            roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
            label = f"ROC Class {class_label} (AUC = {roc_auc[class_label]:.2f})"
            ax.plot(fpr[class_label], tpr[class_label], lw=2, label=label)
    valid_aucs = [roc_auc[k] for k in present_classes if k in roc_auc]
    macro_avg_roc_auc = np.mean(valid_aucs) if valid_aucs else np.nan
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} (Macro Avg AUC = {macro_avg_roc_auc:.2f})")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"ROC curve plot saved to: {save_path}")
    return macro_avg_roc_auc


def _convert_to_json_serializable(item: Any) -> Any:
    """Recursively converts items for JSON compatibility."""
    if isinstance(item, dict):
        return {k: _convert_to_json_serializable(v) for k, v in item.items()}
    if isinstance(item, list):
        return [_convert_to_json_serializable(i) for i in item]
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(
        item,
        (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64),
    ):
        return int(item)
    if isinstance(item, (np.float64, np.float16, np.float32, np.float64)):
        return float(item)
    if isinstance(item, (np.bool_)):
        return bool(item)
    return item


def randomforest() -> None:
    """Main function to run the final optimized RandomForest experiments."""
    logger.info("Starting Final Optimized RF experiments. " f"Results will be saved to: {RESULTS_DIR}")

    overall_results = {}
    master_label_set: Set[int] = set()

    for config_key in CONFIGS_TO_RUN:
        logger.info(f"\n--- Processing Fusion Configuration: {config_key} ---")

        X_train, y_train, X_val, y_val, X_test, y_test, _, error = load_data(config_key, SPLITS_DIR)
        if error or X_train is None or y_train is None:
            logger.error("Error while loading the data. Check directories.")
            continue

        assert X_train is not None and y_train is not None
        assert X_val is not None and y_val is not None
        assert X_test is not None and y_test is not None

        X_full_train, y_full_train = shuffle(
            np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)), random_state=42
        )
        master_label_set.update(y_full_train)
        if y_test is not None:
            master_label_set.update(y_test)
        sorted_labels = sorted(list(master_label_set))

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, bootstrap=True)),
            ]
        )

        param_dist = {
            "rf__n_estimators": [100, 200, 300, 400],
            "rf__max_depth": [5, 8, 10, 12],
            "rf__min_samples_split": [40, 60, 80, 100],
            "rf__min_samples_leaf": [20, 30, 40],
            "rf__max_features": ["sqrt", "log2"],
            "rf__max_samples": [0.7, 0.8, 0.9],
            "rf__ccp_alpha": [0.0, 0.001, 0.005, 0.01],
        }
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=30,
            cv=stratified_kfold,
            scoring=make_scorer(f1_score, average="macro", zero_division=0),
            refit=True,
            n_jobs=-1,
            random_state=42,
        )

        logger.info("Starting hyperparameter search...")
        random_search.fit(X_full_train, y_full_train)
        best_model = random_search.best_estimator_

        logger.info(f"Best params found: {random_search.best_params_}")
        logger.info(f"Best CV Macro F1 score: {random_search.best_score_:.4f}")

        # evaluate on test
        logger.info(f"Evaluating best model for '{config_key}' on the TEST set")
        y_pred_test = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
        test_report_str = classification_report(y_test, y_pred_test, labels=sorted_labels, zero_division=0)
        test_roc_auc = calculate_and_plot_roc_auc(
            best_model,
            X_test,
            y_test,
            sorted_labels,
            f"ROC Curves - {config_key} (Test)",
            RESULTS_DIR / f"roc_test_{config_key}.png",
        )
        logger.info(f"\n--- TEST SET METRICS for {config_key} ---")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Macro F1-Score: {test_f1_macro:.4f}")
        logger.info(f"Test ROC AUC: {test_roc_auc:.4f}")
        logger.info(f"Test Classification Report:\n{test_report_str}")

        # evaluate on train
        logger.info(f"Evaluating best model for '{config_key}' on the TRAINING set")
        y_pred_train = best_model.predict(X_full_train)
        train_accuracy = accuracy_score(y_full_train, y_pred_train)
        train_f1_macro = f1_score(y_full_train, y_pred_train, average="macro", zero_division=0)
        train_report_str = classification_report(y_full_train, y_pred_train, labels=sorted_labels, zero_division=0)
        train_roc_auc = calculate_and_plot_roc_auc(
            best_model,
            X_full_train,
            y_full_train,
            sorted_labels,
            f"ROC Curves - {config_key} (Train)",
            RESULTS_DIR / f"roc_train_{config_key}.png",
        )
        logger.info(f"\n--- TRAINING SET METRICS for {config_key} ---")
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Train Macro F1-Score: {train_f1_macro:.4f}")
        logger.info(f"Train ROC AUC: {train_roc_auc:.4f}")
        logger.info(f"Train Classification Report:\n{train_report_str}")

        # save the model
        model_path = RESULTS_DIR / f"model_{config_key}.joblib"
        joblib.dump(best_model, model_path)
        logger.info(f"Best model for '{config_key}' saved to: {model_path}")

        metrics_to_save = {
            "best_cv_macro_f1": random_search.best_score_,
            "best_hyperparameters": random_search.best_params_,
            "training_set_metrics": {
                "accuracy": train_accuracy,
                "macro_f1": train_f1_macro,
                "macro_roc_auc_ovr": train_roc_auc,
            },
            "training_set_classification_report": train_report_str,
            "test_set_metrics": {
                "accuracy": test_accuracy,
                "macro_f1": test_f1_macro,
                "macro_roc_auc_ovr": test_roc_auc,
            },
            "test_set_classification_report": test_report_str,
        }

        # save the metrics
        metrics_path = RESULTS_DIR / f"metrics_{config_key}.json"
        with open(metrics_path, "w") as f:
            json.dump(_convert_to_json_serializable(metrics_to_save), f, indent=4)
        logger.info(f"Final metrics for '{config_key}' saved to: {metrics_path}")

        # generate plots
        display_labels = [f"Class {lab}" for lab in sorted_labels]
        cm_test = confusion_matrix(y_test, y_pred_test, labels=sorted_labels)
        plot_confusion_matrix(
            cm_test, display_labels, f"Test CM - {config_key}", RESULTS_DIR / f"cm_test_{config_key}.png"
        )
        cm_train = confusion_matrix(y_full_train, y_pred_train, labels=sorted_labels)
        plot_confusion_matrix(
            cm_train, display_labels, f"Train CM - {config_key}", RESULTS_DIR / f"cm_train_{config_key}.png"
        )

        plot_learning_curve(
            best_model, f"Final RF LC - {config_key}", X_full_train, y_full_train, RESULTS_DIR / f"lc_{config_key}.png"
        )

        # add to overall results
        overall_results[config_key] = {
            "Train_Accuracy": train_accuracy,
            "Test_Accuracy": test_accuracy,
            "Train_Macro_F1": train_f1_macro,
            "Test_Macro_F1": test_f1_macro,
        }

    # summary of results
    logger.info("\n--- Overall Summary of RF Model Runs ---")
    try:
        import pandas as pd

        results_df = pd.DataFrame.from_dict(overall_results, orient="index")
        logger.info(f"\n{results_df.to_markdown()}")
        results_df.to_csv(RESULTS_DIR / "overall_summary.csv")
    except ImportError:
        logger.warning("Pandas is not installed. To see the summary table," " run: pip install pandas")
        logger.info(f"Raw results dictionary: {json.dumps(overall_results, indent=2)}")


if __name__ == "__main__":
    logger.info("Starting main execution...")
    try:
        randomforest()
        logger.info("Script finished successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
