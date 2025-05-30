"""Module for the entire pipeline endpoints."""
import os
import tempfile
import re
from typing import Dict, List, Optional, Any, Tuple
import joblib
import numpy as np
import mne
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, status
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.schemas import (
    ResponseMessage,
    # PreprocessingOutput,
    # UploadResponse,
    # PredictionInput,
    # PredictionOutput,
    PredictEDFResponse
)

from src.features.feature_engineering import FeatureEngineering
from src.utils.logger import get_logger

from src.data.preprocessing import (
    bandpass_filter,
    notch_filter,
    ANNOTATION_MAP,
    EVENT_ID_MAP,
    TARGET_SFREQ,
    EPOCH_DURATION,
    NOTCH_FREQ,
    EEG_BANDPASS,
    EOG_BANDPASS,
    EMG_BANDPASS
)

logger = get_logger("pipeline")

router = APIRouter(prefix="/pipeline", tags=["ML Pipeline"])

MODELS_DIR = "results"
AVAILABLE_CONFIGS = ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]
DEFAULT_CONFIG = "eeg_emg_eog"


def preprocess_edf_for_api(
        file_path: str,
        hypno_path: Optional[str] = None
) -> Tuple[mne.Epochs, bool]:
    logger.info(f"Processing EDF file for API: {file_path}")

    exclude_ch = ["Event marker", "Marker", "Status"]
    with mne.utils.use_log_level("WARNING"):
        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            exclude=exclude_ch,
            infer_types=True,
        )

    logger.info(
        f"Data loaded. SFreq: {raw.info['sfreq']:.2f} Hz. "
        f"Channels: {len(raw.ch_names)}"
    )

    eog_channel_name = "horizontal"
    if eog_channel_name in raw.ch_names:
        try:
            current_type = raw.get_channel_types(picks=[eog_channel_name])[0]
            if current_type != "eog":
                raw.set_channel_types({eog_channel_name: "eog"})
                logger.info(f"Set '{eog_channel_name}' channel type to 'eog'")
        except Exception as e:
            logger.warning(f"Could not set channel type: {e}")

    annotations_loaded = False
    if hypno_path and os.path.exists(hypno_path):
        try:
            temp_annots = mne.read_annotations(hypno_path, verbose=False)
            raw.set_annotations(temp_annots, emit_warning=False)
            annotations_loaded = True
            logger.info(f"Annotations loaded from {hypno_path}")
        except Exception as e:
            logger.warning(f"Could not load annotations: {e}")

    current_sfreq = raw.info["sfreq"]
    if current_sfreq != TARGET_SFREQ:
        logger.info(
            f"Resampling from {current_sfreq:.2f} Hz to {TARGET_SFREQ:.2f} Hz"
        )
        raw.resample(sfreq=TARGET_SFREQ, npad="auto", verbose=False)

    logger.info("Applying filters using existing preprocessing functions...")
    raw = bandpass_filter(raw, EEG_BANDPASS[0], EEG_BANDPASS[1], "eeg")
    raw = bandpass_filter(raw, EOG_BANDPASS[0], EOG_BANDPASS[1], "eog")
    raw = bandpass_filter(raw, EMG_BANDPASS[0], EMG_BANDPASS[1], "emg")

    nyquist = TARGET_SFREQ / 2.0
    if NOTCH_FREQ < nyquist:
        logger.info(
            f"Applying {NOTCH_FREQ} Hz notch filter using existing function"
        )
        if NOTCH_FREQ > EEG_BANDPASS[0] and NOTCH_FREQ < EEG_BANDPASS[1]:
            raw = notch_filter(raw, NOTCH_FREQ, "eeg")
        if NOTCH_FREQ > EOG_BANDPASS[0] and NOTCH_FREQ < EOG_BANDPASS[1]:
            raw = notch_filter(raw, NOTCH_FREQ, "eog")
        if NOTCH_FREQ > EMG_BANDPASS[0] and NOTCH_FREQ < EMG_BANDPASS[1]:
            raw = notch_filter(raw, NOTCH_FREQ, "emg")

    if annotations_loaded:
        logger.info("Creating epochs from annotations...")
        try:
            events, _ = mne.events_from_annotations(
                raw,
                event_id=ANNOTATION_MAP,
                chunk_duration=EPOCH_DURATION,
                verbose=False,
            )

            if events.shape[0] > 0:
                present_ids = np.unique(events[:, 2])
                epochs_event_id = {
                    stage_name: stage_id
                    for stage_name, stage_id in EVENT_ID_MAP.items()
                    if stage_id in present_ids
                }

                if epochs_event_id:
                    epochs = mne.Epochs(
                        raw,
                        events=events,
                        event_id=epochs_event_id,
                        tmin=0.0,
                        tmax=EPOCH_DURATION - 1 / raw.info["sfreq"],
                        preload=True,
                        baseline=None,
                        reject_by_annotation=True,
                        verbose=False,
                    )
                    logger.info(
                        f"Created {len(epochs)} labeled "
                        "epochs from annotations"
                    )
                else:
                    logger.warning(
                        "No valid event mapping found, falling back"
                        " to fixed-length epochs"
                    )
                    annotations_loaded = False
            else:
                logger.warning(
                    "No events found, falling back to fixed-length epochs"
                )
                annotations_loaded = False
        except Exception as e:
            logger.error(f"Error creating epochs from annotations: {e}")
            annotations_loaded = False

    if not annotations_loaded:
        logger.info("Creating fixed-length epochs...")
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=EPOCH_DURATION,
            preload=True,
            overlap=0.0,
            verbose=False,
        )
        logger.info(f"Created {len(epochs)} unlabeled epochs")

    return epochs, annotations_loaded


def extract_features_for_config(
        epochs: mne.Epochs, config: str
) -> Tuple[np.ndarray, List[str]]:
    modality_mapping = {
        "eeg": ["eeg"],
        "eeg_emg": ["eeg", "emg"],
        "eeg_eog": ["eeg", "eog"],
        "eeg_emg_eog": ["eeg", "emg", "eog"]
    }

    if config not in modality_mapping:
        raise ValueError(f"Unknown configuration: {config}")

    modalities_to_include = modality_mapping[config]

    all_ch_names = epochs.info.ch_names
    all_ch_types = epochs.info.get_channel_types(unique=False, picks="all")
    sfreq = epochs.info["sfreq"]

    selected_ch_indices = [
        idx for idx, ch_type in enumerate(all_ch_types)
        if ch_type in modalities_to_include
    ]

    if not selected_ch_indices:
        raise ValueError(f"No channels found for configuration '{config}'")

    selected_ch_names = [all_ch_names[i] for i in selected_ch_indices]
    modality_ch_types = [all_ch_types[i] for i in selected_ch_indices]

    modality_info = mne.create_info(
        ch_names=selected_ch_names,
        sfreq=sfreq,
        ch_types=modality_ch_types
    )

    epochs_data = epochs.get_data()[:, selected_ch_indices, :]

    if hasattr(epochs, 'events') and epochs.events is not None:
        events = epochs.events
    else:
        events = np.column_stack([
            np.arange(len(epochs_data)),
            np.zeros(len(epochs_data), dtype=int),
            np.zeros(len(epochs_data), dtype=int)
        ])

    epochs_for_extraction = mne.EpochsArray(
        epochs_data,
        info=modality_info,
        events=events,
        tmin=0.0,
        baseline=None,
        verbose=False
    )

    feature_engineer = FeatureEngineering()
    X_features, _, feature_names = feature_engineer._extract_features(
        epochs_for_extraction
    )

    return X_features, feature_names


def load_model(config: str) -> SklearnPipeline:
    model_path = os.path.join(MODELS_DIR, f"svm_model_{config}.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model for configuration: {config}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def evaluate_model_on_data(model, X_test, y_test, config: str,
                           request: Request = None) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, roc_auc_score
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, average=None, zero_division=0
    )
    macro_f1 = f1.mean()
    macro_precision = precision.mean()
    macro_recall = recall.mean()

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test, predictions, average='weighted', zero_division=0
        )
    )

    unique_labels = np.unique(np.concatenate([y_test, predictions]))

    if request and hasattr(request.app.state, 'label_mapping'):
        label_mapping = request.app.state.label_mapping
        class_names = [
            label_mapping.get(label,
                              f"Class_{label}") for label in unique_labels]
    else:
        class_names = [f"Class_{label}" for label in unique_labels]

    per_class_metrics = {}
    for i, (label, name) in enumerate(zip(unique_labels, class_names)):
        if i < len(precision):
            per_class_metrics[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }

    cm = confusion_matrix(y_test, predictions, labels=unique_labels)

    roc_auc = None
    if hasattr(model, 'predict_proba') and len(unique_labels) > 2:
        try:
            probabilities = model.predict_proba(X_test)
            roc_auc = roc_auc_score(
                y_test, probabilities,
                multi_class="ovr", average="macro",
                labels=unique_labels
            )
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")

    unique_test, counts_test = np.unique(y_test, return_counts=True)
    test_distribution = {}
    for label, count in zip(unique_test, counts_test):
        if request and hasattr(request.app.state, 'label_mapping'):
            label_name = request.app.state.label_mapping.get(
                label, f"Class_{label}"
            )
        else:
            label_name = f"Class_{label}"
        test_distribution[label_name] = int(count)

    return {
        "dataset_size": len(y_test),
        "overall_metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1_score": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1_score": float(weighted_f1),
            "roc_auc_macro": float(roc_auc) if roc_auc else None
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": class_names
        },
        "class_distribution": test_distribution
    }


def load_pretrained_metrics(config: str) -> Dict[str, Any]:
    """
    Load metrics from SVM training log files instead of .npz files.
    Parses the log to extract training, validation, and test metrics.
    """
    # change if needed
    possible_log_dirs = [
        "logs",
        "../logs",
        "../../logs",
        "/scratch/s5130727/Applied-ML/logs",
        "/Users/jananjahed/Desktop/ML_applied/Applied-ML/logs"
    ]

    log_file_path = None

    for log_dir in possible_log_dirs:
        if os.path.exists(log_dir):
            try:
                log_files = [f for f in os.listdir(log_dir) if f.startswith("svm_train_") and f.endswith(".err")]
                if log_files:
                    log_files.sort(reverse=True)
                    log_file_path = os.path.join(log_dir, log_files[0])
                    break
            except Exception as e:
                logger.warning(f"Could not access log directory {log_dir}: {e}")
                continue

    if not log_file_path or not os.path.exists(log_file_path):
        logger.warning(f"No SVM training log file found for config '{config}'")
        return {
            "training_metrics": None,
            "validation_metrics": None,
            "test_metrics": None
        }

    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()

        logger.info(f"Found and reading log file: {log_file_path}")
        metrics = parse_metrics_from_log(log_content, config)
        return metrics

    except Exception as e:
        logger.error(f"Error reading log file {log_file_path}: {e}")
        return {
            "training_metrics": None,
            "validation_metrics": None,
            "test_metrics": None
        }


def parse_metrics_from_log(log_content: str, config: str) -> Dict[str, Any]:
    """
    Parse metrics from the SVM training log content for a specific configuration.
    """
    config_sections = log_content.split("--- Processing Fusion Configuration")

    target_section = None
    for section in config_sections:
        if f": {config} ---" in section:
            target_section = section
            break

    if not target_section:
        logger.warning(f"Configuration '{config}' not found in log")
        return {
            "training_metrics": None,
            "validation_metrics": None,
            "test_metrics": None
        }

    training_metrics = extract_dataset_metrics(target_section, "Training", config)
    validation_metrics = extract_dataset_metrics(target_section, "Validation", config)
    test_metrics = extract_dataset_metrics(target_section, "Test", config)

    return {
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics
    }


def extract_dataset_metrics(section: str, dataset_type: str, config: str) -> Optional[Dict[str, Any]]:
    """
    Extract metrics for a specific dataset type (Training/Validation/Test) from log section.
    """
    try:
        # Find the performance section for this dataset type
        pattern = rf"--- Evaluating on {dataset_type} set for config '{config}' ---.*?(?=--- Evaluating on|--- Processing|$)"
        match = re.search(pattern, section, re.DOTALL)

        if not match:
            return None

        dataset_section = match.group(0)

        accuracy_match = re.search(r"Accuracy: ([\d.]+)", dataset_section)
        f1_match = re.search(r"Macro F1: ([\d.]+)", dataset_section)
        roc_auc_match = re.search(r"ROC AUC \(OVR Macro\): ([\d.]+|N/A)", dataset_section)

        if not (accuracy_match and f1_match):
            return None

        accuracy = float(accuracy_match.group(1))
        macro_f1 = float(f1_match.group(1))
        roc_auc = None if not roc_auc_match or roc_auc_match.group(1) == "N/A" else float(roc_auc_match.group(1))

        report_pattern = r"precision\s+recall\s+f1-score\s+support(.*?)accuracy"
        report_match = re.search(report_pattern, dataset_section, re.DOTALL)

        per_class_metrics = {}
        dataset_size = 0

        if report_match:
            report_lines = report_match.group(1).strip().split('\n')

            for line in report_lines:
                line = line.strip()
                if line.startswith('Class'):
                    parts = line.split()
                    if len(parts) >= 5:
                        class_name = parts[0] + " " + parts[1]  # "Class 0", "Class 1", etc.
                        precision = float(parts[2])
                        recall = float(parts[3])
                        f1_score = float(parts[4])
                        support = int(parts[5])

                        per_class_metrics[class_name] = {
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1_score,
                            "support": support
                        }
                        dataset_size += support

        cm_pattern = r"Confusion Matrix \(" + dataset_type + rf" - Config: {config}\):\s*(\[.*?\])"
        cm_match = re.search(cm_pattern, dataset_section, re.DOTALL)

        confusion_matrix = None
        if cm_match:
            try:
                cm_text = cm_match.group(1)
                cm_lines = []
                for line in cm_text.split('\n'):
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            cm_lines.append([int(n) for n in numbers])

                if cm_lines:
                    confusion_matrix = {
                        "matrix": cm_lines,
                        "labels": list(per_class_metrics.keys())
                    }
            except Exception as e:
                logger.warning(f"Could not parse confusion matrix: {e}")

        if per_class_metrics:
            precisions = [m["precision"] for m in per_class_metrics.values()]
            recalls = [m["recall"] for m in per_class_metrics.values()]
            f1_scores = [m["f1_score"] for m in per_class_metrics.values()]
            supports = [m["support"] for m in per_class_metrics.values()]

            total_support = sum(supports)
            weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support if total_support > 0 else 0
            weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support if total_support > 0 else 0
            weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_support if total_support > 0 else 0
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0

        return {
            "dataset_size": dataset_size,
            "overall_metrics": {
                "accuracy": accuracy,
                "macro_precision": sum(precisions) / len(precisions) if precisions else 0,
                "macro_recall": sum(recalls) / len(recalls) if recalls else 0,
                "macro_f1_score": macro_f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1_score": weighted_f1,
                "roc_auc_macro": roc_auc
            },
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion_matrix,
            "class_distribution": {name: metrics["support"] for name, metrics in per_class_metrics.items()},
            "data_source": f"Parsed from SVM training log - {dataset_type} set"
        }

    except Exception as e:
        logger.error(f"Error extracting {dataset_type} metrics for {config}: {e}")
        return None


@router.post("/predict-edf", response_model=PredictEDFResponse)
async def predict_edf_file(
    edf_file: UploadFile = File(..., description="EDF sleep recording file"),
    hypno_file: Optional[UploadFile] = File(
        None, description="Optional hypnogram file for better epoch creation"
    ),
    config: str = DEFAULT_CONFIG,
    request: Request = None
):
    """
    Complete Automated Sleep Stage Prediction Pipeline

    Model Performance (Validation Set):
    - Accuracy: ~60% vs 20% random guessing (5-class problem)
    - Tested on 10 subjects
    - Significantly outperforms random classification

    Just upload your EDF file and get sleep stage predictions!

    This endpoint automatically:
    1. Uploads and validates your EDF file
    2. Preprocesses using proven clinical pipeline
    3. Extracts sleep-relevant features from EEG/EOG/EMG
    4. Predicts sleep stages using trained SVM models
    5. Returns detailed predictions with confidence scores

    Input: EDF file (+ optional hypnogram)
    Output: Sleep stage predictions for every 30-second epoch

    Supported configurations:
    - `eeg`: EEG channels only
    - `eeg_emg`: EEG + EMG (good for REM detection)
    - `eeg_eog`: EEG + EOG (good for eye movement artifacts)
    - `eeg_emg_eog`: All channels (most comprehensive, default)
    """
    if config not in AVAILABLE_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid configuration '{config}'. "
                + f"Available: {AVAILABLE_CONFIGS}"
            )
        )

    if not edf_file.filename or not edf_file.filename.lower().endswith('.edf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an EDF file with .edf extension"
        )

    file_size_mb = edf_file.size / (1024 * 1024) if edf_file.size else 0
    if file_size_mb > 500:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large ({file_size_mb:.1f}MB). "
                + "Maximum size is 500MB."
            )
        )

    logger.info(
        f"Starting automated prediction pipeline for {edf_file.filename}"
        f" using {config} configuration"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        edf_path = os.path.join(temp_dir, edf_file.filename)
        hypno_path = None

        logger.info("Saving uploaded files...")
        with open(edf_path, "wb") as f:
            content = await edf_file.read()
            f.write(content)

        if hypno_file and hypno_file.filename:
            hypno_path = os.path.join(temp_dir, hypno_file.filename)
            with open(hypno_path, "wb") as f:
                content = await hypno_file.read()
                f.write(content)
            logger.info(f"Hypnogram file provided: {hypno_file.filename}")

        try:
            logger.info(
                "Step 1/4: Preprocessing EDF data using existing pipeline..."
            )
            epochs, annotations_loaded = preprocess_edf_for_api(
                edf_path, hypno_path
            )
            logger.info(
                f"Preprocessing complete. Created {len(epochs)} epochs"
                f" of {EPOCH_DURATION}s each"
            )

            logger.info(
                f"Step 2/4: Extracting features for {config} configuration..."
            )
            features, feature_names = extract_features_for_config(
                epochs, config
            )
            logger.info(
                f"Feature extraction complete. Extracted "
                f" {features.shape[1]} features per epoch"
            )

            logger.info(
                "Step 3/4: Loading trained model and making predictions..."
            )
            model = load_model(config)
            predictions = model.predict(features)
            logger.info(f"Predictions complete for {len(predictions)} epochs")

            logger.info(
                "Step 4/4: Generating confidence scores"
                " and formatting results..."
            )
            probabilities_per_segment = None
            if hasattr(model, 'predict_proba'):
                probabilities_per_segment = model.predict_proba(
                    features).tolist()

            if request and hasattr(request.app.state, 'label_mapping'):
                label_mapping = request.app.state.label_mapping
                prediction_labels = [
                    label_mapping.get(pred, f"Unknown_{pred}")
                    for pred in predictions
                ]
                class_labels_legend = label_mapping
            else:
                prediction_labels = [f"Class_{pred}" for pred in predictions]
                class_labels_legend = None

            unique_stages, counts = np.unique(
                prediction_labels, return_counts=True
            )
            stage_distribution = dict(zip(unique_stages, counts.tolist()))
            total_time_hours = len(predictions) * EPOCH_DURATION / 3600

            current_file_metrics = None
            if annotations_loaded and hasattr(epochs, 'events') and \
                    epochs.events is not None:
                logger.info(
                    "Ground truth available - calculating performance"
                    " metrics on current file..."
                )
                try:
                    ground_truth = epochs.events[:, -1]
                    current_file_metrics = evaluate_model_on_data(
                        model, features, ground_truth, config, request
                    )
                    current_file_metrics["note"] = (
                        "Performance metrics calculated on the uploaded"
                        + " file with ground truth annotations"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not calculate metrics on current file: {e}"
                    )

            logger.info("Complete pipeline finished successfully!")
            logger.info(f"Sleep stage distribution: {stage_distribution}")
            logger.info(f"Total recording time: {total_time_hours:.1f} hours")

            return PredictEDFResponse(
                model_configuration_used=config,
                input_file_name=edf_file.filename,
                num_segments_processed=len(predictions),
                predictions=prediction_labels,
                prediction_ids=predictions.tolist(),
                probabilities_per_segment=probabilities_per_segment,
                class_labels_legend=class_labels_legend,
                processing_summary={
                    "total_recording_time_hours": round(total_time_hours, 2),
                    "epoch_duration_seconds": EPOCH_DURATION,
                    "annotations_from_hypnogram": annotations_loaded,
                    "features_extracted_per_epoch": features.shape[1],
                    "sleep_stage_distribution": stage_distribution,
                    "current_file_performance": current_file_metrics
                }
            )

        except Exception as e:
            logger.error(f"Error in complete EDF prediction pipeline: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error in prediction pipeline: {str(e)}"
            )


@router.get("/all-performance/{config}", response_model=Dict[str, Any])
async def get_all_performance_metrics(config: str, request: Request = None):
    """
    Get Complete Performance Analysis

    Returns ALL performance metrics for a model configuration:
    - Training metrics (how well model fit training data)
    - Validation metrics (hyperparameter tuning performance)
    - Test metrics (final unseen data performance)
    - Model comparison and overfitting analysis

    Model Performance Evidence:
    - Validation Accuracy: ~70% vs 20% random guessing
    - Test Accuracy: ~60%
    """
    if config not in AVAILABLE_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid configuration '{config}'."
                + f" Available: {AVAILABLE_CONFIGS}"
            )
        )

    model_path = os.path.join(MODELS_DIR, f"svm_model_{config}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found for configuration '{config}'"
        )

    try:
        pretrained_metrics = load_pretrained_metrics(config)

        overfitting_analysis = {}
        if pretrained_metrics["training_metrics"] and \
                pretrained_metrics["test_metrics"]:

            train_acc = pretrained_metrics["training_metrics"][
                "overall_metrics"]["accuracy"]
            test_acc = pretrained_metrics["test_metrics"]["overall_metrics"][
                "accuracy"
            ]
            train_f1 = pretrained_metrics["training_metrics"][
                "overall_metrics"]["macro_f1_score"]
            test_f1 = pretrained_metrics["test_metrics"]["overall_metrics"][
                "macro_f1_score"
            ]

            overfitting_analysis = {
                "accuracy_drop": round(train_acc - test_acc, 4),
                "f1_drop": round(train_f1 - test_f1, 4),
                "overfitting_severity": (
                    "Low" if (train_acc - test_acc) < 0.05
                    else "Moderate" if (train_acc - test_acc) < 0.15
                    else "High"
                ),
                "generalization_quality": (
                    "Excellent" if (train_acc - test_acc) < 0.03
                    else "Good" if (train_acc - test_acc) < 0.08
                    else "Fair" if (train_acc - test_acc) < 0.15
                    else "Poor"
                ),
                "vs_random_guessing": {
                    "random_accuracy": 0.20,
                    "test_accuracy": test_acc,
                    "improvement_over_random": round(
                        (test_acc - 0.20) / 0.20 * 100, 1
                    ),
                    "significantly_above_random": test_acc > 0.35
                }
            }

        summary = {}
        if pretrained_metrics["training_metrics"]:
            summary["training"] = {
                "accuracy": (
                    pretrained_metrics["training_metrics"]["overall_metrics"][
                        "accuracy"
                    ]
                ),
                "f1_score": (
                    pretrained_metrics["training_metrics"]["overall_metrics"][
                        "macro_f1_score"
                    ]
                ),
                "dataset_size": (
                    pretrained_metrics["training_metrics"]["dataset_size"]
                )
            }

        if pretrained_metrics["validation_metrics"]:
            summary["validation"] = {
                "accuracy": (
                    pretrained_metrics["validation_metrics"][
                        "overall_metrics"]["accuracy"]
                ),
                "f1_score": (
                    pretrained_metrics["validation_metrics"][
                        "overall_metrics"]["macro_f1_score"]
                ),
                "dataset_size": (
                    pretrained_metrics["validation_metrics"]["dataset_size"]
                )
            }

        if pretrained_metrics["test_metrics"]:
            summary["test"] = {
                "accuracy": (
                    pretrained_metrics["test_metrics"]["overall_metrics"][
                        "accuracy"
                    ]
                ),
                "f1_score": (
                    pretrained_metrics["test_metrics"]["overall_metrics"][
                        "macro_f1_score"
                    ]
                ),
                "dataset_size": (
                    pretrained_metrics["test_metrics"]["dataset_size"]
                )
            }

        return {
            "model_configuration": config,
            "performance_summary": summary,
            "overfitting_analysis": overfitting_analysis,
            "detailed_metrics": {
                "training": pretrained_metrics["training_metrics"],
                "validation": pretrained_metrics["validation_metrics"],
                "test": pretrained_metrics["test_metrics"]
            },
            "model_status": {
                "training_data_available": (
                    pretrained_metrics["training_metrics"] is not None
                ),
                "validation_data_available": (
                    pretrained_metrics["validation_metrics"] is not None
                ),
                "test_data_available": (
                    pretrained_metrics["test_metrics"] is not None
                ),
                "performance_analysis_complete": all([
                    pretrained_metrics["training_metrics"],
                    pretrained_metrics["test_metrics"]
                ])
            },
            "recommendations": {
                "model_quality": (
                    "Production ready"
                    if overfitting_analysis.get("generalization_quality")
                    in ["Excellent", "Good"]
                    else "Needs improvement"
                ),
                "above_random_performance": (
                    overfitting_analysis.get("vs_random_guessing", {})
                    .get("significantly_above_random", False)
                ),
                "next_steps": [
                    ("Model performs well on unseen data"
                     if overfitting_analysis.get("overfitting_severity")
                     == "Low"
                     else (
                        "Consider regularization or more training data"
                     )
                     ),
                    ("Ready for deployment"
                     if summary.get("test", {}).get("accuracy", 0) > 0.8
                     else "Consider model improvements"
                     )
                ]
            }
        }

    except Exception as e:
        logger.error(f"Error loading performance metrics for {config}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading performance metrics: {str(e)}"
        )


@router.get("/available-models", response_model=Dict[str, Any])
async def get_available_models():
    available_models = {}

    for config in AVAILABLE_CONFIGS:
        model_path = os.path.join(MODELS_DIR, f"svm_model_{config}.joblib")
        available_models[config] = {
            "available": os.path.exists(model_path),
            "path": model_path
        }

    return {
        "available_configurations": available_models,
        "default_configuration": DEFAULT_CONFIG
    }


@router.get("/health", response_model=ResponseMessage)
async def pipeline_health_check():
    available_count = 0
    available_models = []
    for config in AVAILABLE_CONFIGS:
        model_path = os.path.join(MODELS_DIR, f"svm_model_{config}.joblib")
        if os.path.exists(model_path):
            available_count += 1
            available_models.append(config)

    if available_count == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained models available. Please ensure model files are"
                + " in the 'results/' directory."
            )
        )

    return ResponseMessage(
        message=(
            f"ML Pipeline is healthy! {available_count}/"
            f"{len(AVAILABLE_CONFIGS)} models ready: {available_models}"
        )
    )


@router.post("/test-prediction", response_model=Dict[str, Any])
async def test_model_prediction(
    config: str = DEFAULT_CONFIG,
    request: Request = None
):
    """
    Test the ML pipeline with synthetic data

    This endpoint lets you test if a model configuration works without
    uploading real EDF files.
    It generates synthetic features and returns a prediction to verify the
    model is working.

    Useful for:
    - Testing if models are loaded correctly
    - Checking model configurations
    - API debugging and development
    """
    if config not in AVAILABLE_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid configuration '{config}'. "
                f"Available: {AVAILABLE_CONFIGS}"
            )
        )

    try:
        model = load_model(config)

        feature_counts = {
            "eeg": 16,
            "eeg_emg": 30,
            "eeg_eog": 35,
            "eeg_emg_eog": 19
        }

        n_features = feature_counts.get(config, 30)
        synthetic_features = np.random.rand(1, n_features)

        prediction_id = model.predict(synthetic_features)[0]

        confidence_score = None
        class_probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(synthetic_features)[0]
            confidence_score = float(np.max(probabilities))

            if request and hasattr(request.app.state, 'label_mapping'):
                label_mapping = request.app.state.label_mapping
                class_probabilities = {
                    label_mapping.get(i, f"Class_{i}"): float(prob)
                    for i, prob in enumerate(probabilities)
                }

        if request and hasattr(request.app.state, 'label_mapping'):
            label_mapping = request.app.state.label_mapping
            prediction_label = label_mapping.get(
                prediction_id, f"Unknown_{prediction_id}"
            )
        else:
            prediction_label = f"Class_{prediction_id}"

        return {
            "status": "success",
            "message": f"Model '{config}' is working correctly!",
            "test_results": {
                "model_configuration": config,
                "synthetic_features_count": n_features,
                "predicted_sleep_stage": prediction_label,
                "prediction_id": int(prediction_id),
                "confidence_score": confidence_score,
                "class_probabilities": class_probabilities
            },
            "note": (
                "This was a test with synthetic data. Upload real EDF"
                + " files using /predict-edf for actual predictions."
            )
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Model file not found for configuration '{config}': {str(e)}"
            )
        )
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing model '{config}': {str(e)}"
        )
