import os
import tempfile
from typing import Dict, List, Optional, Any, Tuple
import joblib
import numpy as np
import mne
import json
from fastapi import APIRouter, HTTPException, Request, status 
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.features.feature_engineering import FeatureEngineering
from src.schemas.base import ResponseMessage
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
from src.schemas.model_schemas import ModelConfig, AvailableModels, ModelDetails, ModelPerformanceSummary 
from src.schemas.prediction_schemas import PredictEDFResponse
from src.endpoints.recordings import get_all_recordings

router = APIRouter(prefix="/models", tags=["Models"])

MODELS_DIR = "../results"
AVAILABLE_CONFIGS = ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]
DEFAULT_CONFIG = ModelConfig.EEG_EMG_EOG

logger = get_logger("models_endpoint")


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
            temp_annots = mne.read_annotations(hypno_path)
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
    model_path = os.path.join(MODELS_DIR, f"model_{config}.joblib")

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


def _parse_rf_classification_report(
        report_str: str) -> Tuple[Dict[str, Any], int]:
    """Helper to parse a string classification report.

    :param report_str: the classification report.
    :return: tuple containing the a dictionary of
             metrics per class and the total support.
    """
    lines = report_str.strip().split('\n')
    per_class_metrics = {}
    total_support = 0

    for line in lines:
        parts = line.split()
        if len(parts) > 2 and parts[0].isdigit():
            class_id = int(parts[0])
            class_name = f"Class {class_id}"

            if len(parts) >= 5:
                precision = float(parts[1])
                recall = float(parts[2])
                f1_score = float(parts[3])
                support = int(parts[4])

                per_class_metrics[class_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "support": support
                }
        elif "accuracy" in line and len(parts) >= 2 and parts[-1].isdigit():
            total_support = int(parts[-1])

    if total_support == 0:
        total_support = sum(m["support"] for m in per_class_metrics.values())

    return per_class_metrics, total_support


def load_pretrained_metrics(config: str) -> Dict[str, Any]:
    """
    Load metrics for the Random Forest model from the specified JSON file,
    including both training and test set metrics.
    """
    metrics_path = os.path.join(MODELS_DIR, f"metrics_{config}.json")

    if not os.path.exists(metrics_path):
        logger.warning(
            f"Metrics file not found for config '{config}' at {metrics_path}"
        )
        return {"training_metrics": None, "validation_metrics": None,
                "test_metrics": None}

    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        logger.info(f"Successfully loaded metrics from {metrics_path}")

        def process_metric_block(metric_key: str, report_key: str,
                                 data_source: str) -> Optional[Dict[str, Any]]:
            if metric_key not in metrics_data:
                return None

            metrics_raw = metrics_data[metric_key]
            report_str = metrics_data.get(report_key, "")
            per_class_metrics, dataset_size = _parse_rf_classification_report(
                report_str
            )

            (macro_precision, macro_recall, weighted_precision,
             weighted_recall, weighted_f1) = 0, 0, 0, 0, 0
            if per_class_metrics and dataset_size > 0:
                precisions = [
                    m["precision"] for m in per_class_metrics.values()
                ]
                recalls = [m["recall"] for m in per_class_metrics.values()]
                macro_precision = (
                    sum(precisions) / len(precisions) if precisions else 0
                )
                macro_recall = sum(recalls) / len(recalls) if recalls else 0
                for metrics in per_class_metrics.values():
                    weighted_precision += (
                        metrics["precision"] * metrics["support"]
                    )
                    weighted_recall += metrics["recall"] * metrics["support"]
                    weighted_f1 += metrics["f1_score"] * metrics["support"]
                weighted_precision /= dataset_size
                weighted_recall /= dataset_size
                weighted_f1 /= dataset_size

            return {
                "dataset_size": dataset_size,
                "overall_metrics": {
                    "accuracy": metrics_raw.get("accuracy"),
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1_score": metrics_raw.get("macro_f1"),
                    "weighted_precision": weighted_precision,
                    "weighted_recall": weighted_recall,
                    "weighted_f1_score": weighted_f1,
                    "roc_auc_macro": metrics_raw.get("macro_roc_auc_ovr"),
                },
                "per_class_metrics": per_class_metrics,
                "confusion_matrix": None,
                "class_distribution": {name: metrics["support"] for name, metrics in per_class_metrics.items()},
                "data_source": data_source
            }

        training_metrics = process_metric_block(
            "training_set_metrics",
            "training_set_classification_report",
            "Parsed from JSON - Training set"
        )
        test_metrics = process_metric_block(
            "test_set_metrics",
            "test_set_classification_report",
            "Parsed from JSON - Test set"
        )

        validation_metrics = {
             "dataset_size": None,
             "overall_metrics": {"macro_f1_score": metrics_data.get("best_cv_macro_f1")},
             "details": {"best_hyperparameters": metrics_data.get("best_hyperparameters")},
             "data_source": "From Cross-Validation (best_cv_macro_f1)"
        }

        return {
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics
        }

    except Exception as e:
        logger.error(
            f"Error reading or parsing metrics file {metrics_path}: {e}"
        )
        return {"training_metrics": None, "validation_metrics": None,
                "test_metrics": None}

def health_check():
    available_count = 0
    available_models = []
    for config in AVAILABLE_CONFIGS:
        model_path = os.path.join(MODELS_DIR, f"model_{config}.joblib")
        if os.path.exists(model_path):
            available_count += 1
            available_models.append(config)

    if available_count == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "⚠️ No trained models available. Please ensure model files are"
                + " in the 'results/' directory."
            )
        )

    return ResponseMessage(
        message=(
            f"✅ ML Pipeline is healthy! {available_count}/"
            f"{len(AVAILABLE_CONFIGS)} models ready: {available_models}"
        )
    )

@router.get(
    "/",
    summary="Get all available model configurations",
    description="Returns a dictionary of available models and their configurations.",
    response_model=AvailableModels,
)
def get_available_models():
    """Checks for all possible model configurations and returns their availability."""
    available_models = {}
    try:

        for config in ModelConfig:
            model_path = os.path.join(MODELS_DIR, f"model_{config.value}.joblib")
            available_models[config.value] = {
                "available": os.path.exists(model_path),
                "path": model_path,
            }
        return AvailableModels(
            available_configurations=available_models,
            default_configuration=DEFAULT_CONFIG,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"⚠️ Error checking model availability: {str(e)}"
        )

@router.get(
    "/{model_id}",
    summary="Get detailed metadata for a specific model",
    description="Provides key information about a model, including its expected inputs and a performance summary.",
    response_model=ModelDetails,
)
def get_model_details(model_id: ModelConfig, request: Request):
    """Retrieves detailed metadata for a single specified model configuration."""
    model_path = os.path.join(MODELS_DIR, f"model_{model_id.value}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found for configuration '{model_id.value}'",
        )

    modalities = model_id.value.split('_')
    
    feature_counts = {
        ModelConfig.EEG: 16,
        ModelConfig.EEG_EMG: 30,
        ModelConfig.EEG_EOG: 35,
        ModelConfig.EEG_EMG_EOG: 49,
    }

    metrics = load_pretrained_metrics(model_id.value)
    test_metrics_data = metrics.get("test_metrics")
    
    performance_summary = None
    if test_metrics_data:
        performance_summary = ModelPerformanceSummary(
            test_accuracy=test_metrics_data.get("overall_metrics", {}).get("accuracy"),
            test_macro_f1_score=test_metrics_data.get("overall_metrics", {}).get("macro_f1_score"),
        )

    class_labels = getattr(request.app.state, 'label_mapping', {})

    return ModelDetails(
        config_name=model_id,
        modalities_used=modalities,
        expected_features_count=feature_counts.get(model_id, 0),
        class_labels_legend=class_labels,
        performance_summary=performance_summary,
    )

@router.get("/{model_id}/performance", response_model=Dict[str, Any])
async def get_model_performance(model_id: ModelConfig):
    """
    Get Complete Performance Analysis for the Random Forest model.

    Returns ALL available performance metrics for a model configuration:
    - Training metrics (how well model fit training data)
    - Validation metrics (from cross-validation during training)
    - Test metrics (final unseen data performance)
    - Model comparison and overfitting analysis
    """
    config = model_id.value
    if config not in AVAILABLE_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid configuration '{config}'."
                + f"Available: {AVAILABLE_CONFIGS}")
        )

    model_path = os.path.join(MODELS_DIR, f"model_{config}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found for configuration '{config}'"
        )

    try:
        pretrained_metrics = load_pretrained_metrics(config)

        overfitting_analysis = {}
        train_metrics = pretrained_metrics.get("training_metrics")
        test_metrics = pretrained_metrics.get("test_metrics")

        if train_metrics and test_metrics:
            train_acc = train_metrics["overall_metrics"].get("accuracy", 0.0)
            test_acc = test_metrics["overall_metrics"].get("accuracy", 0.0)
            train_f1 = train_metrics["overall_metrics"].get("macro_f1_score", 0.0)
            test_f1 = test_metrics["overall_metrics"].get("macro_f1_score", 0.0)

            accuracy_drop = train_acc - test_acc
            f1_drop = train_f1 - test_f1

            overfitting_analysis = {
                "accuracy_dropoff": f"{accuracy_drop:.4f} (Train: {train_acc:.4f}, Test: {test_acc:.4f})",
                "f1_score_dropoff": f"{f1_drop:.4f} (Train: {train_f1:.4f}, Test: {test_f1:.4f})",
                "overfitting_severity": (
                    "Low" if accuracy_drop < 0.05 else
                    "Moderate" if accuracy_drop < 0.15 else
                    "High"
                ),
                "generalization_quality": (
                    "Excellent" if accuracy_drop < 0.03 else
                    "Good" if accuracy_drop < 0.08 else
                    "Fair" if accuracy_drop < 0.15 else
                    "Poor"
                ),
                "vs_random_guessing": {
                    "random_accuracy": 0.20,
                    "test_accuracy": test_acc,
                    "significantly_above_random": round((test_acc - 0.20) / 0.20 * 100, 1),
                }
            }

        summary = {}
        if train_metrics:
            summary["training"] = {
                "accuracy": train_metrics["overall_metrics"].get("accuracy"),
                "f1_score": train_metrics["overall_metrics"].get("macro_f1_score"),
                "dataset_size": train_metrics.get("dataset_size")
            }

        val_metrics_data = pretrained_metrics.get("validation_metrics")
        if val_metrics_data:
            summary["validation"] = {
                "cross_validation_f1_score": val_metrics_data.get("overall_metrics", {}).get("macro_f1_score"),
            }

        if test_metrics:
            summary["test"] = {
                "accuracy": test_metrics["overall_metrics"].get("accuracy"),
                "f1_score": test_metrics["overall_metrics"].get("macro_f1_score"),
                "dataset_size": test_metrics.get("dataset_size")
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
                "training_data_available": pretrained_metrics["training_metrics"] is not None,
                "validation_data_available": pretrained_metrics["validation_metrics"] is not None,
                "test_data_available": pretrained_metrics["test_metrics"] is not None,
                "performance_analysis_complete": all([
                    pretrained_metrics["training_metrics"],
                    pretrained_metrics["test_metrics"]
                ])
            },
            "recommendations": {
                "model_quality": (
                    "Production ready"
                    if summary.get("test", {}).get("accuracy", 0) > 0.70
                    else "Review test metrics, may need improvement"
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

@router.post("/{model_id}/predict/{recording_id}", response_model=PredictEDFResponse)
async def predict_from_recording(
    model_id: ModelConfig,
    recording_id: int,
    request: Request,
):
    """
    Complete Automated Sleep Stage Prediction Pipeline.
    This endpoint can be triggered by a direct file upload or internally with a server file path.
    Model Performance (Validation Set):
     - Accuracy: ~60% vs 20% random guessing (5-class problem)
     - Tested on 10 subjects
     - Significantly outperforms random classification

    Just upload or select your EDF file and get sleep stage predictions!

    This endpoint automatically:
     1. Uploads and validates your EDF file
     2. Preprocesses using proven clinical pipeline
     3. Extracts sleep-relevant features from EEG/EOG/EMG
     4. Predicts sleep stages using trained Random Forest models
     5. Returns detailed predictions with confidence scores

    Input: EDF file (+ optional hypnogram)
    Output: Sleep stage predictions for every 30-second epoch

    Supported configurations:
     - `eeg`: EEG channels only
     - `eeg_emg`: EEG + EMG (good for REM detection)
     - `eeg_eog`: EEG + EOG (good for eye movement artifacts)
     - `eeg_emg_eog`: All channels (most comprehensive, default)
    """
    config = model_id.value

    try:
        all_recordings = get_all_recordings()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing recordings: {str(e)}")

    all_recordings_flat = {**all_recordings["cassette_files"], **all_recordings["telemetry_files"]}
    if recording_id not in all_recordings_flat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

    recording = all_recordings_flat[recording_id]
    edf_path = recording.file_path
    hypno_path = recording.anno_path
    input_filename = ""
    temp_dir_manager = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_manager.name

    if not os.path.exists(edf_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"EDF file not found at path: {edf_path}"
        )

    logger.info(
            f"Starting automated prediction pipeline for {input_filename} using {config} configuration"
        )


    try:
        logger.info("Step 1/4: Preprocessing EDF data...")
        epochs, annotations_loaded = preprocess_edf_for_api(edf_path, hypno_path)
        logger.info(f"Preprocessing complete. Created {len(epochs)} epochs of {EPOCH_DURATION}s each.")

        logger.info(f"Step 2/4: Extracting features for {config} configuration...")
        features, feature_names = extract_features_for_config(epochs, config)
        logger.info(f"Feature extraction complete. Extracted {features.shape[1]} features per epoch.")

        logger.info("Step 3/4: Loading trained model and making predictions...")
        model = load_model(config)
        predictions = model.predict(features)
        logger.info(f"Predictions complete for {len(predictions)} epochs.")

        logger.info("Step 4/4: Generating confidence scores and formatting results...")
        probabilities_per_segment = None
        if hasattr(model, 'predict_proba'):
            probabilities_per_segment = model.predict_proba(features).tolist()

        if hasattr(request.app.state, 'label_mapping'):
            label_mapping = request.app.state.label_mapping
            prediction_labels = [label_mapping.get(pred, f"Unknown_{pred}") for pred in predictions]
            class_labels_legend = label_mapping
        else:
            prediction_labels = [f"Class_{pred}" for pred in predictions]
            class_labels_legend = None

        unique_stages, counts = np.unique(prediction_labels, return_counts=True)
        stage_distribution = dict(zip(unique_stages, counts.tolist()))
        total_time_hours = len(predictions) * EPOCH_DURATION / 3600

        current_file_metrics = None
        if annotations_loaded and hasattr(epochs, 'events') and epochs.events is not None:
            logger.info("Ground truth available - calculating performance metrics on current file...")
            try:
                ground_truth = epochs.events[:, -1]
                current_file_metrics = evaluate_model_on_data(model, features, ground_truth, config, request)
                current_file_metrics["note"] = "Performance metrics calculated on the uploaded file with ground truth annotations."
            except Exception as e:
                logger.warning(f"Could not calculate metrics on current file: {e}")

        logger.info("Complete pipeline finished successfully!")
        logger.info(f"Sleep stage distribution: {stage_distribution}")
        logger.info(f"Total recording time: {total_time_hours:.1f} hours")

        return PredictEDFResponse(
            model_configuration_used=config,
            input_file_name=input_filename,
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
    finally:
        temp_dir_manager.cleanup()

