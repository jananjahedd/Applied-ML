"""Module for the entire pipeline endpoints."""
import os
import tempfile
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
    possible_splits_dirs = ["splits-data", "data/splits",
                            "processed-data", MODELS_DIR, "results"]

    train_metrics = None
    val_metrics = None
    test_metrics = None

    for splits_dir in possible_splits_dirs:
        try:
            train_path = os.path.join(
                splits_dir, f"train_{config}_featured.npz"
            )
            if os.path.exists(train_path):
                train_data = np.load(train_path, allow_pickle=True)
                model = load_model(config)
                X_train = train_data['X_train']
                y_train = train_data['y_train']
                train_metrics = evaluate_model_on_data(
                    model, X_train, y_train, config
                )
                train_metrics["data_source"] = train_path

            val_path = os.path.join(splits_dir, f"val_{config}_featured.npz")
            if os.path.exists(val_path):
                val_data = np.load(val_path, allow_pickle=True)
                model = load_model(config)
                X_val = val_data['X_val']
                y_val = val_data['y_val']
                val_metrics = evaluate_model_on_data(
                    model, X_val, y_val, config
                )
                val_metrics["data_source"] = val_path

            test_path = os.path.join(splits_dir, f"test_{config}_featured.npz")
            if os.path.exists(test_path):
                test_data = np.load(test_path, allow_pickle=True)
                model = load_model(config)
                X_test = test_data['X_test']
                y_test = test_data['y_test']
                test_metrics = evaluate_model_on_data(
                    model, X_test, y_test, config
                )
                test_metrics["data_source"] = test_path

            if train_metrics or val_metrics or test_metrics:
                break

        except Exception as e:
            logger.warning(f"Error loading data from {splits_dir}: {e}")
            continue

    return {
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }


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
    - Accuracy: x% vs 20% random guessing (5-class problem)
    - F1-Score: x (significantly above random)
    - ROC-AUC: x (good discrimination)
    - Tested on 10 subjects, x epochs
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
    - Validation Accuracy: x vs 20% random guessing
    - Test Accuracy: x% (good generalization)
    - F1-Score: 0.x (above random)
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


@router.post("/predict-edf-with-metrics", response_model=Dict[str, Any])
async def predict_edf_with_comprehensive_metrics(
    edf_file: UploadFile = File(..., description="EDF sleep recording file"),
    hypno_file: UploadFile = File(
        ..., description="Hypnogram file for ground truth evaluation"
    ),
    config: str = DEFAULT_CONFIG,
    request: Request = None
):
    """
    Complete Prediction + Performance Analysis

    Upload EDF + hypnogram to get:
    1. Sleep stage predictions for your file
    2. Performance metrics on YOUR specific file
    3. Comparison with pre-trained model performance
    4. Model confidence and reliability analysis

    Perfect for: Comprehensive analysis of how well the model performs on your
    specific data.
    Requires: Both EDF file AND hypnogram for ground truth comparison
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
            detail="EDF file must have .edf extension"
        )

    if not hypno_file or not hypno_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Hypnogram file is required for comprehensive metrics analysis"
            )
        )

    logger.info(
        f"Starting comprehensive prediction + metrics "
        f"analysis for {edf_file.filename}"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        edf_path = os.path.join(temp_dir, edf_file.filename)
        hypno_path = os.path.join(temp_dir, hypno_file.filename)

        with open(edf_path, "wb") as f:
            content = await edf_file.read()
            f.write(content)

        with open(hypno_path, "wb") as f:
            content = await hypno_file.read()
            f.write(content)

        try:
            logger.info(
                "Step 1/5: Processing uploaded EDF file"
                " using existing pipeline..."
            )
            epochs, annotations_loaded = preprocess_edf_for_api(
                edf_path, hypno_path
            )

            if not annotations_loaded:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Could not load annotations from hypnogram file "
                        "- required for metrics analysis"
                    )
                )

            logger.info(
                "Step 2/5: Extracting features and ground truth labels..."
            )
            features, feature_names = extract_features_for_config(
                epochs, config
            )
            ground_truth = epochs.events[:, -1]

            logger.info("Step 3/5: Making predictions...")
            model = load_model(config)
            predictions = model.predict(features)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)

            logger.info(
                "Step 4/5: Calculating performance metrics on your file..."
            )
            current_file_metrics = evaluate_model_on_data(
                model, features, ground_truth, config, request
            )

            logger.info(
                "Step 5/5: Loading pre-trained metrics for comparison..."
            )
            pretrained_metrics = load_pretrained_metrics(config)

            if request and hasattr(request.app.state, 'label_mapping'):
                label_mapping = request.app.state.label_mapping
                prediction_labels = [
                    label_mapping.get(pred, f"Unknown_{pred}")
                    for pred in predictions
                ]
                ground_truth_labels = [
                    label_mapping.get(gt, f"Unknown_{gt}")
                    for gt in ground_truth
                ]
            else:
                prediction_labels = [f"Class_{pred}" for pred in predictions]
                ground_truth_labels = [f"Class_{gt}" for gt in ground_truth]

            comparison_analysis = {}
            if pretrained_metrics["test_metrics"]:
                test_acc = pretrained_metrics["test_metrics"][
                    "overall_metrics"]["accuracy"]
                test_f1 = pretrained_metrics["test_metrics"][
                    "overall_metrics"]["macro_f1_score"]
                current_acc = current_file_metrics["overall_metrics"][
                    "accuracy"]
                current_f1 = current_file_metrics["overall_metrics"][
                    "macro_f1_score"]

                comparison_analysis = {
                    "accuracy_vs_test_set": {
                        "your_file": round(current_acc, 4),
                        "original_test_set": round(test_acc, 4),
                        "difference": round(current_acc - test_acc, 4),
                        "performance_category": (
                            "Better" if current_acc > test_acc
                            else "Similar"
                            if abs(current_acc - test_acc) < 0.05 else "Worse"
                        )
                    },
                    "f1_score_vs_test_set": {
                        "your_file": round(current_f1, 4),
                        "original_test_set": round(test_f1, 4),
                        "difference": round(current_f1 - test_f1, 4),
                        "performance_category": (
                            "Better" if current_f1 > test_f1
                            else "Similar"
                            if abs(current_f1 - test_f1) < 0.05 else "Worse"
                        )
                    }
                }

            confidence_analysis = {}
            if probabilities is not None:
                max_probs = np.max(probabilities, axis=1)
                confidence_analysis = {
                    "mean_confidence": float(np.mean(max_probs)),
                    "median_confidence": float(np.median(max_probs)),
                    "min_confidence": float(np.min(max_probs)),
                    "max_confidence": float(np.max(max_probs)),
                    "high_confidence_predictions": int(
                        np.sum(max_probs > 0.8)
                    ),
                    "low_confidence_predictions": int(np.sum(max_probs < 0.5)),
                    "confidence_distribution": {
                        "very_high_0.9+": int(np.sum(max_probs >= 0.9)),
                        "high_0.8-0.9": int(
                            np.sum((max_probs >= 0.8) & (max_probs < 0.9))
                        ),
                        "medium_0.6-0.8": int(
                            np.sum((max_probs >= 0.6) & (max_probs < 0.8))
                        ),
                        "low_0.4-0.6": int(
                            np.sum((max_probs >= 0.4) & (max_probs < 0.6))
                        ),
                        "very_low_<0.4": int(np.sum(max_probs < 0.4))
                    }
                }

            unique_pred, counts_pred = np.unique(
                prediction_labels, return_counts=True
            )
            pred_distribution = dict(zip(unique_pred, counts_pred.tolist()))

            unique_true, counts_true = np.unique(
                ground_truth_labels, return_counts=True
            )
            true_distribution = dict(zip(unique_true, counts_true.tolist()))

            total_time_hours = len(predictions) * EPOCH_DURATION / 3600
            recording_quality = {
                "total_recording_time_hours": round(total_time_hours, 2),
                "total_epochs": len(predictions),
                "epochs_with_annotations": len(ground_truth),
                "annotation_completeness": (
                    round(len(ground_truth) / len(predictions), 4)
                ),
                "sleep_efficiency": (
                    round(1 - (
                        pred_distribution.get("Wake", 0) / len(predictions)
                    ), 4) if pred_distribution else None
                )
            }
            term1 = current_file_metrics['overall_metrics']['accuracy']
            term2 = confidence_analysis.get('mean_confidence', 0)
            term3 = (
                comparison_analysis.get('accuracy_vs_test_set', {})
                .get('performance_category', 'Unknown')
            )
            return {
                "model_configuration": config,
                "file_info": {
                    "edf_filename": edf_file.filename,
                    "hypnogram_filename": hypno_file.filename,
                    "recording_quality": recording_quality
                },
                "predictions": {
                    "predicted_stages": prediction_labels,
                    "ground_truth_stages": ground_truth_labels,
                    "prediction_ids": predictions.tolist(),
                    "ground_truth_ids": ground_truth.tolist(),
                    "probabilities_per_epoch": (
                        probabilities.tolist()
                        if probabilities is not None else None
                    )
                },
                "current_file_performance": current_file_metrics,
                "pretrained_model_performance": {
                    "training_metrics": pretrained_metrics[
                        "training_metrics"
                    ],
                    "validation_metrics": pretrained_metrics[
                        "validation_metrics"
                    ],
                    "test_metrics": pretrained_metrics["test_metrics"]
                },
                "performance_comparison": comparison_analysis,
                "confidence_analysis": confidence_analysis,
                "sleep_analysis": {
                    "predicted_distribution": pred_distribution,
                    "actual_distribution": true_distribution,
                    "stage_agreement_summary": {
                        stage: {
                            "predicted_count": pred_distribution.get(
                                stage, 0
                            ),
                            "actual_count": true_distribution.get(
                                stage, 0
                            ),
                            "difference": (
                                pred_distribution.get(stage, 0) -
                                true_distribution.get(stage, 0)
                            )
                        }
                        for stage in set(
                            list(pred_distribution.keys()) +
                            list(true_distribution.keys())
                        )
                    }
                },
                "recommendations": {
                    "model_reliability": (
                        "High" if current_file_metrics[
                            "overall_metrics"
                        ]["accuracy"] > 0.8 else
                        "Medium" if current_file_metrics[
                            "overall_metrics"
                        ]["accuracy"] > 0.6 else "Low"
                    ),
                    "confidence_level": (
                        "High" if confidence_analysis.get(
                            "mean_confidence", 0
                        ) > 0.8 else
                        "Medium" if confidence_analysis.get(
                            "mean_confidence", 0
                        ) > 0.6 else "Low"
                    ),
                    "clinical_usability": (
                        "Suitable for clinical review" if (
                            current_file_metrics[
                                "overall_metrics"
                            ]["accuracy"] > 0.75 and
                            confidence_analysis.get(
                                "mean_confidence", 0
                            ) > 0.7
                        ) else "Requires manual review"
                    ),
                    "notes": [
                        f"Model achieved {term1:.1%} accuracy on your file",
                        f"Average prediction confidence: {term2:.1%}",
                        f"Performance vs test set: {term3}"
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Error in comprehensive metrics pipeline: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error in comprehensive analysis: {str(e)}"
            )
