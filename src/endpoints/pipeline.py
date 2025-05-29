# src/endpoints/pipeline.py
"""Complete pipeline endpoints for EDF preprocessing, feature extraction, and prediction."""

import os
import tempfile
import pathlib
from typing import Dict, List, Optional, Any, Tuple
import joblib
import numpy as np
import mne
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, status
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.schemas import (
    ResponseMessage,
    PreprocessingOutput,
    UploadResponse,
    PredictionInput,
    PredictionOutput,
    PredictEDFResponse
)
from src.features.feature_engineering import FeatureEngineering
from src.utils.logger import get_logger

logger = get_logger("pipeline")

router = APIRouter(prefix="/pipeline", tags=["ML Pipeline"])

MODELS_DIR = "results"
AVAILABLE_CONFIGS = ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]
DEFAULT_CONFIG = "eeg_emg_eog"

TARGET_SFREQ = 100.0
EPOCH_DURATION = 30.0
NOTCH_FREQ = 50.0
EEG_BANDPASS = (0.3, 35.0)
EOG_BANDPASS = (0.3, 35.0)
EMG_BANDPASS = (10.0, 45.0)

ANNOTATION_MAP = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
    "Sleep stage ?": 0,
    "Movement time": 0,
}


def bandpass_filter(raw: mne.io.BaseRaw, low_freq: float, high_freq: float, signal_type: str) -> mne.io.BaseRaw:
    """Apply bandpass filter to specific channel types."""
    sfreq = raw.info["sfreq"]
    nyquist_freq = sfreq / 2.0
    effective_high_freq = min(high_freq, nyquist_freq - 0.5)

    if low_freq >= effective_high_freq:
        logger.warning(f"Skipping {signal_type} filter: low_freq >= effective_high_freq")
        return raw

    picks = None
    if signal_type == "eeg":
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
    elif signal_type == "eog":
        picks = mne.pick_types(raw.info, meg=False, eog=True, exclude="bads")
    elif signal_type == "emg":
        picks = mne.pick_types(raw.info, meg=False, emg=True, exclude="bads")

    if picks is not None and len(picks) > 0:
        try:
            raw.filter(
                l_freq=low_freq,
                h_freq=effective_high_freq,
                picks=picks,
                method="fir",
                phase="zero-double",
                fir_design="firwin",
                skip_by_annotation="edge",
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Error applying bandpass filter for {signal_type}: {e}")

    return raw


def notch_filter(raw: mne.io.BaseRaw, freq: float, signal_type: str) -> mne.io.BaseRaw:
    """Apply notch filter to specific channel types."""
    sfreq = raw.info["sfreq"]
    nyquist = sfreq / 2.0

    if freq >= nyquist:
        return raw

    picks = None
    if signal_type == "eeg":
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
    elif signal_type == "eog":
        picks = mne.pick_types(raw.info, meg=False, eog=True, exclude="bads")
    elif signal_type == "emg":
        picks = mne.pick_types(raw.info, meg=False, emg=True, exclude="bads")

    if picks is not None and len(picks) > 0:
        try:
            raw.notch_filter(
                freqs=freq,
                picks=picks,
                method="fir",
                phase="zero-double",
                fir_design="firwin",
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Error applying notch filter for {signal_type}: {e}")

    return raw


def preprocess_edf_data(file_path: str, hypno_path: Optional[str] = None) -> Tuple[mne.Epochs, bool]:
    """
    Preprocess EDF data following the same pipeline as training.

    Returns:
        Tuple of (epochs, annotations_loaded)
    """
    logger.info(f"Loading EDF file: {file_path}")

    # Load raw data
    exclude_ch = ["Event marker", "Marker", "Status"]
    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        exclude=exclude_ch,
        infer_types=True,
        verbose=False
    )

    logger.info(f"Data loaded. SFreq: {raw.info['sfreq']:.2f} Hz. Channels: {len(raw.ch_names)}")

    # Set EOG channel type if present
    eog_channel_name = "horizontal"
    if eog_channel_name in raw.ch_names:
        try:
            current_type = raw.get_channel_types(picks=[eog_channel_name])[0]
            if current_type != "eog":
                raw.set_channel_types({eog_channel_name: "eog"})
                logger.info(f"Set '{eog_channel_name}' channel type to 'eog'")
        except Exception as e:
            logger.warning(f"Could not set channel type for '{eog_channel_name}': {e}")

    # Load annotations if provided
    annotations_loaded = False
    if hypno_path and os.path.exists(hypno_path):
        try:
            temp_annots = mne.read_annotations(hypno_path, verbose=False)
            raw.set_annotations(temp_annots, emit_warning=False)
            annotations_loaded = True
            logger.info(f"Annotations loaded from {hypno_path}")
        except Exception as e:
            logger.warning(f"Could not load annotations: {e}")

    # Resample if needed
    current_sfreq = raw.info["sfreq"]
    if current_sfreq != TARGET_SFREQ:
        logger.info(f"Resampling from {current_sfreq:.2f} Hz to {TARGET_SFREQ:.2f} Hz")
        raw.resample(sfreq=TARGET_SFREQ, npad="auto", verbose=False)

    # Apply filters
    logger.info("Applying filters...")
    raw = bandpass_filter(raw, EEG_BANDPASS[0], EEG_BANDPASS[1], "eeg")
    raw = bandpass_filter(raw, EOG_BANDPASS[0], EOG_BANDPASS[1], "eog")
    raw = bandpass_filter(raw, EMG_BANDPASS[0], EMG_BANDPASS[1], "emg")

    # Apply notch filter
    nyquist = TARGET_SFREQ / 2.0
    if NOTCH_FREQ < nyquist:
        logger.info(f"Applying {NOTCH_FREQ} Hz notch filter")
        raw = notch_filter(raw, NOTCH_FREQ, "eeg")
        raw = notch_filter(raw, NOTCH_FREQ, "eog")
        raw = notch_filter(raw, NOTCH_FREQ, "emg")

    # Create epochs
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
                # Create event_id mapping for present labels
                present_ids = np.unique(events[:, 2])
                event_id_map = {
                    "Wake": 1, "N1": 2, "N2": 3, "N3/N4": 4, "REM": 5, "Unknown": 0
                }
                epochs_event_id = {
                    name: id_ for name, id_ in event_id_map.items()
                    if id_ in present_ids
                }

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
                logger.info(f"Created {len(epochs)} labeled epochs")
            else:
                logger.warning("No events found, creating fixed-length epochs")
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


def extract_features_for_config(epochs: mne.Epochs, config: str) -> Tuple[np.ndarray, List[str]]:
    """Extract features for a specific configuration."""
    # Define which modalities to include for each config
    modality_mapping = {
        "eeg": ["eeg"],
        "eeg_emg": ["eeg", "emg"],
        "eeg_eog": ["eeg", "eog"],
        "eeg_emg_eog": ["eeg", "emg", "eog"]
    }

    if config not in modality_mapping:
        raise ValueError(f"Unknown configuration: {config}")

    modalities_to_include = modality_mapping[config]

    # Get all channel info
    all_ch_names = epochs.info.ch_names
    all_ch_types = epochs.info.get_channel_types(unique=False, picks="all")
    sfreq = epochs.info["sfreq"]

    # Select channels for this configuration
    selected_ch_indices = [
        idx for idx, ch_type in enumerate(all_ch_types)
        if ch_type in modalities_to_include
    ]

    if not selected_ch_indices:
        raise ValueError(f"No channels found for configuration '{config}'")

    selected_ch_names = [all_ch_names[i] for i in selected_ch_indices]
    modality_ch_types = [all_ch_types[i] for i in selected_ch_indices]

    # Create info object for selected channels
    modality_info = mne.create_info(
        ch_names=selected_ch_names,
        sfreq=sfreq,
        ch_types=modality_ch_types
    )

    # Extract data for selected channels
    epochs_data = epochs.get_data()[:, selected_ch_indices, :]

    # Create epochs array for feature extraction
    if hasattr(epochs, 'events') and epochs.events is not None:
        events = epochs.events
    else:
        # Create dummy events for unlabeled data
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

    # Extract features
    feature_engineer = FeatureEngineering()
    X_features, _, feature_names = feature_engineer._extract_features(epochs_for_extraction)

    return X_features, feature_names


def load_model(config: str) -> SklearnPipeline:
    """Load the trained model for a specific configuration."""
    model_path = os.path.join(MODELS_DIR, f"svm_model_{config}.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model for configuration: {config}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


@router.post("/upload-and-preprocess", response_model=UploadResponse, include_in_schema=False)
async def upload_and_preprocess_edf(
    edf_file: UploadFile = File(...),
    hypno_file: Optional[UploadFile] = File(None),
    config: str = DEFAULT_CONFIG
):
    """
    Internal endpoint: Upload an EDF file and get extracted features.
    Use /predict-edf for complete automated pipeline instead.
    """
    if config not in AVAILABLE_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration. Available: {AVAILABLE_CONFIGS}"
        )

    if not edf_file.filename.endswith('.edf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an EDF file"
        )

    # Save uploaded files temporarily
    with tempfile.TemporaryDirectory() as temp_dir:
        edf_path = os.path.join(temp_dir, edf_file.filename)
        hypno_path = None

        # Save EDF file
        with open(edf_path, "wb") as f:
            content = await edf_file.read()
            f.write(content)

        # If provided hypnograph file save it
        if hypno_file and hypno_file.filename:
            hypno_path = os.path.join(temp_dir, hypno_file.filename)
            with open(hypno_path, "wb") as f:
                content = await hypno_file.read()
                f.write(content)

        try:
            # Preprocess the data
            epochs, annotations_loaded = preprocess_edf_data(edf_path, hypno_path)

            # Extract features for the given configuration
            features, feature_names = extract_features_for_config(epochs, config)

            # Create preprocessing output
            preprocessing_output = PreprocessingOutput(
                original_filename=edf_file.filename,
                status="success",
                message=f"Successfully preprocessed {len(epochs)} epochs with {len(feature_names)} features for {config} configuration",
                extracted_features=features.flatten().tolist(),  # Flatten for API response
                feature_names=feature_names
            )

            return UploadResponse(
                filename=edf_file.filename,
                detail=f"File uploaded and preprocessed successfully. Annotations loaded: {annotations_loaded}",
                preprocessing_output=preprocessing_output
            )

        except Exception as e:
            logger.error(f"Error preprocessing file {edf_file.filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during preprocessing: {str(e)}"
            )


@router.post("/predict-features", response_model=PredictionOutput, include_in_schema=False)
async def predict_from_features(
    prediction_input: PredictionInput,
    config: str = DEFAULT_CONFIG,
    request: Request = None
):
    """
    Internal endpoint: Predict sleep stage from pre-extracted features.
    Use /predict-edf for complete automated pipeline instead.
    """
    if config not in AVAILABLE_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration. Available: {AVAILABLE_CONFIGS}"
        )

    try:
        # Load the model
        model = load_model(config)

        # Prepare features for prediction
        features_array = np.array(prediction_input.features).reshape(1, -1)

        # Make prediction
        prediction_id = model.predict(features_array)[0]

        # Get prediction probabilities if available
        confidence_score = None
        class_probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            confidence_score = float(np.max(probabilities))

            # Get class labels from app state
            if request and hasattr(request.app.state, 'label_mapping'):
                label_mapping = request.app.state.label_mapping
                class_probabilities = {
                    label_mapping.get(i, f"Class_{i}"): float(prob)
                    for i, prob in enumerate(probabilities)
                }

        # Convert prediction ID to label
        if request and hasattr(request.app.state, 'label_mapping'):
            label_mapping = request.app.state.label_mapping
            prediction_label = label_mapping.get(prediction_id, f"Unknown_{prediction_id}")
        else:
            prediction_label = f"Class_{prediction_id}"

        return PredictionOutput(
            prediction_label=prediction_label,
            prediction_id=int(prediction_id),
            confidence_score=confidence_score,
            class_probabilities=class_probabilities
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}"
        )


@router.post("/predict-edf", response_model=PredictEDFResponse)
async def predict_edf_file(
    edf_file: UploadFile = File(..., description="EDF sleep recording file"),
    hypno_file: Optional[UploadFile] = File(None, description="Optional hypnogram file for better epoch creation"),
    config: str = DEFAULT_CONFIG,
    request: Request = None
):
    """
    **Complete Automated Sleep Stage Prediction Pipeline**

    **Just upload your EDF file and get sleep stage predictions!**

    This endpoint automatically:
    1. Uploads and validates your EDF file
    2. Preprocesses the signal (filtering, resampling, epoching)
    3. Extracts sleep-relevant features from EEG/EOG/EMG
    4. Predicts sleep stages using trained SVM models
    5. Returns detailed predictions with confidence scores

    **Input**: Just your EDF file (+ optional hypnogram)
    **Output**: Sleep stage predictions for every 30-second epoch

    **Supported configurations:**
    - `eeg`: EEG channels only
    - `eeg_emg`: EEG + EMG (good for REM detection)
    - `eeg_eog`: EEG + EOG (good for eye movement artifacts)
    - `eeg_emg_eog`: All channels (most comprehensive, default)
    """
    if config not in AVAILABLE_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration '{config}'. Available: {AVAILABLE_CONFIGS}"
        )

    if not edf_file.filename or not edf_file.filename.lower().endswith('.edf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an EDF file with .edf extension"
        )

    file_size_mb = edf_file.size / (1024 * 1024) if edf_file.size else 0
    if file_size_mb > 500:  # change if needed
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is 500MB."
        )

    logger.info(f"Starting automated prediction pipeline for {edf_file.filename} using {config} configuration")

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
            # Preprocess the EDF data
            logger.info("Step 1/4: Preprocessing EDF data...")
            epochs, annotations_loaded = preprocess_edf_data(edf_path, hypno_path)
            logger.info(f"Preprocessing complete. Created {len(epochs)} epochs of {EPOCH_DURATION}s each")

            # Extract features for the specified configuration
            logger.info(f"Step 2/4: Extracting features for {config} configuration...")
            features, feature_names = extract_features_for_config(epochs, config)
            logger.info(f"Feature extraction complete. Extracted {features.shape[1]} features per epoch")

            # Load model and predict
            logger.info("Step 3/4: Loading trained model and making predictions...")
            model = load_model(config)
            predictions = model.predict(features)
            logger.info(f"Predictions complete for {len(predictions)} epochs")

            # Get prediction probabilities and format results
            logger.info("Step 4/4: Generating confidence scores and formatting results...")
            probabilities_per_segment = None
            if hasattr(model, 'predict_proba'):
                probabilities_per_segment = model.predict_proba(features).tolist()

            # Convert predictions to human-readable labels
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

            # Calculate summary statistics
            unique_stages, counts = np.unique(prediction_labels, return_counts=True)
            stage_distribution = dict(zip(unique_stages, counts.tolist()))
            total_time_hours = len(predictions) * EPOCH_DURATION / 3600

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
                    "sleep_stage_distribution": stage_distribution
                }
            )

        except Exception as e:
            logger.error(f"Error in complete EDF prediction pipeline: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error in prediction pipeline: {str(e)}"
            )


@router.get("/available-models", response_model=Dict[str, Any])
async def get_available_models():
    """Get list of available model configurations."""
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
    """ Health check for the ML pipeline - Check if models are available and ready."""
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
            detail=" No trained models available. Please ensure model files are in the 'models/' directory."
        )

    return ResponseMessage(
        message=f" ML Pipeline is healthy! {available_count}/{len(AVAILABLE_CONFIGS)} models ready: {available_models}"
    )


@router.post("/test-prediction", response_model=Dict[str, Any])
async def test_model_prediction(
    config: str = DEFAULT_CONFIG,
    request: Request = None
):
    """
     **Test the ML pipeline with synthetic data**

    This endpoint lets you test if a model configuration works without uploading real EDF files.
    It generates synthetic features and returns a prediction to verify the model is working.

    **Useful for:**
    - Testing if models are loaded correctly
    - Checking model configurations
    - API debugging and development
    """
    if config not in AVAILABLE_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration '{config}'. Available: {AVAILABLE_CONFIGS}"
        )

    try:

        model = load_model(config)

        # Generate synthetic features (approximate feature count for each config) - change depending on new model
        feature_counts = {
            "eeg": 25,
            "eeg_emg": 30,
            "eeg_eog": 35,
            "eeg_emg_eog": 40
        }

        n_features = feature_counts.get(config, 30)
        synthetic_features = np.random.rand(1, n_features)

        # Make prediction
        prediction_id = model.predict(synthetic_features)[0]

        # Get probabilities
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
            prediction_label = label_mapping.get(prediction_id, f"Unknown_{prediction_id}")
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
            "note": "This was a test with synthetic data. Upload real EDF files using /predict-edf for actual predictions."
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f" Model file not found for configuration '{config}': {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing model '{config}': {str(e)}"
        )
