"""Module to test the schemas.py"""
import pytest  # type: ignore
from pydantic import ValidationError


from src.schemas import (
    PreprocessingOutput,
    UploadResponse,
    PredictionInput,
    PredictionOutput
)


def test_preprocessing_output_valid():
    data = {
        "original_filename": "test.edf",
        "status": "success",
        "message": "Processed successfully",
        "extracted_features": [0.1, 0.2],
        "feature_names": ["feat1", "feat2"]
    }
    output = PreprocessingOutput(**data)
    assert output.original_filename == data["original_filename"]
    assert output.status == data["status"]
    assert output.extracted_features == data["extracted_features"]


def test_preprocessing_output_optional_fields():
    data = {
        "original_filename": "test.edf",
        "status": "pending",
        "message": "Processing..."
    }
    output = PreprocessingOutput(**data)
    assert output.original_filename == data["original_filename"]
    assert output.extracted_features is None
    assert output.feature_names is None


def test_preprocessing_output_missing_required():
    with pytest.raises(ValidationError):
        PreprocessingOutput(status="error")


def test_upload_response_valid():
    preprocessing_data = {
        "original_filename": "uploaded.edf",
        "status": "completed",
        "message": "Uploaded and preprocessed."
    }
    data = {
        "filename": "uploaded.edf",
        "detail": "File processed",
        "preprocessing_output": preprocessing_data
    }
    response = UploadResponse(**data)
    assert response.filename == data["filename"]
    assert response.preprocessing_output.status == "completed"


def test_prediction_input_valid():
    data = {"features": [1.0, 2.5, 3.0]}
    p_input = PredictionInput(**data)
    assert p_input.features == data["features"]
    assert p_input.feature_names is None


def test_prediction_input_with_feature_names():
    data = {"features": [1.0, 2.0], "feature_names": ["a", "b"]}
    p_input = PredictionInput(**data)
    assert p_input.features == data["features"]
    assert p_input.feature_names == data["feature_names"]


def test_prediction_input_missing_features():
    with pytest.raises(ValidationError):
        PredictionInput(feature_names=["a", "b"])


def test_prediction_input_invalid_features_type():
    with pytest.raises(ValidationError):
        PredictionInput(features="not_a_list")


def test_prediction_output_valid():
    data = {"prediction_label": "N1", "prediction_id": 2}
    p_output = PredictionOutput(**data)
    assert p_output.prediction_label == "N1"
    assert p_output.prediction_id == 2
    assert p_output.confidence_score is None
    assert p_output.class_probabilities is None


def test_prediction_output_with_all_fields():
    data = {
        "prediction_label": "REM",
        "prediction_id": 5,
        "confidence_score": 0.95,
        "class_probabilities": {"REM": 0.95, "Wake": 0.05}
    }
    p_output = PredictionOutput(**data)
    assert p_output.prediction_label == "REM"
    assert p_output.confidence_score == 0.95
    assert p_output.class_probabilities == data["class_probabilities"]


def test_prediction_output_missing_required():
    with pytest.raises(ValidationError):
        PredictionOutput(confidence_score=0.5)
