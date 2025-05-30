import os
import pathlib
import pytest  # type: ignore
from fastapi.testclient import TestClient
from main import app

from src.utils.paths import get_repo_root

client = TestClient(app)


ROOT = pathlib.Path(get_repo_root())
TEST_DATA_DIR = ROOT / "example-data" / "sleep-cassette"
SAMPLE_EDF_PATH = os.path.join(TEST_DATA_DIR, "SC4161E0-PSG.edf")


@pytest.mark.skipif(
        not os.path.exists(SAMPLE_EDF_PATH),
        reason="Sample EDF file not found"
    )
def test_predict_edf_default_config():
    """Test the /pipeline/predict-edf endpoint with a sample EDF file."""
    with open(SAMPLE_EDF_PATH, "rb") as edf_file:
        response = client.post(
            "/pipeline/predict-edf",
            files={
                "edf_file": ("sample.edf", edf_file,
                             "application/octet-stream")
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["model_configuration_used"] == "eeg_emg_eog"
    assert data["input_file_name"] == "sample.edf"
    assert data["num_segments_processed"] > 0
    assert isinstance(data["predictions"], list)
    assert isinstance(data["prediction_ids"], list)
    assert len(data["predictions"]) == data["num_segments_processed"]

    if data["class_labels_legend"]:
        assert data["class_labels_legend"]["1"] == "Wake"


@pytest.mark.skipif(
        not os.path.exists(SAMPLE_EDF_PATH),
        reason="Sample EDF file not found"
    )
def test_predict_edf_specific_config_eeg():
    """Test with a specific model configuration like 'eeg'."""
    with open(SAMPLE_EDF_PATH, "rb") as edf_file:
        response = client.post(
            "/pipeline/predict-edf",
            files={
                "edf_file": ("sample.edf", edf_file,
                             "application/octet-stream")
            },
            params={"config": "eeg"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["model_configuration_used"] == "eeg"
    assert data["input_file_name"] == "sample.edf"


def test_predict_edf_invalid_config():
    """Test /pipeline/predict-edf with an invalid model configuration."""
    dummy_file_content = b"this is not an edf"
    response = client.post(
        "/pipeline/predict-edf",
        files={
            "edf_file": ("dummy.edf", dummy_file_content,
                         "application/octet-stream")
        },
        params={"config": "invalid_config_name"}
    )
    assert response.status_code == 400
    assert "Invalid configuration" in response.json()["detail"]


def test_predict_edf_no_file():
    response = client.post("/pipeline/predict-edf")
    assert response.status_code == 422


def test_predict_edf_wrong_file_type():
    response = client.post(
        "/pipeline/predict-edf",
        files={"edf_file": ("sample.txt", b"content", "text/plain")}
    )
    assert response.status_code == 400
    assert "File must be an EDF file" in response.json()["detail"]


def test_synthetic_prediction_default_config():
    """Test /pipeline/test-prediction with default config."""
    response = client.post("/pipeline/test-prediction")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["test_results"]["model_configuration"] == "eeg_emg_eog"
    assert "predicted_sleep_stage" in data["test_results"]
    assert "prediction_id" in data["test_results"]


def test_synthetic_prediction_specific_config_eeg():
    """Test /pipeline/test-prediction with 'eeg' config."""
    response = client.post(
        "/pipeline/test-prediction",
        params={"config": "eeg"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["test_results"]["model_configuration"] == "eeg"


def test_synthetic_prediction_invalid_config():
    response = client.post(
        "/pipeline/test-prediction",
        params={"config": "non_existent_config"}
    )
    assert response.status_code == 400
    assert "Invalid configuration" in response.json()["detail"]


def test_synthetic_prediction_model_not_found(mocker):
    """
    Test /pipeline/test-prediction when a model file is missing.
    We use 'mocker' (from pytest-mock) to simulate os.path.exists
    returning False for the specific model being tested by load_model.
    """
    mocker.patch("src.endpoints.pipeline.load_model",
                 side_effect=FileNotFoundError("Mocked: Model file not found"))

    response = client.post(
        "/pipeline/test-prediction",
        params={"config": "eeg"}
    )
    assert response.status_code == 404
    assert "Model file not found" in response.json()["detail"]


# tests for other pipeline endpoints
def test_get_available_models():
    response = client.get("/pipeline/available-models")
    assert response.status_code == 200
    data = response.json()
    assert "available_configurations" in data
    assert "eeg" in data["available_configurations"]


def test_pipeline_health_check():
    response = client.get("/pipeline/health")
    if any(
        os.path.exists(f"results/svm_model_{cfg}.joblib")
        for cfg in ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]
    ):
        assert response.status_code == 200
        assert "ML Pipeline is healthy!" in response.json()["message"]
    else:
        assert response.status_code == 503
        assert "No trained models available" in response.json()["detail"]
