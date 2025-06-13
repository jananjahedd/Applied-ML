"""Tests the main.py file."""
from fastapi.testclient import TestClient
from backend.main import app


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "message": (
                "Welcome to the Sleep Stage Prediction API! "
                "Use /docs to explore endpoints."
            )
        }


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "API is running successfully!"}


def test_startup_event():
    with TestClient(app) as client:
        assert hasattr(
            app.state, 'label_mapping'
        ), "app.state.label_mapping should be set by lifespan startup"
        expected_label_mapping = {
            1: "Wake",
            2: "N1",
            3: "N2",
            4: "N3",
            5: "REM",
            0: "Unknown/Movement"
        }
        assert app.state.label_mapping == expected_label_mapping

        response = client.get("/pipeline/available-models")
        assert response.status_code == 200

        response = client.post("/pipeline/test-prediction")

        # check the endpoint response
        assert response.status_code == 200, (
            f"Test prediction endpoint failed: {response.text}"
        )
        data = response.json()

        assert data.get("status") == "success"
        assert "predicted_sleep_stage" in data.get("test_results", {})

        prediction_id = data.get("test_results", {}).get("prediction_id")
        if prediction_id is not None:
            assert prediction_id in expected_label_mapping
            assert (
                data["test_results"]["predicted_sleep_stage"]
                == expected_label_mapping.get(prediction_id)
            )
