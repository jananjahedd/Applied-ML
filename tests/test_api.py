""" For testing the FastAPI application endpoints. """
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# main.py tests

def test_read_root():
    """
    Test the root endpoint (/).
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": (
            "Welcome to the Sleep Stage Prediction API! "
            "Use /docs to explore endpoints."
        )
    }


def test_health_check():
    """
    Test the health check endpoint (/health).
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running successfully!"}
