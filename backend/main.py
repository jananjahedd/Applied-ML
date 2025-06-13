"""Main application file for the FastAPI app."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.endpoints.models import health_check as models_health_check
from src.endpoints.models import router as models_router
from src.endpoints.recordings import health_check as recordings_health_check
from src.endpoints.recordings import router as recordings_router
from starlette.status import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR
from glob import glob
from fastapi import APIRouter
from src.endpoints.recordings import CASSETTE_DATA_DIR, TELEMETRY_DATA_DIR

# configuration
LABEL_TO_NAME_MAPPING: dict[int, str] = {1: "Wake", 2: "N1", 3: "N2", 4: "N3", 5: "REM", 0: "Unknown/Movement"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize application state."""
    print("Starting Sleep Stage Prediction API...")
    app.state.label_mapping = LABEL_TO_NAME_MAPPING
    models_dir = "results"
    if not os.path.exists(models_dir):
        print(f"WARNING: Models directory '{models_dir}' not found.")
        print("Models will be loaded dynamically by pipeline endpoints.")
    else:
        available_models = []
        for config in ["eeg", "eeg_emg", "eeg_eog", "eeg_emg_eog"]:
            model_path = os.path.join(models_dir, f"svm_model_{config}.joblib")
            if os.path.exists(model_path):
                available_models.append(config)
        print(f"Found {len(available_models)} trained models: {available_models}")
    print("Application startup complete!")

    yield

    print("Shutting down Sleep Stage Prediction API...")


app = FastAPI(
    title="Sleep Stage Prediction API",
    description=(
        "API for EDF sleep recording processing, feature extraction, "
        "and sleep stage prediction using pre-trained Random Forest models. "
    ),
    version="1.0.3",
    lifespan=lifespan,
)

origins = [
    "http://localhost",
    "http://localhost:8501",
]


@app.get("/", tags=["General"])
def homepage() -> Dict[str, str | Dict[str, str]]:
    """Provides a welcome message and API status."""
    return {
        "message": "Welcome to the Sleep Stage Prediction API!",
        "version": "1.0.3",
        "endpoints": {
            "recordings": "/recordings -> See available recordings, and upload new ones",
            "pipeline": "/pipeline -> Process EDF files and predict sleep stages",
            "health_checks": "See individual endpoints for health checks",
        },
    }


@app.get(
    "/health",
    tags=["General"],
    summary="Health Check",
    description="Endpoint to check the health of the API and its components.",
    responses={
        HTTP_200_OK: {
            "description": "API is healthy",
            "content": {"application/json": {"example": {"status": "OK", "recordings": "OK"}}},
        },
        HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {"status": "ERROR", "recordings": "Failed to connect to recordings service"}
                }
            },
        },
    },
)
def health_check() -> Dict[str, str]:
    """Health check endpoint to verify API is running."""
    status = "OK"
    recordings_response = None
    models_response = None
    error = None
    try:
        recordings_response = recordings_health_check()
        models_response = models_health_check()
    except Exception as e:
        status = "ERROR"
        error = str(e)

    if status == "OK":
        return {
            "status": status,
            "recordings": recordings_response.message if recordings_response else "OK",
            "models": models_response.message if models_response else "OK",
        }
    else:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": status,
                "recordings": recordings_response.message if recordings_response else error,
                "models": models_response.message if models_response else error,
            },
        )

# --- START OF NEW DEBUGGING CODE ---

debug_router = APIRouter()

@debug_router.get("/debug-info", tags=["Debug"])
def get_debug_info():
    """
    This endpoint provides live diagnostic information from within the container.
    """
    cassette_path = CASSETTE_DATA_DIR
    telemetry_path = TELEMETRY_DATA_DIR

    # Test what the glob pattern finds
    cassette_glob_results = glob(f"{cassette_path}/*-PSG.edf")
    telemetry_glob_results = glob(f"{telemetry_path}/*-PSG.edf")

    # Test what os.listdir() finds (this is often more reliable)
    try:
        cassette_listdir = os.listdir(cassette_path)
    except FileNotFoundError:
        cassette_listdir = f"Directory NOT FOUND at: {cassette_path}"
    
    try:
        telemetry_listdir = os.listdir(telemetry_path)
    except FileNotFoundError:
        telemetry_listdir = f"Directory NOT FOUND at: {telemetry_path}"

    return {
        "info": "Live diagnostics from the backend container.",
        "cassette_path_used": cassette_path,
        "cassette_path_exists": os.path.isdir(cassette_path),
        "cassette_listdir_contents": cassette_listdir,
        "cassette_glob_pattern": f"{cassette_path}/*-PSG.edf",
        "cassette_glob_found_count": len(cassette_glob_results),
        "cassette_glob_results": cassette_glob_results,
        "telemetry_path_used": telemetry_path,
        "telemetry_path_exists": os.path.isdir(telemetry_path),
        "telemetry_listdir_contents": telemetry_listdir,
        "telemetry_glob_pattern": f"{telemetry_path}/*-PSG.edf",
        "telemetry_glob_found_count": len(telemetry_glob_results),
        "telemetry_glob_results": telemetry_glob_results,
    }

# --- END OF NEW DEBUGGING CODE ---

app.include_router(recordings_router)
app.include_router(models_router)
app.include_router(debug_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
