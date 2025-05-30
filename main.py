"""Main application file for the FastAPI app."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from src.endpoints.files import router as files_router
from src.endpoints.patient import router as patient_router
from src.endpoints.pipeline import router as pipeline_router
from src.schemas import ResponseMessage

# configuration
LABEL_TO_NAME_MAPPING: dict[int, str] = {
    1: "Wake",
    2: "N1",
    3: "N2",
    4: "N3",
    5: "REM",
    0: "Unknown/Movement"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
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
        print(
            f"Found {len(available_models)} trained models: {available_models}"
        )
    print("Application startup complete!")

    yield

    print("Shutting down Sleep Stage Prediction API...")


app = FastAPI(
    title="Sleep Stage Prediction API",
    description=(
        "API for EDF sleep recording processing, feature extraction, "
        "and sleep stage prediction using pre-trained SVM models. "
        "Supports EDF file uploads for automatic preprocessing and prediction."
    ),
    version="1.0.3",
    lifespan=lifespan
)


@app.get("/", response_model=ResponseMessage, tags=["General"])
async def read_root():
    """Provides a welcome message and API status."""
    return ResponseMessage(
        message=(
            "Welcome to the Sleep Stage Prediction API! "
            "Use /docs to explore endpoints."
        )
    )


@app.get("/health", response_model=ResponseMessage, tags=["General"])
async def health_check():
    """General health check for the API."""
    return ResponseMessage(message="API is running successfully!")


app.include_router(files_router)
app.include_router(patient_router)
app.include_router(pipeline_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
