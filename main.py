"""This is a sample Python script."""

from fastapi import FastAPI  # type: ignore

from src.endpoints.files import router as files_router
from src.endpoints.patient import router as patient_router
from src.endpoints.pipeline import router as pipeline_router

app = FastAPI(title="Placeholder", version="1.0.0", description="Placeholder")

app.include_router(files_router)
app.include_router(patient_router)
app.include_router(pipeline_router)
