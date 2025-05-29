"""Patient endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException  # type: ignore
from starlette.status import (  # type: ignore
    HTTP_200_OK,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from src.endpoints.data import patients, selected_filenames
from src.utils.patient import patient_from_filepath

router = APIRouter(prefix="/patient", tags=["Patient"])


@router.get(  # type: ignore
    "/",
    summary="Get Patient Information",
    description="Returns the patient information",
    responses={
        HTTP_200_OK: {
            "description": "Patient information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "patient": {
                            "number": 123,
                            "age": 45,
                        }
                    }
                }
            },
        },
        HTTP_404_NOT_FOUND: {
            "description": "Patient not found",
            "content": {"application/json": {"example": {"message": "Patient not found."}}},
        },
    },
)
def get_patient_info() -> Dict[int, Any]:
    """Get patient information."""
    if not patients:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="No patients loaded.")

    response = {}
    for patient_number, patient in patients.items():
        response[patient_number] = {
            "number": patient.number,
            "age": patient.age,
            "sex": patient.sex.value,
            "num_recordings": len(patient.recordings),
        }
    return response


@router.get(  # type: ignore
    "/{patient_number}",
    summary="Get Patient by Number",
    description="Returns the patient information for the given patient number",
    responses={
        HTTP_200_OK: {
            "description": "Patient information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "patient": {
                            "number": 123,
                            "age": 45,
                        }
                    }
                }
            },
        },
        HTTP_404_NOT_FOUND: {
            "description": "Patient not found",
            "content": {"application/json": {"example": {"message": "Patient not found."}}},
        },
    },
)
def get_patient_by_number(patient_number: int) -> Dict[str, Any]:
    """Get patient information by patient number."""
    if patient_number not in patients:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Patient not found.")

    return patients[patient_number].dict()


@router.post(  # type: ignore
    "/load",
    summary="Load patients from selected files",
    description="Load patients from the selected files in the example-data directory.",
    responses={
        HTTP_200_OK: {
            "description": "Patient added successfully",
            "content": {"application/json": {"example": {"message": "Patient added successfully."}}},
        },
        HTTP_404_NOT_FOUND: {
            "description": "No selected files found",
            "content": {"application/json": {"example": {"message": "No selected files found."}}},
        },
        HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"message": "An error occurred while loading the patient."}}},
        },
    },
)
async def load_patient() -> Dict[str, str]:
    """Load patient data from selected files."""
    if not selected_filenames:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="No selected files found.")
    try:
        for idx, file in selected_filenames.items():
            patients[idx] = patient_from_filepath(file)
        return {"message": "Patient loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(  # type: ignore
    "/{patient_number}/visualize-recording/{recording_number}",
    summary="Visualize a Patient's Recording",
    description="Opens an interactive plot of the patient's recording",
    responses={
        HTTP_200_OK: {
            "description": "Plot visualized successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Plot viualization successful.",
                    }
                }
            },
        },
        HTTP_404_NOT_FOUND: {
            "description": "Patient not found",
            "content": {"application/json": {"example": {"message": "Patient not found."}}},
        },
        HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"message": "An error occurred while visualizing the data."}}},
        },
    },
)
def visualize_recording(patient_number: int, recording_number: int) -> Dict[str, Any]:
    """Returns a visualization of the patient's recording."""
    if patient_number not in patients:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Patient not found.")
    if recording_number not in patients[patient_number].recordings:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Recording not found.")

    try:
        patient = patients[patient_number]
        recording = patient.recordings[recording_number]
        recording.visualize()
        return {"message": "Plot visualization successful."}
    except Exception as e:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
