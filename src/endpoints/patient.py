# src/endpoints/patient.py
"""Patient data management endpoints - focused on loaded EDF data organization."""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
import os

from src.schemas import (
    PatientInfo,
    ResponseMessage,
    AllPatientsResponse
)
from src.utils.logger import get_logger

logger = get_logger("patient")

patients_data_store: Dict[str, Dict[str, Any]] = {}

from .files import selected_filenames

router = APIRouter(prefix="/patient", tags=["Patient Data"])


@router.get(
    "/{patient_id}",
    summary="Get information for a specific patient",
    description="Retrieve metadata for a patient whose EDF file has been processed.",
    response_model=PatientInfo
)
def get_patient_info(patient_id: str) -> PatientInfo:
    """Get patient information by patient ID."""
    if patient_id not in patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found. Use POST /patient/register to add patient metadata."
        )

    patient_data = patients_data_store[patient_id]
    return PatientInfo(**patient_data)


@router.get(
    "/",
    summary="Get all registered patients",
    description="Retrieve metadata for all patients in the system.",
    response_model=AllPatientsResponse
)
def get_all_patients() -> AllPatientsResponse:
    """Get information for all registered patients."""
    if not patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No patients registered. Use POST /patient/register to add patient metadata."
        )

    patients_dict = {
        patient_id: PatientInfo(**patient_data)
        for patient_id, patient_data in patients_data_store.items()
    }

    return AllPatientsResponse(patients=patients_dict)


@router.post(
    "/register",
    summary="Register patient metadata",
    description="Register metadata for a patient. This is separate from EDF processing.",
    response_model=ResponseMessage
)
def register_patient(
    patient_id: str,
    age: int = None,
    sex: str = None,
    additional_info: Dict[str, Any] = None
) -> ResponseMessage:
    """Register or update patient metadata."""

    # Basic validation
    if age is not None and (age < 0 or age > 150):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Age must be between 0 and 150"
        )

    if sex is not None and sex.lower() not in ['male', 'female', 'm', 'f', 'other']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sex must be one of: male, female, m, f, other"
        )

    patient_data = {
        "number": int(patient_id) if patient_id.isdigit() else hash(patient_id) % 10000,
        "age": age,
        "sex": sex.lower() if sex else None,
        "num_recordings": 0
    }

    if additional_info:
        patient_data.update(additional_info)

    patients_data_store[patient_id] = patient_data

    action = "updated" if patient_id in patients_data_store else "registered"
    return ResponseMessage(message=f"Patient {patient_id} {action} successfully.")


@router.delete(
    "/{patient_id}",
    summary="Remove patient from system",
    description="Remove patient metadata from the system.",
    response_model=ResponseMessage
)
def remove_patient(patient_id: str) -> ResponseMessage:
    """Remove patient from the system."""
    if patient_id not in patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found."
        )

    del patients_data_store[patient_id]
    return ResponseMessage(message=f"Patient {patient_id} removed successfully.")


@router.post(
    "/link-to-file",
    summary="Link patient to processed EDF file",
    description="Associate a patient with a processed EDF file from the files endpoint.",
    response_model=ResponseMessage
)
def link_patient_to_file(patient_id: str, file_id: str) -> ResponseMessage:
    """Link a patient to a selected EDF file."""

    if patient_id not in patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found. Register patient first."
        )

    if file_id not in selected_filenames:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File ID {file_id} not found in selected files. Select the file first using /files/select."
        )

    patients_data_store[patient_id]["num_recordings"] = patients_data_store[patient_id].get("num_recordings", 0) + 1
    patients_data_store[patient_id]["linked_files"] = patients_data_store[patient_id].get("linked_files", [])

    if file_id not in patients_data_store[patient_id]["linked_files"]:
        patients_data_store[patient_id]["linked_files"].append(file_id)

    file_path = selected_filenames[file_id]
    return ResponseMessage(
        message=f"Patient {patient_id} linked to file {file_id} ({os.path.basename(file_path)}) successfully."
    )


@router.get(
    "/{patient_id}/files",
    summary="Get files linked to patient",
    description="Get all EDF files linked to a specific patient.",
    response_model=Dict[str, Any]
)
def get_patient_files(patient_id: str) -> Dict[str, Any]:
    """Get all files linked to a patient."""
    if patient_id not in patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found."
        )

    patient_data = patients_data_store[patient_id]
    linked_files = patient_data.get("linked_files", [])

    file_details = {}
    for file_id in linked_files:
        if file_id in selected_filenames:
            file_details[file_id] = {
                "path": selected_filenames[file_id],
                "filename": os.path.basename(selected_filenames[file_id])
            }

    return {
        "patient_id": patient_id,
        "linked_files": file_details,
        "total_files": len(file_details)
    }
