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
    description="""
    Step 3: Get Patient Information

    Retrieve metadata for a patient whose data has been registered.

    Example patient_id:`SC4161`

    Workflow:
    1. First register patient: `POST /patient/register`
    2. Select EDF files: `POST /files/select`
    3. Link patient to files: `POST /patient/link-to-file`
    4. Then use this endpoint to view patient info
    """,
    response_model=PatientInfo
)
def get_patient_info(patient_id: str) -> PatientInfo:
    """Get patient information by patient ID."""
    if patient_id not in patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found. Use POST /patient/register to add patient metadata first."
        )

    patient_data = patients_data_store[patient_id]
    patient_data_with_id = {**patient_data, "patient_id": patient_id}
    return PatientInfo(**patient_data_with_id)


@router.get(
    "/",
    summary="Get all registered patients",
    description="""
    Step 4: View All Patients

    Retrieve metadata for all patients in the system.

    Returns: Dictionary with patient_id as key and patient info as value

    Prerequisites:
    - At least one patient must be registered via `POST /patient/register`
    """,
    response_model=AllPatientsResponse
)
def get_all_patients() -> AllPatientsResponse:
    """Get information for all registered patients."""
    if not patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No patients registered. Use POST /patient/register to add patient metadata first."
        )

    patients_dict = {}
    for patient_id, patient_data in patients_data_store.items():
        patient_data_with_id = {**patient_data, "patient_id": patient_id}
        patients_dict[patient_id] = PatientInfo(**patient_data_with_id)

    return AllPatientsResponse(patients=patients_dict)


@router.post(
    "/register",
    summary="Step 1: Register patient metadata",
    description="""
    Step 1: Register Patient (Required First Step)

    Register metadata for a patient before linking to EDF files.

    Example Usage:
    json
    {
        "patient_id": "SC4161",
        "age": 32,
        "sex": "M"
    }

    Complete Workflow:
    1. Register patient ← You are here
    2. Select EDF files: `POST /files/select`
    3. Link patient to files: `POST /patient/link-to-file`
    4. View patient info: `GET /patient/{patient_id}`
    """,
    response_model=ResponseMessage
)
def register_patient(
    patient_id: str,
    age: int = None,
    sex: str = None,
    additional_info: Dict[str, Any] = None
) -> ResponseMessage:
    """Register or update patient metadata."""

    if not patient_id or not patient_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient ID cannot be empty."
        )

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
        "patient_id": patient_id,  # Keep original string ID
        "number": int(patient_id) if patient_id.isdigit() else hash(patient_id) % 10000,
        "age": age,
        "sex": sex.lower() if sex else None,
        "num_recordings": 0,
        "linked_files": []
    }

    if additional_info:
        patient_data.update(additional_info)

    action = "updated" if patient_id in patients_data_store else "registered"
    patients_data_store[patient_id] = patient_data

    return ResponseMessage(
        message=f"Patient '{patient_id}' {action} successfully. Next: Select EDF files using POST /files/select"
    )


@router.delete(
    "/{patient_id}",
    summary="Remove patient from system",
    description="""
    Remove patient metadata from the system.

    Example: `/patient/SC4161` to remove patient with ID
    """,
    response_model=ResponseMessage
)
def remove_patient(patient_id: str) -> ResponseMessage:
    """Remove patient from the system."""
    if patient_id not in patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found."
        )

    del patients_data_store[patient_id]
    return ResponseMessage(message=f"Patient '{patient_id}' removed successfully.")


@router.post(
    "/link-to-file",
    summary="Step 3: Link patient to processed EDF file",
    description="""
    Step 3: Link Patient to EDF File

    Associate a registered patient with a selected EDF file.

    Example Request Body:
    json
    {
        "patient_id": "SC4161",
        "file_id": "SC4161E0-PSG"
    }


    Parameters Explained:
    - `patient_id`: String ID you used when registering (e.g., "SC4161")
    - `file_id`: Filename from selected files (e.g., "SC4171E0-PSG.edf")

    Prerequisites:
    1. Patient must be registered: `POST /patient/register`
    2. File must be selected: `POST /files/select`

    Get file_id from: `GET /files/selected` (shows available file IDs)
    """,
    response_model=ResponseMessage
)
def link_patient_to_file(patient_id: str, file_id: str) -> ResponseMessage:
    """Link a patient to a selected EDF file."""

    if patient_id not in patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found. Register patient first using POST /patient/register"
        )

    file_found = False
    file_path = None

    for selected_id, selected_path in selected_filenames.items():
        if os.path.basename(selected_path) == file_id or selected_path == file_id:
            file_found = True
            file_path = selected_path
            break

    if not file_found:
        available_files = [os.path.basename(path) for path in selected_filenames.values()]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found in selected files. Available files: {available_files}. Select files first using POST /files/select"
        )

    patients_data_store[patient_id]["num_recordings"] = patients_data_store[patient_id].get("num_recordings", 0) + 1

    if "linked_files" not in patients_data_store[patient_id]:
        patients_data_store[patient_id]["linked_files"] = []

    if file_id not in patients_data_store[patient_id]["linked_files"]:
        patients_data_store[patient_id]["linked_files"].append(file_id)

    return ResponseMessage(
        message=f"Patient '{patient_id}' linked to file '{file_id}' successfully. Patient now has {patients_data_store[patient_id]['num_recordings']} recording(s)."
    )


@router.get(
    "/{patient_id}/files",
    summary="Get files linked to patient",
    description="""
    Step 4: View Patient's Linked Files

    Get all EDF files linked to a specific patient.

    Example: `/patient/SC4161/files`

    Returns: List of files associated with this patient
    """,
    response_model=Dict[str, Any]
)
def get_patient_files(patient_id: str) -> Dict[str, Any]:
    """Get all files linked to a patient."""
    if patient_id not in patients_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient '{patient_id}' not found."
        )

    patient_data = patients_data_store[patient_id]
    linked_files = patient_data.get("linked_files", [])

    file_details = {}
    for file_id in linked_files:
        for selected_id, selected_path in selected_filenames.items():
            if os.path.basename(selected_path) == file_id or selected_path == file_id:
                file_details[file_id] = {
                    "path": selected_path,
                    "filename": os.path.basename(selected_path),
                    "selected_id": selected_id
                }
                break

    return {
        "patient_id": patient_id,
        "patient_info": {
            "age": patient_data.get("age"),
            "sex": patient_data.get("sex"),
            "num_recordings": patient_data.get("num_recordings", 0)
        },
        "linked_files": file_details,
        "total_files": len(file_details)
    }
