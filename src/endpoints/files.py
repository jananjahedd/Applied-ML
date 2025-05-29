"""File handling endpoints."""

import glob
from os.path import basename
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status  # type: ignore
from starlette.status import (  # type:ignore
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
)

from src.endpoints.data import selected_filenames

router = APIRouter(prefix="/files", tags=["Files"])


@router.get(  # type: ignore
    "/available",
    summary="Get available patient files",
    description="Retrieve a list of available patient files in the example-data directory.",
    responses={
        HTTP_200_OK: {
            "description": "Available files retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "cassette_files": {1: "patient1-PSG.edf", 2: "patient2-PSG.edf"},
                        "telemetry_files": {3: "patient3-PSG.edf", 4: "patient4-PSG.edf"},
                    }
                }
            },
        },
    },
)
def get_available_files() -> Dict[str, Any]:
    """Get a list of available files in the example-data directory."""
    cassette_files = glob.glob("example-data/sleep-cassette/*-PSG.edf")
    telemetry_files = glob.glob("example-data/sleep-telemetry/*-PSG.edf")

    return {
        "cassette_files": {i + 1: basename(file) for i, file in enumerate(cassette_files)},
        "telemetry_files": {i + 1 + len(cassette_files): basename(file) for i, file in enumerate(telemetry_files)},
    }


@router.get(  # type: ignore
    "/selected",
    summary="Get selected patient files",
    description="Retrieve the list of currently selected patient files.",
    responses={
        HTTP_200_OK: {
            "description": "Selected files retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "selected_files": {1: "patient1-PSG.edf", 2: "patient2-PSG.edf"},
                        "total_selected": 2,
                    }
                }
            },
        },
        HTTP_404_NOT_FOUND: {
            "description": "No files selected",
            "content": {"application/json": {"example": {"message": "No files selected."}}},
        },
    },
)
def get_selected_files() -> Dict[str, Any]:
    """Get list of currently selected patient files."""
    if not selected_filenames:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files selected.")

    return {
        "selected_files": selected_filenames,
        "total_selected": len(selected_filenames),
    }


@router.post(  # type: ignore
    "/select",
    summary="Select patient files",
    description="Select patient files by id from the available files.",
    responses={
        HTTP_200_OK: {
            "description": "Files selected successfully",
            "content": {"application/json": {"example": {"message": "Files selected successfully."}}},
        },
        HTTP_400_BAD_REQUEST: {
            "description": "File already selected",
            "content": {"application/json": {"example": {"message": "File already selected."}}},
        },
        HTTP_404_NOT_FOUND: {
            "description": "File not found",
            "content": {"application/json": {"example": {"message": "File not found."}}},
        },
    },
)
def select_patient_files(file_ids: List[int]) -> Dict[str, Any]:
    """Select patient files by their ids from the available files."""
    cassette_files = glob.glob("example-data/sleep-cassette/*-PSG.edf")
    telemetry_files = glob.glob("example-data/sleep-telemetry/*-PSG.edf")

    all_files = {i + 1: file for i, file in enumerate(cassette_files + telemetry_files)}

    for file_id in file_ids:
        if file_id not in all_files:
            raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="File not found.")
        if file_id in selected_filenames:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="File already selected.")
        selected_filenames[file_id] = all_files[file_id]

    return {"message": "Files selected successfully."}


@router.delete(  # type: ignore
    "/deselect",
    summary="Deselect patient files",
    description="Deselect patient files by their ids from the selected files.",
    responses={
        HTTP_200_OK: {
            "description": "Files deselected successfully",
            "content": {"application/json": {"example": {"message": "Files deselected successfully."}}},
        },
        HTTP_404_NOT_FOUND: {
            "description": "File not found in selected files",
            "content": {"application/json": {"example": {"message": "File not found in selected files."}}},
        },
    },
)
def deselect_patient_files(file_ids: List[int]) -> Dict[str, Any]:
    """Deselect patient files by their ids from the selected files."""
    for file_id in file_ids:
        if file_id not in selected_filenames:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found in selected files.")
        del selected_filenames[file_id]

    return {"message": "Files deselected successfully."}
