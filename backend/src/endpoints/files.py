# src/endpoints/files.py
"""File handling endpoints for server-side EDF files."""

import glob
from os.path import basename
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status, Body
from src.schemas import (
    AvailableFilesResponse,
    SelectedFilesResponse,
    ResponseMessage
)

selected_filenames: Dict[str, str] = {}


EXAMPLE_DATA_ROOT = "example-data"
CASSETTE_DATA_DIR = f"{EXAMPLE_DATA_ROOT}/sleep-cassette"
TELEMETRY_DATA_DIR = f"{EXAMPLE_DATA_ROOT}/sleep-telemetry"


router = APIRouter(
    prefix="/files",
    tags=["File Management"]
)


@router.get(
    "/available",
    summary="Step 1: Get available EDF files from server",
    description="""
    Step 1: Discover Available EDF Files

    Retrieves all EDF files found on the server in predefined directories.

    Server Directories Scanned:
    - `example-data/sleep-cassette/` (Cassette recordings)
    - `example-data/sleep-telemetry/` (Telemetry recordings)

    Returns: Dictionary with numeric IDs and filenames

    Example Response:
    json
    {
        "cassette_files": {
            "1": "SC4171E0-PSG.edf",
            "2": "SC4172E0-PSG.edf"
        },
        "telemetry_files": {
            "3": "ST7052J0-PSG.edf"
        }
    }

    Next Step: Use the numeric IDs with `POST /files/select`
    """,
    response_model=AvailableFilesResponse
)
def get_available_files() -> AvailableFilesResponse:
    """Get a list of available files in the example-data directory."""
    try:
        cassette_files_paths = sorted(glob.glob(f"{CASSETTE_DATA_DIR}/*-PSG.edf"))
        telemetry_files_paths = sorted(glob.glob(f"{TELEMETRY_DATA_DIR}/*-PSG.edf"))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error accessing server file directories: {str(e)}")

    cassette_files_dict = {
        str(i + 1): basename(file) for i, file in enumerate(cassette_files_paths)
    }
    telemetry_files_dict = {
        str(i + 1 + len(cassette_files_paths)): basename(file)
        for i, file in enumerate(telemetry_files_paths)
    }
    return AvailableFilesResponse(
        cassette_files=cassette_files_dict,
        telemetry_files=telemetry_files_dict
    )


@router.get(
    "/selected",
    summary="Step 3: Get currently selected files",
    description="""
    Step 3: View Selected Files

    Shows which EDF files are currently marked as "selected" and ready for patient linking.

    Returns: Full file paths of selected files

    Example Response:
    ```json
    {
        "selected_files": {
            "1": "example-data/sleep-cassette/SC4171E0-PSG.edf",
            "2": "example-data/sleep-cassette/SC4172E0-PSG.edf"
        },
        "total_selected": 2
    }
    ```

    Use the filenames (e.g., "SC4171E0-PSG.edf") for linking to patients
    """,
    response_model=SelectedFilesResponse
)
def get_selected_files() -> SelectedFilesResponse:
    """Get list of currently selected patient files."""
    # if not selected_filenames:
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail="No files selected. Use POST /files/select to select files first."
    #     )

    return SelectedFilesResponse(
        selected_files=selected_filenames,
        total_selected=len(selected_filenames)
    )


@router.post(
    "/select",
    summary="Step 2: Select EDF files for processing",
    description="""
    Step 2: Select Files for Processing

    Mark specific EDF files as "selected" using their numeric IDs from `/files/available`.

    Example Request Body:
    json
    {
        "file_ids": [1, 2, 3]
    }

    How to get file IDs:
    1. Call `GET /files/available` first
    2. Note the numeric IDs (1, 2, 3, etc.)
    3. Use those IDs in this endpoint

    What this does:
    - Marks files as "ready for patient linking"
    - Required before you can link files to patients

    Next Step: Link patients to selected files using `POST /patient/link-to-file`
    """,
    response_model=ResponseMessage
)
def select_patient_files(file_ids: List[int] = Body(..., embed=True)) -> ResponseMessage: # Expect list of integer IDs
    """Select patient files by their ids from the available files."""

    try:
        cassette_files_paths = sorted(glob.glob(f"{CASSETTE_DATA_DIR}/*-PSG.edf"))
        telemetry_files_paths = sorted(glob.glob(f"{TELEMETRY_DATA_DIR}/*-PSG.edf"))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error accessing server file directories during selection: {str(e)}")

    all_files_on_server_map: Dict[int, str] = {}
    current_id = 1
    for file_path in cassette_files_paths:
        all_files_on_server_map[current_id] = file_path
        current_id += 1
    for file_path in telemetry_files_paths:
        all_files_on_server_map[current_id] = file_path
        current_id += 1

    if not file_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file IDs provided. Example: [1, 2, 3]. Get IDs from GET /files/available"
        )

    newly_selected_count = 0
    already_selected_count = 0
    not_found_ids = []

    for file_id in file_ids:
        str_file_id = str(file_id)
        if file_id not in all_files_on_server_map:
            not_found_ids.append(file_id)
            continue
        if str_file_id in selected_filenames:
            already_selected_count += 1
            continue

        selected_filenames[str_file_id] = all_files_on_server_map[file_id]
        newly_selected_count += 1

    if not_found_ids:
        available_ids = list(all_files_on_server_map.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File IDs not found: {not_found_ids}. Available IDs: {available_ids}. Get valid IDs from GET /files/available"
        )

    if newly_selected_count == 0 and already_selected_count > 0 and not not_found_ids:
        return ResponseMessage(
            message=f"All {already_selected_count} specified files were already selected. View with GET /files/selected"
        )

    selected_filenames_list = [basename(path) for path in selected_filenames.values()]
    return ResponseMessage(
        message=f"{newly_selected_count} file(s) selected successfully. {already_selected_count} were already selected. Selected files: {selected_filenames_list}. Next: Register patients with POST /patient/register"
    )


@router.delete(
    "/deselect",
    summary="Remove files from selected list",
    description="""
    Remove Files from Selection

    Remove specific files from the "selected" list using their numeric IDs.

    Example Request Body:**
    json
    {
        "file_ids": [1, 2]
    }

    Use Case: If you accidentally selected wrong files
    """,
    response_model=ResponseMessage
)
def deselect_patient_files(file_ids: List[int] = Body(..., embed=True)) -> ResponseMessage:
    """Deselect patient files by their ids from the selected files."""
    if not file_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file IDs provided to deselect. Example: [1, 2]"
        )

    deselected_count = 0
    not_found_in_selection_ids = []

    for file_id in file_ids:
        str_file_id = str(file_id)
        if str_file_id not in selected_filenames:
            not_found_in_selection_ids.append(file_id)
        else:
            del selected_filenames[str_file_id]
            deselected_count += 1

    if not_found_in_selection_ids:
        return ResponseMessage(
            message=f"{deselected_count} file(s) deselected. File IDs not found in current selection: {not_found_in_selection_ids}. View current selection with GET /files/selected"
        )

    if deselected_count == 0 and not not_found_in_selection_ids:
        return ResponseMessage(
            message="No specified files were found in the current selection to deselect."
        )

    return ResponseMessage(
        message=f"{deselected_count} file(s) deselected successfully."
    )
