# src/endpoints/files.py
"""File handling endpoints for server-side EDF files."""

import glob
from os.path import basename
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status
# Assuming schemas.py is in src
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
)


@router.get(
    "/available",
    summary="Get available patient EDF files from server",
    description=(
        "Retrieves a list of available EDF patient files found in predefined "
        "server directories (e.g., 'example-data/sleep-cassette/' and "
        "'example-data/sleep-telemetry/')."
    ),
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

    # Assign string IDs for consistency with selection keys
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
    summary="Get currently selected patient files",
    description="Retrieve the list of patient files currently marked as selected for processing.",
    response_model=SelectedFilesResponse
)
def get_selected_files() -> SelectedFilesResponse:
    """Get list of currently selected patient files."""
    if not selected_filenames:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files selected.")

    return SelectedFilesResponse(
        selected_files=selected_filenames,
        total_selected=len(selected_filenames)
    )

@router.post(
    "/select",
    summary="Select patient files by ID",
    description=(
        "Mark specific patient files (by their IDs obtained from '/files/available') "
        "as selected. This prepares them for further operations like loading patient data."
    ),
    response_model=ResponseMessage
)
def select_patient_files(file_ids: List[int]) -> ResponseMessage: # Expect list of integer IDs
    """Select patient files by their ids from the available files."""
    # This function needs full paths, not just basenames, to store in selected_filenames
    # So, it must reglob or have access to the full paths from get_available_files
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file IDs provided.")

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

        selected_filenames[str_file_id] = all_files_on_server_map[file_id] # Store full path
        newly_selected_count += 1

    if not_found_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"File IDs not found: {not_found_ids}. "
                                   f"{newly_selected_count} files newly selected. "
                                   f"{already_selected_count} files were already selected.")

    if newly_selected_count == 0 and already_selected_count > 0 and not not_found_ids:
        return ResponseMessage(message=f"All {already_selected_count} specified files were already selected. No new files added.")

    return ResponseMessage(message=f"{newly_selected_count} file(s) selected successfully. "
                                   f"{already_selected_count} were already selected.")


@router.delete(
    "/deselect",
    summary="Deselect patient files by ID",
    description="Remove specified patient files (by ID) from the 'selected' list.",
    response_model=ResponseMessage
)
def deselect_patient_files(file_ids: List[int]) -> ResponseMessage:
    """Deselect patient files by their ids from the selected files."""
    if not file_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file IDs provided to deselect.")

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
             message=f"{deselected_count} file(s) deselected. "
                     f"File IDs not found in current selection: {not_found_in_selection_ids}."
         )

    if deselected_count == 0 and not not_found_in_selection_ids:
        return ResponseMessage(message="No specified files were found in the current selection to deselect.")

    return ResponseMessage(message=f"{deselected_count} file(s) deselected successfully.")