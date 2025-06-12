from glob import glob
from os.path import exists
from typing import Any, Dict
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR

from src.schemas.schemas import AvailableRecordings, Patient, RecordingSummary, Recording, ResponseMessage
from src.utils.patient import patient_from_filepath
from src.utils.recording import Recording as RecordingUtil, is_valid_annotation_name, is_valid_edf_name

router = APIRouter(prefix="/recordings", tags=["Recordings"])

CASSETTE_DATA_DIR = f"example-data/sleep-cassette"
TELEMETRY_DATA_DIR = f"example-data/sleep-telemetry"

def _get_all_recordings() -> Dict:
    try:
        casette_paths = sorted(glob(f"{CASSETTE_DATA_DIR}/*-PSG.edf"))
        telemetry_paths = sorted(glob(f"{TELEMETRY_DATA_DIR}/*-PSG.edf"))
    except Exception as e:
        raise Exception(f"Error accessing available recordings: {str(e)}")

    # Gather cassette and telemetry recordings
    cassette_files_dict = {
        i + 1: RecordingUtil(file) for i, file in enumerate(casette_paths)
    }
    telemetry_files_dict = {
        i + 1 + len(cassette_files_dict): RecordingUtil(file)
        for i, file in enumerate(telemetry_paths)
    }

    return {
        "cassette_files": cassette_files_dict,
        "telemetry_files": telemetry_files_dict
    }

@router.get(
    "/health",
    summary="Health check for available recordings",
    description="Check if the recordings directory is accessible, each recording file is readable and has a corresponding annotation file.",
    responses={
        HTTP_200_OK: {
            "description": "All recordings are accessible and have corresponding annotation files.",
            "content": {
                "application/json": {
                    "example": {
                        "message": "✅ All recordings are accessible and have corresponding annotation files."
                    }
                }
            }
        },
        HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Recordings directory is not set or does not exist."
                    }
                }
            }
        }
    }
)
def health_check() -> Any:
    """
    Health check for available recordings.

    This endpoint checks if the recordings directory is accessible and if each recording file is readable.
    """
    # check if CASETTE_DATA_DIR and TELEMETRY_DATA_DIR exist
    if not (CASSETTE_DATA_DIR and TELEMETRY_DATA_DIR):
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recordings directory is not set or does not exist."
        )

    # check if each reacording has a corresponding annotation file
    cassette_files = sorted(glob(f"{CASSETTE_DATA_DIR}/*-PSG.edf"))
    telemetry_files = sorted(glob(f"{TELEMETRY_DATA_DIR}/*-PSG.edf"))

    for file in cassette_files:
        try:
            number = file.split("SC4")[1][:2]
        except IndexError:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid cassette file name format: {file}. Files are named in the form SC4ssNEO-PSG.edf where ss is the subject number, and N is the night."
            )
        if not glob(f"{CASSETTE_DATA_DIR}/SC4{number}*-Hypnogram.edf"):
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Missing annotation file for cassette recording: {file}"
            )

    for file in telemetry_files:
        try:
            number = file.split("ST7")[1][:2]
        except IndexError:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid telemetry file name format: {file}. Files are named in the form ST7ssNJ0-PSG.edf where ss is the subject number, and N is the night."
            )
        if not glob(f"{TELEMETRY_DATA_DIR}/ST7{number}*-Hypnogram.edf"):
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Missing annotation file for telemetry recording: {file}"
            )
    
    return ResponseMessage(
        message="✅ All recordings are accessible and have corresponding annotation files."
    )

@router.get(
    "/",
    summary="Get all recordings",
    response_model=AvailableRecordings,
    description="Retrieve a list of all available recordings, including both cassette and telemetry recordings.",
    responses={
        HTTP_200_OK: {
            "description": "List of available recordings",
            "content": {
                "application/json": {
                    "example": {
                        "cassette_files": {
                            "1": {
                                "recording_path": f"{CASSETTE_DATA_DIR}/SC4171E0-PSG.edf",
                                "annotation_path": f"{CASSETTE_DATA_DIR}/SC4171E0-Hypnogram.edf"
                            }
                        },
                        "telemetry_files": {
                            "1": {
                                "recording_path": f"{TELEMETRY_DATA_DIR}/ST70001N0-PSG.edf",
                                "annotation_path": f"{TELEMETRY_DATA_DIR}/ST70001N0-Hypnogram.edf"
                            }
                        }
                    }
                }
            }
        },
        HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Error accessing available recordings: <error_message>"
                    }
                }
            }
        }
    }
)
def get_all_recordings() -> AvailableRecordings:
    """
    Retrieve a list of all recordings.
    """
    try:
        all_recordings = _get_all_recordings()
    except Exception as e:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing recordings: {str(e)}")

    return AvailableRecordings(
        cassette_files={
            int(key): RecordingSummary(
                recording_path=recording.file_path,
                annotation_path=recording.anno_path,
                study_type=recording.study_type.value,
            ) for key, recording in all_recordings["cassette_files"].items()
        },
        telemetry_files={
            int(key): RecordingSummary(
                recording_path=recording.file_path,
                annotation_path=recording.anno_path,
                study_type=recording.study_type.value,
            ) for key, recording in all_recordings["telemetry_files"].items()
        }
    )


@router.get(
    "/{recording_id}",
    summary="Get recording by ID",
    response_model=Recording,
    description="Retrieve all information about a specific recording by its ID.",
    responses={
        HTTP_200_OK: {
            "description": "Recording details",
            "content": {
                "application/json": {
                    "example": {
                        "recording_path": f"{CASSETTE_DATA_DIR}/SC4171E0-PSG.edf",
                        "annotation_path": f"{CASSETTE_DATA_DIR}/SC4171E0-Hypnogram.edf",
                        "study_type": "Cassette",
                        "night": 1,
                        "patient": {
                            "number": 10,
                            "age": 45,
                            "sex": "Male"
                        }
                    }
                }
            }
        },
        HTTP_400_BAD_REQUEST: {
            "description": "Invalid recording ID",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid recording ID"
                    }
                }
            }
        },
        HTTP_404_NOT_FOUND: {
            "description": "Recording not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Recording not found"
                    }
                }
            }
        },
        HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Error processing recordings: <error_message>"
                    }
                }
            }
        }
    }
)
async def get_recording_by_id(recording_id: int) -> Recording:
    try:
        all_recordings = _get_all_recordings()
    except Exception as e:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing recordings: {str(e)}")

    if recording_id <= 0 or not isinstance(recording_id, int):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Invalid recording ID")

    all_recordings_flat = {**all_recordings["cassette_files"], **all_recordings["telemetry_files"]}
    if recording_id not in all_recordings_flat:
        raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Recording not found")

    recording = all_recordings_flat[recording_id]
    patient_info = patient_from_filepath(recording.file_path)
    return Recording(
        recording_path=recording.file_path,
        annotation_path=recording.anno_path,
        study_type=recording.study_type.value,
        night=recording.night,
        patient=Patient(
            number=patient_info.number,
            age=patient_info.age,
            sex=patient_info.sex.value,
        )
    )

@router.post(
    "/upload",
    summary="Upload a new recording",
    response_model=ResponseMessage,
    description="Upload a new recording file along with its annotation file.",
)
async def upload_recording(
        edf_file: UploadFile = File(None, description="EDF recording file"),
        hypno_file: UploadFile = File(None, description="EDF annotation file")
) -> ResponseMessage:
    """
    Upload a new recording file along with its annotation file.
    """
    # check if both files are named correctly
    edf_name, hypno_name = str(edf_file.filename), str(hypno_file.filename)
    if not is_valid_edf_name(edf_name):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Invalid EDF file name: {edf_file.filename}. Files should be named in the form SC4ssNEO-PSG.edf or ST7ssNJ0-PSG.edf."
        )
    if not is_valid_annotation_name(hypno_name):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Invalid annotation file name: {hypno_file.filename}. Files should be named in the form SC4ssNEO-Hypnogram.edf or ST7ssNJ0-Hypnogram.edf."
        )

    # Check if patient number and night match in both files
    if edf_name[:6] != hypno_name[:6]:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="EDF file and annotation file do not match. Ensure both files have the same patient number and night."
        )

    # Check if files already exist
    edf_path = f"{CASSETTE_DATA_DIR}/{edf_name}"
    hypno_path = f"{CASSETTE_DATA_DIR}/{hypno_name}"
    if exists(edf_path) or exists(hypno_path):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Files with the same name already exist: {edf_name} or {hypno_name}. Please rename your files."
        )

    # Save the uploaded files
    with open(edf_path, "wb") as edf_out:
        edf_out.write(await edf_file.read())
    with open(hypno_path, "wb") as hypno_out:
        hypno_out.write(await hypno_file.read())

    # Create a Recording object to validate the files
    try:
        recording = RecordingUtil(edf_path)
    except ValueError as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Error processing recording files: {str(e)}"
        )

    return ResponseMessage(
        message=f"✅ Recording '{recording.file_path}' and annotation '{recording.anno_path}' uploaded successfully."
    )

