# src/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ResponseMessage(BaseModel):
    message: str


class FileDetail(BaseModel):
    id: int
    name: str


class AvailableFilesResponse(BaseModel):
    cassette_files: Dict[str, str] = Field(
        ...,
        description="Cassette EDF files with ID as key and filename as value",
        json_schema_extra={"example": {"1": "SC4171E0-PSG.edf", "2": "SC4172E0-PSG.edf"}}
    )
    telemetry_files: Dict[str, str] = Field(
        ...,
        description="Telemetry EDF files with ID as key and filename as value",
        json_schema_extra={"example": {"3": "ST7052J0-PSG.edf", "4": "ST7053J0-PSG.edf"}}
    )


class SelectedFileDetail(BaseModel):
    id: str
    path: str


class SelectedFilesResponse(BaseModel):
    selected_files: Dict[str, str] = Field(
        ...,
        description="Selected files with ID as key and full path as value",
        json_schema_extra={"example": {
            "1": "example-data/sleep-cassette/SC4171E0-PSG.edf",
            "2": "example-data/sleep-cassette/SC4172E0-PSG.edf"
        }}
    )
    total_selected: int


class PatientInfo(BaseModel):
    patient_id: Optional[str] = Field(None, description="String identifier for the patient")
    number: int = Field(description="Internal numeric ID")
    age: Optional[int] = None
    sex: Optional[str] = None
    num_recordings: Optional[int] = Field(0, description="Number of EDF files linked to this patient")
    linked_files: Optional[List[str]] = Field([], description="List of linked EDF filenames")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "john_doe",
                "number": 1234,
                "age": 45,
                "sex": "male",
                "num_recordings": 2,
                "linked_files": ["SC4171E0-PSG.edf", "SC4172E0-PSG.edf"]
            }
        }


class AllPatientsResponse(BaseModel):
    patients: Dict[str, PatientInfo] = Field(
        description="Dictionary with patient_id as key and patient info as value"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "patients": {
                    "john_doe": {
                        "patient_id": "john_doe",
                        "number": 1234,
                        "age": 45,
                        "sex": "male",
                        "num_recordings": 1,
                        "linked_files": ["SC4171E0-PSG.edf"]
                    },
                    "jane_smith": {
                        "patient_id": "jane_smith",
                        "number": 5678,
                        "age": 32,
                        "sex": "female",
                        "num_recordings": 2,
                        "linked_files": ["SC4172E0-PSG.edf", "SC4173E0-PSG.edf"]
                    }
                }
            }
        }


class PreprocessingOutput(BaseModel):
    original_filename: str
    status: str
    message: str
    extracted_features: Optional[List[float]] = Field(
        None, json_schema_extra={"example": [0.1, 0.5, 0.2, 1.3, 0.8]}
    )
    feature_names: Optional[List[str]] = Field(
        None, json_schema_extra={"example": [
            "EEG_Fpz_Cz_delta_RelP", "EEG_Fpz_Cz_theta_RelP",
            "EOG_horizontal_Var"
        ]}
    )


class UploadResponse(BaseModel):
    filename: str
    detail: str
    preprocessing_output: PreprocessingOutput


class PredictionInput(BaseModel):
    features: List[float] = Field(
        ..., json_schema_extra={"example": [0.1, 0.5, 0.2, 0.8, 1.5]}
    )
    feature_names: Optional[List[str]] = None


class PredictionOutput(BaseModel):
    prediction_label: str = Field(..., json_schema_extra={"example": "N2"})
    prediction_id: int = Field(..., json_schema_extra={"example": 3})
    confidence_score: Optional[float] = Field(
        None, json_schema_extra={"example": 0.85}
    )
    class_probabilities: Optional[Dict[str, float]] = None


class PredictEDFResponse(BaseModel):
    model_configuration_used: str
    input_file_name: Optional[str]
    num_segments_processed: int
    predictions: List[str]
    prediction_ids: List[int]
    probabilities_per_segment: Optional[List[List[float]]] = None
    class_labels_legend: Optional[Dict[int, str]] = None
    processing_summary: Optional[Dict[str, Any]] = Field(
        None,
        json_schema_extra={
            "example": {
                "total_recording_time_hours": 8.5,
                "epoch_duration_seconds": 30.0,
                "annotations_from_hypnogram": True,
                "features_extracted_per_epoch": 45,
                "sleep_stage_distribution": {
                    "Wake": 120,
                    "N1": 45,
                    "N2": 180,
                    "N3": 90,
                    "REM": 85
                }
            }
        }
    )
