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
        ..., json_schema_extra={"example": {"1": "patient1-PSG.edf"}}
    )
    telemetry_files: Dict[str, str] = Field(
        ..., json_schema_extra={"example": {"3": "patient3-PSG.edf"}}
    )


class SelectedFileDetail(BaseModel):
    id: str
    path: str


class SelectedFilesResponse(BaseModel):
    selected_files: Dict[str, str] = Field(
        ..., json_schema_extra={"example": {
            "1": "example-data/sleep-cassette/SCA121B-PSG.edf"}
        }
    )
    total_selected: int


class PatientInfo(BaseModel):
    number: int
    age: Optional[int] = None
    sex: Optional[str] = None
    num_recordings: Optional[int] = None


class AllPatientsResponse(BaseModel):
    patients: Dict[str, PatientInfo]


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


# Input for the prediction endpoint
class PredictionInput(BaseModel):
    features: List[float] = Field(
        ..., json_schema_extra={"example": [0.1, 0.5, 0.2, 0.8, 1.5]}
    )
    feature_names: Optional[List[str]] = None


# Output for the prediction endpoint
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
