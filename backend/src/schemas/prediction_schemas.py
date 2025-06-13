from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.schemas.model_schemas import DetailedMetrics


class PredictionProcessingSummary(BaseModel):
    """Summarizes the results of a prediction task."""
    total_recording_time_hours: float = Field(..., description="Total duration of the recording in hours.")
    epoch_duration_seconds: int = Field(..., description="Duration of each processed segment (epoch) in seconds.")
    annotations_from_hypnogram: bool = Field(..., description="True if a hypnogram was used to create epochs.")
    features_extracted_per_epoch: int = Field(..., description="Number of features extracted from each epoch.")
    sleep_stage_distribution: Dict[str, int] = Field(..., description="Count of predicted segments for each sleep stage.")
    current_file_performance: Optional[DetailedMetrics] = Field(None, description="Performance metrics if ground truth was available for the processed file.")

class PredictionResponse(BaseModel):
    """The response model for a successful prediction request."""
    model_configuration_used: str = Field(..., description="The model configuration used for this prediction (e.g., 'eeg_emg_eog').")
    input_file_name: str = Field(..., description="Name of the recording file processed.")
    num_segments_processed: int = Field(..., description="Total number of segments (epochs) processed from the recording.")
    predictions: List[str] = Field(..., description="A list of predicted sleep stage labels for each segment.")
    prediction_ids: List[int] = Field(..., description="A list of numeric prediction IDs corresponding to the labels for each segment.")
    probabilities_per_segment: Optional[List[List[float]]] = Field(None, description="List of prediction probabilities for each class for each segment, if available.")
    class_labels_legend: Optional[Dict[int, str]] = Field(None, description="Legend mapping numeric prediction IDs to human-readable labels.")
    processing_summary: PredictionProcessingSummary = Field(..., description="A detailed summary of the processing and results.")





















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
