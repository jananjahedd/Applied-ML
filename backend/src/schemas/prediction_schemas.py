"""Prediction response schema for EDF files."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictEDFResponse(BaseModel):
    """Response schema for predictions made on EDF files."""

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
                "sleep_stage_distribution": {"Wake": 120, "N1": 45, "N2": 180, "N3": 90, "REM": 85},
            }
        },
    )
