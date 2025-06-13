"""Schemas for recording data, including patient information and recording summaries."""

from typing import Dict

from pydantic import BaseModel, Field


class Patient(BaseModel):
    """Schema representing patient information associated with a recording."""

    number: int = Field(..., description="Internal numeric ID of the patient", json_schema_extra={"example": 10})
    age: int = Field(..., description="Age of the patient", json_schema_extra={"example": 45})
    sex: str = Field(..., description="Sex of the patient", json_schema_extra={"example": "Male"})


class RecordingSummary(BaseModel):
    """Schema representing a summary of a recording, including paths and study type."""

    recording_path: str = Field(
        ...,
        description="Path to the recording file",
        json_schema_extra={"recording_path": "example-data/sleep-cassette/SC4171E0-PSG.edf"},
    )
    annotation_path: str = Field(
        ...,
        description="Path to the annotation file",
        json_schema_extra={"annotation_path": "example-data/sleep-cassette/SC4171E0-Hypnogram.edf"},
    )
    study_type: str = Field(
        ..., description="Type of study (Cassette or Telemetry)", json_schema_extra={"study_type": "Cassette"}
    )


class Recording(BaseModel):
    """Schema representing a recording with detailed patient information."""

    recording_path: str = Field(
        ...,
        description="Path to the recording file",
        json_schema_extra={"recording_path": "example-data/sleep-cassette/SC4171E0-PSG.edf"},
    )
    annotation_path: str = Field(
        ...,
        description="Path to the annotation file, if available",
        json_schema_extra={"annotation_path": "example-data/sleep-cassette/SC4171E0-Hypnogram.edf"},
    )
    study_type: str = Field(
        ..., description="Type of study (Cassette or Telemetry)", json_schema_extra={"study_type": "Cassette"}
    )
    night: int = Field(..., description="Night number of the recording", json_schema_extra={"night": 1})
    patient: Patient = Field(
        ...,
        description="Patient information associated with the recording",
        json_schema_extra={"patient": {"number": 10, "age": 45, "sex": "Male"}},
    )


class AvailableRecordings(BaseModel):
    """Schema representing available recordings, categorized by type."""

    cassette_files: Dict[int, RecordingSummary] = Field(
        ...,
        description="Cassette files, file_path and annotations_path",
        json_schema_extra={
            "example": {
                "1": {
                    "recording_path": "example-data/sleep-cassette/SC4171E0-PSG.edf",
                    "annotation_path": "example-data/sleep-cassette/SC4171E0-Hypnogram.edf",
                },
            }
        },
    )
    telemetry_files: Dict[int, RecordingSummary] = Field(
        ...,
        description="Telemetry files, file_path and annotations_path",
        json_schema_extra={
            "example": {
                "1": {
                    "recording_path": "example-data/sleep-telemetry/ST70001N0-PSG.edf",
                    "annotation_path": "example-data/sleep-telemetry/ST70001N0-Hypnogram.edf",
                },
            }
        },
    )
