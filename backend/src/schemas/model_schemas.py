"""Schemas for model configurations and metadata.

This is for the sleep stage classification application.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelConfig(str, Enum):
    """Enumeration for the available model configurations."""

    EEG = "eeg"
    EEG_EMG = "eeg_emg"
    EEG_EOG = "eeg_eog"
    EEG_EMG_EOG = "eeg_emg_eog"


class AvailableModels(BaseModel):
    """Response model for the list of available models."""

    available_configurations: Dict[ModelConfig, Dict[str, Any]] = Field(
        ...,
        description=("Dictionary of available model " + "configurations with their status and paths"),
        json_schema_extra={
            "example": {
                "eeg": {"available": True, "path": "results/svm_model_eeg.joblib"},
                "eeg_emg": {"available": True, "path": "results/svm_model_eeg_emg.joblib"},
                "eeg_eog": {"available": False, "path": "results/svm_model_eeg_eog.joblib"},
                "eeg_emg_eog": {"available": True, "path": "results/svm_model_eeg_emg_eog.joblib"},
            }
        },
    )
    default_configuration: ModelConfig = Field(
        ModelConfig.EEG,
        description="Default model configuration to use if none is specified",
        json_schema_extra={"example": "eeg"},
    )


class ModelPerformanceSummary(BaseModel):
    """A brief summary of a model's performance."""

    test_accuracy: Optional[float] = Field(None, description="Accuracy on the final test set.")
    test_macro_f1_score: Optional[float] = Field(None, description="Macro F1-score on the final test set.")


class ModelDetails(BaseModel):
    """Detailed metadata for a single model configuration."""

    config_name: ModelConfig = Field(..., description="The name of the model configuration.")
    modalities_used: List[str] = Field(
        ..., description=("A list of the physiological signals " + "the model uses (e.g., 'eeg', 'emg').")
    )
    expected_features_count: int = Field(..., description="The number of features the model expects as input.")
    class_labels_legend: Dict[int, str] = Field(
        ..., description="Mapping of numeric prediction IDs to human-readable sleep stage labels."
    )
    performance_summary: ModelPerformanceSummary = Field(
        ..., description="A summary of the model's key performance metrics."
    )
