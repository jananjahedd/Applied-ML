from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


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
            description="Dictionary of available model configurations with their status and paths",
            json_schema_extra={
                "example": {
                    "eeg": {
                        "available": True,
                        "path": "results/svm_model_eeg.joblib"
                    },
                    "eeg_emg": {
                        "available": True,
                        "path": "results/svm_model_eeg_emg.joblib"
                    },
                    "eeg_eog": {
                        "available": False,
                        "path": "results/svm_model_eeg_eog.joblib"
                    },
                    "eeg_emg_eog": {
                        "available": True,
                        "path": "results/svm_model_eeg_emg_eog.joblib"
                    }
                }
            }
    )
    default_configuration: ModelConfig = Field(
        ModelConfig.EEG,
        description="Default model configuration to use if none is specified",
        json_schema_extra={"example": "eeg"}
    )


class ModelPerformanceSummary(BaseModel):
    """A brief summary of a model's performance."""
    test_accuracy: Optional[float] = Field(None, description="Accuracy on the final test set.")
    test_macro_f1_score: Optional[float] = Field(None, description="Macro F1-score on the final test set.")


class ModelDetails(BaseModel):
    """Detailed metadata for a single model configuration."""
    config_name: ModelConfig = Field(..., description="The name of the model configuration.")
    modalities_used: List[str] = Field(..., description="A list of the physiological signals the model uses (e.g., 'eeg', 'emg').")
    expected_features_count: int = Field(..., description="The number of features the model expects as input.")
    class_labels_legend: Dict[int, str] = Field(..., description="Mapping of numeric prediction IDs to human-readable sleep stage labels.")
    performance_summary: ModelPerformanceSummary = Field(..., description="A summary of the model's key performance metrics.")

class PerClassMetrics(BaseModel):
    """Performance metrics for a single class."""
    precision: float
    recall: float
    f1_score: float
    support: int

class ConfusionMatrix(BaseModel):
    """Confusion matrix structure."""
    matrix: List[List[int]]
    labels: List[str]

class OverallMetrics(BaseModel):
    """Detailed overall performance metrics."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1_score: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1_score: float
    roc_auc_macro: Optional[float]

class DetailedMetrics(BaseModel):
    """A full set of detailed metrics for a dataset split (train/val/test)."""
    dataset_size: int
    overall_metrics: OverallMetrics
    per_class_metrics: Dict[str, PerClassMetrics]
    confusion_matrix: Optional[ConfusionMatrix]
    class_distribution: Dict[str, int]


class OverfittingAnalysis(BaseModel):
    """Analysis of model overfitting and generalization."""
    accuracy_drop: float
    f1_drop: float
    overfitting_severity: str
    generalization_quality: str

class ModelPerformanceResponse(BaseModel):
    """The complete performance report for a model."""
    model_configuration: ModelConfig
    performance_summary: Dict[str, Dict[str, Any]]
    overfitting_analysis: Optional[OverfittingAnalysis]
    detailed_metrics: Dict[str, Optional[DetailedMetrics]]
