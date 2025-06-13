"""Recording class."""

from enum import Enum
from os import listdir
from os.path import basename, dirname
from typing import Any, Dict

from mne.io import read_raw_edf
from src.utils.plotting import plot_signals_mne


class StudyType(Enum):
    """Study type enumeration."""

    SC = "Cassette"
    ST = "Telemetry"


class Recording:
    """Recording class.

    Attributes:
        file_path (str): File path of the recording
        anno_path (str): Path to the annotation file
        study_type (StudyType): Type of study (Cassette or Telemetry)
        patient_number (int): Patient number
        night (int): Night number
    """

    def __init__(self, file_path: str):
        """Initialize the Recording class.

        Args:
            file_path (str): File path of the recording
        """
        self.file_path = file_path
        file_name = basename(file_path)
        directory = dirname(file_path)

        if file_name.startswith("ST7"):  # Telemetry
            # Files are named in the form ST7ssNJ0-PSG.edf where ss is the
            # subject number, and N is the night.
            self.study_type = StudyType.ST
            self.patient_number = int(file_name.split("ST7")[1][:2])
            self.night = int(file_name.split("ST7")[1][2:3])
            self.anno_path = next(
                directory + "/" + f
                for f in listdir(directory)
                if f.startswith(f"ST7{self.patient_number:02d}{self.night}") and f.endswith("Hypnogram.edf")
            )

        elif file_name.startswith("SC4"):  # Cassette
            # Files are named in the form SC4ssNEO-PSG.edf where ss is the
            # subject number, and N is the night.
            self.study_type = StudyType.SC
            self.patient_number = int(file_name.split("SC4")[1][:2])
            self.night = int(file_name.split("SC4")[1][2:3])
            self.anno_path = next(
                directory + "/" + f
                for f in listdir(directory)
                if f.startswith(f"SC4{self.patient_number:02d}{self.night}") and f.endswith("Hypnogram.edf")
            )

        else:
            raise ValueError("Unknown file name format")

    def __str__(self) -> str:
        """Return a string representation of the Recording class.

        Returns:
            str: String representation of the Recording class
        """
        return f"Recording: {self.study_type.value}, Patient-" f"{self.patient_number}, Night-{self.night}"

    def dict(self) -> Dict[str, Any]:
        """Convert Recording to dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the Recording class
        """
        return {
            "file_path": self.file_path,
            "anno_path": self.anno_path,
            "study_type": self.study_type.value,
            "patient_number": self.patient_number,
            "night": self.night,
        }

    def visualize(self) -> None:
        """Visualize the recording."""
        raw_data = read_raw_edf(self.file_path, preload=True, verbose=False)
        plot_signals_mne(recording=self, raw=raw_data, annotations=True)


def is_valid_edf_name(file_name: str) -> bool:
    """Check if the file name is a valid EDF file name.

    Args:
        file_name (str): The file name to check.

    Returns:
        bool: True if the file name is valid, False otherwise.
    """
    return file_name.startswith(("ST7", "SC4")) and file_name.endswith(".edf")


def is_valid_annotation_name(file_name: str) -> bool:
    """Check if the file name is a valid annotation file name.

    Args:
        file_name (str): The file name to check.

    Returns:
        bool: True if the file name is valid, False otherwise.
    """
    return file_name.endswith("Hypnogram.edf") and any(file_name.startswith(prefix) for prefix in ("ST7", "SC4"))
