"""Patient class."""

from enum import Enum
from os.path import basename, isfile
from typing import Any, Dict

import mne

from src.utils.logger import get_logger
from src.utils.recording import Recording

logger = get_logger(__name__)


class Sex(Enum):
    """Sex enumeration."""

    MALE = "Male"
    FEMALE = "Female"


class Patient:
    """Patient class.

    Attributes:
        number (int): Patient number
        age (int): Age of the patient
        sex (Sex): Sex of the patient
        recordings Dict[Recording]: Dictionary of recordings associated with the patient
    """

    def __init__(self, number: int, age: int, sex: str, recordings: Dict[int, Recording]):
        """Initialize the Patient class.

        Args:
            number (int): Patient number
            age (int): Age of the patient
            sex (str): Sex of the patient
            recordings (Dict[Recording]): Dictionary of recordings associated with the patient
        """
        try:
            sex_enum = Sex(sex.capitalize())
        except ValueError:
            raise ValueError(f"Invalid sex value: '{sex}'.")

        self.number = number
        self.age = age
        self.sex = sex_enum
        self.recordings = recordings

    def __repr__(self) -> str:
        """Return a string representation of the Patient class.

        Returns:
            str: String representation of the Patient class
        """
        return f"Patient(number={self.number}, age={self.age}, sex={self.sex.value})"

    def __str__(self) -> str:
        """Return a string representation of the Patient class.

        Returns:
            str: String representation of the Patient class
        """
        return f"Patient #{self.number}: Age-{self.age}, Sex-{self.sex.value}"

    def dict(self) -> Dict[str, Any]:
        """Convert Patient to dictionary for JSON serialization."""
        return {
            "number": self.number,
            "age": self.age,
            "sex": self.sex.value,
            "recordings": {k: v.dict() for k, v in self.recordings.items()},
        }


def patient_from_filepath(file_path: str) -> Patient:
    """Extract the patient information from the file_path.

    Args:
        file_path (str): The file path.

    Returns:
        Patient: The patient information.
    """
    # check if the file exists
    if isfile(file_path) is False:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    raw = mne.io.read_raw_edf(file_path, preload=True)
    patient_info = raw.info["subject_info"]

    # Get the patient name from the filename
    file_name = basename(file_path)

    if file_name.startswith("ST7"):  # Telemetry
        # Files are named in the form ST7ssNJ0-PSG.edf where ss is the
        # subject number, and N is the night.
        number = file_name.split("ST7")[1][:2]

    elif file_name.startswith("SC4"):  # Cassette
        # Files are named in the form SC4ssNEO-PSG.edf where ss is the
        # subject number, and N is the night.
        number = file_name.split("SC4")[1][:2]

    else:
        logger.error("Unknown file name format")
        raise ValueError("Unknown file name format")

    return Patient(
        number=int(number),
        age=patient_info["last_name"].split("yr")[0],
        sex=patient_info["first_name"],
        recordings={1: Recording(file_path)},
    )
