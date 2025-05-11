"""Patient class."""

from enum import Enum


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
    """

    def __init__(self, number: int, age: int, sex: str):
        """Initialize the Patient class.

        Args:
            number (int): Patient number
            age (int): Age of the patient
            sex (str): Sex of the patient
        """
        try:
            sex_enum = Sex(sex.capitalize())
        except ValueError:
            raise ValueError(f"Invalid sex value: '{sex}'.")

        self.number = number
        self.age = age
        self.sex = sex_enum

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
