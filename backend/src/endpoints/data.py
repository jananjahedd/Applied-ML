"""Shared data store for the application."""

from typing import Dict

from src.utils.patient import Patient

patients: Dict[int, Patient] = {}
selected_filenames: Dict[int, str] = {}
