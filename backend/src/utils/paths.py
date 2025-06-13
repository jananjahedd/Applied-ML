"""Module for retrieving all the important paths.

This module is context-aware and will define the correct root path
whether it is running inside a Docker container or a local machine.
"""

import os

if os.path.isdir("/app"):
    # in docker
    APP_ROOT = "/app"
else:
    # running locally
    THIS_FILE = os.path.dirname(os.path.abspath(__file__))
    APP_ROOT = os.path.abspath(os.path.join(THIS_FILE, "..", "..", ".."))

# definition of all paths for the app
BACKEND_DIR = os.path.join(APP_ROOT, "backend")
SRC_DIR = os.path.join(BACKEND_DIR, "src")
DATA_DIR = os.path.join(APP_ROOT, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
SPLITS_DATA_DIR = os.path.join(DATA_DIR, "data_splits")
MODELS_DIR = os.path.join(SRC_DIR, "models")
FEATURES_DIR = os.path.join(SRC_DIR, "features")
LOGS_DIR = os.path.join(BACKEND_DIR, "logs")
RESULTS_DIR = os.path.join(APP_ROOT, "results")


def get_repo_root() -> str:
    """Get the absolute path to the root repository."""
    return APP_ROOT


def get_backend_root() -> str:
    """Get the absolute path to the backend directory."""
    return BACKEND_DIR


def get_src_dir() -> str:
    """Get the absolute path to the src directory."""
    return SRC_DIR


def get_data_dir() -> str:
    """Get the absolute path to the data directory."""
    return DATA_DIR


def get_processed_data_dir() -> str:
    """Get the absolute path to the processed data directory."""
    return PROCESSED_DATA_DIR


def get_splits_data() -> str:
    """Get the absolute path to the splitted data directory."""
    return SPLITS_DATA_DIR


def get_logs_dir() -> str:
    """Get the absolute path to the logs directory."""
    return LOGS_DIR


def get_results_dir() -> str:
    """Get the absolute path to the results directory."""
    return RESULTS_DIR
