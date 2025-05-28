"""Module for retrieving all the important paths."""
import os


# get the directory of the current file
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
# get the root directory of the repository
ROOT = os.path.abspath(os.path.join(UTILS_DIR, '..', '..'))
# define the rest of the paths relative to the root
SRC_DIR = os.path.join(ROOT, 'src')
DATA_DIR = os.path.join(ROOT, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')
SPLITS_DATA_DIR = os.path.join(DATA_DIR, "data_splits")
NOTEBOOKS_DIR = os.path.join(ROOT, 'notebooks')
MODELS_DIR = os.path.join(SRC_DIR, "models")
UTILS_DIR = os.path.join(SRC_DIR, "utils")
SRC_DATA_DIR = os.path.join(SRC_DIR, "data")
FEATURES_DIR = os.path.join(SRC_DIR, "features")
LOGS_DIR = os.path.join(ROOT, "logs")
RESULTS_DIR = os.path.join(ROOT, "results")


for _dir in [SRC_DIR, DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DATA_DIR,
             NOTEBOOKS_DIR, MODELS_DIR, UTILS_DIR, SRC_DATA_DIR,
             FEATURES_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)


def get_repo_root() -> str:
    """Get the absolute path to the root repository."""
    return ROOT


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
