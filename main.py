"""Main script for the whole machine learning pipeline."""
from src.data.preprocessing import preprocess_pipeline
from src.data.loso_cv import load_preprocessed_data, prepare_data, splitting
from src.models.modular_svm import main_svm
from src.utils.logger import get_logger


# setup logger
logger = get_logger("MAIN")

# directory management
DATA_DIR = 

def preprocessing_step():




def main():


if __name__ == "__main__":
    main()
