"""Main script for the whole machine learning pipeline."""
from src.data.preprocessing import preprocess_pipeline
from src.data.loso_cv import loso_cv_main
from src.models.modular_svm import main_svm
from src.utils.logger import get_logger
from src.utils.paths import (get_repo_root, get_results_dir,
                             get_splits_data, get_processed_data_dir)

# setup logger
logger = get_logger("MAIN")


def preprocessing_step():
    """Run the preprocessing pipeline."""
    logger.info("Starting preprocessing pipeline...")
    preprocess_pipeline()
    logger.info("Preprocessing completed.")
    logger.info("Preprocessed data is saved in: "
                f"{get_processed_data_dir()}. ")


def loso_cv_step():
    """Run the Leave-One-Subject-Out cross-validation."""
    logger.info("Starting Leave-One-Subject-Out cross-validation...")
    loso_cv_main()
    logger.info("Leave-One-Subject-Out cross-validation completed.")
    logger.info("Splits are saved in: "
                f"{get_splits_data()}. ")


def svm_step():
    """Function for testing the SVM model."""
    logger.info(f"Project Root: {get_repo_root()}")
    logger.info(f"Splits Directory used by SVM: {get_splits_data()}")
    if not get_splits_data().exists():
        logger.error(
            f"SPLITS_DIR does not exist: {get_splits_data()}."
            "Please create it or check path."
        )
        logger.error(
            "Ensure that `loso_cv.py` has run and saved its output "
            "into this directory."
        )
    else:
        main_svm()
    logger.info("SVM model training and evaluation completed.")
    logger.info("Results are saved in: "
                f"{get_results_dir()}. ")


def main():
    """Main function to run the entire pipeline."""
    logger.info("Starting the machine learning pipeline...")
    preprocessing_step()
    loso_cv_step()
    svm_step()
    logger.info("Machine learning pipeline completed.")


if __name__ == "__main__":
    main()
