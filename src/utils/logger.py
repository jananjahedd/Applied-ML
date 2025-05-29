"""Logger module for the project."""

import logging
import pathlib

try:
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    LOGS_DIR = PROJECT_ROOT / "logs"
except NameError:
    SCRIPT_DIR = pathlib.Path(".").resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    LOGS_DIR = PROJECT_ROOT / "logs"


def get_logger(name: str, log_file: str = "") -> logging.Logger:
    """Get a logger with the specified name and save it.

    :param name: the name of the logger.
    :param log_file: name of the log file.
    :return: the logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    if not logger.hasHandlers():
        formatter = logging.Formatter(
            fmt=("%(asctime)s - %(name)s - %(levelname)s - " + "%(module)s - %(funcName)s - %(message)s"),
            datefmt="%d-%m-%Y %H:%M:%S",
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating log directory {LOGS_DIR}: {e}")
            return logger

        effective_log_file = log_file if log_file else f"{name.replace('.', '_')}.log"
        log_path = LOGS_DIR / effective_log_file

        try:
            fh = logging.FileHandler(log_path, mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            # logger.info(f"Logging to file: {log_file_path}")
        except Exception as e:
            print(f"Error setting up file handler for {log_path}: {e}")

    return logger
