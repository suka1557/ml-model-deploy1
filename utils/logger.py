import logging
import os
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath("./")
CONFILE_FILE = os.path.join(PROJECT_ROOT, 'config.env')

load_dotenv(CONFILE_FILE)


def setup_logger(log_file_path):
    # Create a logger
    logger = logging.getLogger(os.getenv("LOGGER_NAME"))
    logger.setLevel(logging.INFO)

    # Create a file handler and set the log level
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create a stream handler for console output and set the log level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s -- %(module)s -- %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the file and stream handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

#Create the logger
log_file_path = os.path.join(PROJECT_ROOT, os.getenv("LOG_FILENAME"))
logger = setup_logger(log_file_path)


