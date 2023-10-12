import os
import sys
PROJECT_ROOT = os.path.abspath('./')
sys.path.append(PROJECT_ROOT)
from utils.logger import logger

def print_logs():
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

if __name__ == "__main__":
    print(PROJECT_ROOT)
    print_logs()