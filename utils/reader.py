import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)
import yaml
from utils.logger import logger
from ensure import ensure_annotations

@ensure_annotations
def read_yaml(file_path: str) -> dict:
    """
    Function to read data from YAML file and return data in form of a dict

    Arguments:
        file_path - str: location of the YAML file

    Returns:
        file_data - dict: dictionary containing key value pairs of items read from the file

    """
    try:
        with open(file_path, 'r') as file:
            file_data = yaml.safe_load(file)
        logger.info(f" Sucessfully read the configuration from {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to read the configurations from {file_path} , Error: {e}")
        raise FileNotFoundError(f"{e}")

    return file_data

if __name__ == '__main__':
    file_data = ( read_yaml(os.path.join(PROJECT_ROOT, 'config.yaml')) )
    print(type(file_data['MAINTAIN_CLASS_BALANCE']))