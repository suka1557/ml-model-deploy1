import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)
import mlflow
import boto3
from utils.logger import logger
from utils.aws_credentials import load_aws_credentials_into_memory

#Load credentials into memory
load_aws_credentials_into_memory()

#SET UP MLFLOW EXPERIMENT

def set_up_mlflow_tracking(experiment_name: str, tracking_server: str = os.environ['MYSQL_URI'],
                           artifact_server: str = os.environ["ARTIFACT_URI"]):
    """
    This function set up the server for experiment tracking and articat storage

    Arguments:
        experiment_name - str: Name of the experiment
        tracking_server - str: Location of TRACKING SERVER, Usually a SQL Server
        artifact_server - str: Location of ARTIFACT SERVER, USually S3/Blob or any other file storage
        
    
    Returns:
        None
    """

    #SET UP TRACKING
    try:
        mlflow.set_tracking_uri(tracking_server)
        experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_server)
        logger.info(f"Set up mlflow experiment with name: {experiment_name} and id: {experiment_id}")

    except Exception as e:
        logger.error(f"Failed to set up mlflow experiment : Error - {e}")
        raise Exception(f"{e}")

    return experiment_id

