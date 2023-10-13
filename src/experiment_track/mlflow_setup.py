import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)

import mlflow
import boto3
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, 'config.env'))
load_dotenv(os.path.join(PROJECT_ROOT, 'secrets.env'))

#SET UP MLFLOW EXPERIMENT


def set_up_mlflow_tracking(experiment_name: str, tracking_server: str = os.getenv("MYSQL_URI"),
                           artifact_server: str = os.getenv("ARTIFACT_URI")):
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
    mlflow.set_tracking_uri(tracking_server)
    experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=artifact_server)

    return experiment_id

