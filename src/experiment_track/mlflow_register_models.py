import os
import sys
PROJECT_ROOT = os.path.abspath('./')
sys.path.append(PROJECT_ROOT)
from datetime import datetime
from utils.aws_credentials import load_aws_credentials_into_memory
from utils.logger import logger
from utils.reader import read_yaml
import boto3
import mlflow
from mlflow.tracking import MlflowClient
from src.experiment_track.mlflow_extract_results import get_best_run_details
import numpy as np
from typing import Union

configs = read_yaml( os.path.join(PROJECT_ROOT, 'config.yaml') )

#Load credntials into memory
load_aws_credentials_into_memory()

#Set MLFLOW Tracking URI
mlflow.set_tracking_uri(os.environ['MYSQL_URI'])

def get_top_candidate_run( evaluation_criteria: str =configs['EVALUATION_SCORE_PROD']):

    """
    Function to search all mlflow FINAL experiments and find out the best run among them
    using the evaluation_criteria that is passed as an argument

    Arguments:
        evaluation_criteria : str - Criteria using which to compare model performance

    Returns:
        run_id: str - run id of the best run that's found
    """

    # Set the tracking URI to your desired backend (e.g., a local directory)
    client = MlflowClient(os.environ['MYSQL_URI'])

    # Find top 2 Final Experiments
    experiments = client.search_experiments(order_by=["last_update_time DESC"])
    experiments = [e for e in experiments if 'FINAL' in e.name]
    #Limit to Last 10 Final experiments
    experiments = experiments[ : 10]

    run_ids = []
    scores = []
    for exp in experiments:
        _, run_id, eval_metric = get_best_run_details(experiment_name=exp.name,
                                                      evaluation_criteria=evaluation_criteria)
        run_ids.append(run_id)
        scores.append(eval_metric)

    return run_ids[ np.argmax(scores) ]


def get_candidates_for_registration(prod_run_id = configs['OPTION_RUN_ID_PRODUCTION'] 
                                    , stage_run_id =configs['OPTION_RUN_ID_STAGING']) -> (str, str):
    """
    Function to take in a run id for prod and for staging
    If any of these ids are None, then it calls the function internally to 
    finds the best run id that is found in any of the FINAL experiments

    Arguments:
        prod_run_id : str or None
        stage_run_id: str or None

    Returns:
        prod_run_id, stage_run_id : (str, str) - RUN Ids that needs to registered as STAGING and PROD
    
    """
    if prod_run_id is None and stage_run_id is None:
        #find out 2 candidate ids
        best_run_id = get_top_candidate_run()
        prod_run_id, stage_run_id = best_run_id, best_run_id

    elif prod_run_id is None:
        #find out 1 candidate id for production
        prod_run_id = get_top_candidate_run()

    elif stage_run_id is None:
        #find out 1 candidate run id for staging
        stage_run_id = get_top_candidate_run()

    return prod_run_id, stage_run_id


def register_models(run_id: str,
                    environment_name: str,
                    model_name: str = configs['MODEL_NAME_IN_MODEL_REGISTRY'],
                    artifact_path: str = configs['RF_MODEL_NAME']):
    """
    This function receives run id that needs to be registered into the environment
    An option argument model_name is also passed which provides under which name model needs to be registered
    It calls on Mlflow client to register those models into model registry 
    and then transition those models to STAGING/PRODUCTION

    Arguments:
        run_id: str - MLFlow run id of the model
        environment_name: str - Staging or Production
        model_name: str - Name under which model will be registered
        artifact_path: str - artifact path for mlflow model storage
    """

    model_uri = f"runs:/{run_id}/{artifact_path}"

    # Set the tracking URI to your desired backend (e.g., a local directory)
    client = MlflowClient(tracking_uri=os.environ['MYSQL_URI'])

    #Add model to model registry
    new_model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

    #Transition the model to given environment
    client.transition_model_version_stage(name=new_model_details.name,
                                          version=new_model_details.version,
                                          stage=environment_name)