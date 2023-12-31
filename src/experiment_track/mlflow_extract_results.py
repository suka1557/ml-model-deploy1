import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)
from utils.aws_credentials import load_aws_credentials_into_memory
from mlflow.tracking import MlflowClient
import boto3
from utils.logger import logger

def get_best_run_details(experiment_name: str, evaluation_criteria: str):
    """
    Function to take in a MLFlow Experiment name and an evaluation criteria as string arguments
    It then searches through all the runs in that experiment to find the run which has the maximum value for the 
    evaluation criteria.

    Notes: to use this function, the evaluation criteria should be of the type: Higher the Better

    Arguments:
        experiment_name - str: Name of the MLFlow experiment in which to search for runs
        evaluation_criteria - str: Name of evaluation criteria to choose best parameter combinations. 
                                    It should be same as one of the metrics logged in the mlflow experiment

    Returns:
        best_parameters - dict: containing key value pairs for best parameters to fit the model with
    """
    load_aws_credentials_into_memory() #This will make sure to load api keys into environment variables

    try:
        client = MlflowClient(tracking_uri=os.environ["MYSQL_URI"])
        exp = client.get_experiment_by_name(name=experiment_name)

        # extract params/metrics data for run `test_run_id` in a single dict 
        runs = client.search_runs(experiment_ids=[exp.experiment_id])

        current_best_metric = None
        best_parameters = None
        best_run_id = None

        # Iterate through the runs and extract parameters
        for run in runs:
            run_id = run.info.run_id
            parameters = client.get_run(run_id=run_id).data.params
            metrics = client.get_run(run_id=run_id).data.metrics

            if current_best_metric is None:
                current_best_metric = metrics[evaluation_criteria]
                best_parameters = parameters
                best_run_id = run_id
            else:
                if metrics[evaluation_criteria] > current_best_metric:
                    current_best_metric = metrics[evaluation_criteria]
                    best_parameters = parameters
                    best_run_id = run_id

        logger.info(f"Successfully extracted best run parameters for experiment - {experiment_name}: Best Run Id - {best_run_id}")

    except Exception as e:
        logger.error(f"Failed to get best run parameters from experiment: Error - {e}")
        raise Exception(f"{e}")

    return best_parameters, best_run_id, current_best_metric
