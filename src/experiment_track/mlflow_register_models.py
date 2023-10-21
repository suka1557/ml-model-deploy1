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

configs = read_yaml( os.path.join(PROJECT_ROOT, 'config.yaml') )

def get_top_candidate_run( evaluation_criteria=configs['EVALUATION_SCORE_PROD']):

    #Load credntials into memory
    load_aws_credentials_into_memory()

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
        print(exp.name)
        print(run_id, eval_metric)
        run_ids.append(run_id)
        scores.append(eval_metric)

    return run_ids[ np.argmax(scores) ]


def get_candidates_for_registration(prod_run_id = configs['OPTION_RUN_ID_PRODUCTION'] 
                                    , stage_run_id =configs['OPTION_RUN_ID_STAGING']):
    

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


if __name__ == '__main__':
    print(configs['MAINTAIN_CLASS_BALANCE'] is True)
    # print(get_candidates_for_registration())
