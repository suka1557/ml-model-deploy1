import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)

from src.experiment_track.mlflow_register_models import get_candidates_for_registration, register_models
from utils.aws_credentials import load_aws_credentials_into_memory
import mlflow
from utils.reader import read_yaml

configs = read_yaml(os.path.join(PROJECT_ROOT, 'config.yaml'))

if __name__ == '__main__':
    prod_id, stage_id = get_candidates_for_registration()


    # register_models(run_id=prod_id, 
    #                 environment_name='Production', 
    #                 model_name=configs['MODEL_NAME_IN_MODEL_REGISTRY'])
    
    register_models(run_id=stage_id, 
                    environment_name='Staging', 
                    model_name=configs['MODEL_NAME_IN_MODEL_REGISTRY'])
