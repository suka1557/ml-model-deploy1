import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)

from utils.reader import read_yaml
import mlflow
from utils.aws_credentials import load_aws_credentials_into_memory

configs = read_yaml(os.path.join(PROJECT_ROOT, 'config.yaml'))

def load_rf_model_and_pca_decomposer(run_id: str = configs['MLFLOW_RUN_ID'],
                                     model_name: str = configs['RF_MODEL_NAME'],
                                     pca_name: str = configs['PCA_DECOMPOSER']):
    """
    This function takes in a MLFLOW RUN ID and name of model and pca decomposer
    It connects to MLFlow artifact store and loads model from there

    Arguments:
        run_id - str: aplhanumeric run id of mlflow experiment
        model_name - str: name with which the model is stored
        pca_name - str: name with which pca decomposer is stored

    Returns:
        sklearn random forest model and pca decomposer
    
    """
    load_aws_credentials_into_memory()
    mlflow.set_tracking_uri(os.getenv("MYSQL_URI"))

    logged_model_path = f'runs:/{run_id}/{model_name}'
    logged_pca_decomposer_path = f'runs:/{run_id}/{pca_name}'

    logged_model = mlflow.sklearn.load_model(logged_model_path)
    logged_pca_decomposer = mlflow.sklearn.load_model(logged_pca_decomposer_path)

    return logged_pca_decomposer, logged_model


