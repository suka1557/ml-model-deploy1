import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from itertools import product

import mlflow
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score
from utils.logger import logger

from src.model_training.train_test_split import get_train_test_split
from src.feature_selection.feature_selection_pca import get_selected_components_df
from src.experiment_track.mlflow_setup import set_up_mlflow_tracking
from utils.aws_credentials import load_aws_credentials_into_memory

def random_forest_model_experiments(input_df: pd.DataFrame, target: pd.Series,
                        hyperparameters_dict: dict, exp_name: str,
                        test_set_size: float, class_distributed: bool,
                        log_model: bool = False):
    """
    This function takes in input training dataframe and target series.
    It then splits the dataset into training and testing set
    After that it starts an MLFlow experiment and conducts model fitting experiment for given set of hyperparameters
    
    """
    #SET UP MLFLOW EXPERIMENT
    #LOAD CREDENTIALS
    load_aws_credentials_into_memory()

    #SET UP TRACKING
    EXPERIMENT_NAME = exp_name
    experiment_id = set_up_mlflow_tracking(experiment_name=EXPERIMENT_NAME)

    try:
        for param_combination in list(product(*hyperparameters_dict.values())):
            current_parameters = dict(zip(hyperparameters_dict.keys(), param_combination))

            with mlflow.start_run(experiment_id=experiment_id):

                #get pca components
                components_df, pca_decomposer = get_selected_components_df(input_df=input_df, no_components=current_parameters['no_components'])

                #Remove no_components from parameter dict
                no_components = current_parameters['no_components']
                del current_parameters['no_components']

                #define train and valid set
                if test_set_size > 0:
                    train_x,  val_x, train_y, val_y = get_train_test_split(input_df=components_df, target=target, test_size=test_set_size, maintain_class_balance=class_distributed)
                else:
                    train_x, train_y = components_df, target
                    val_x, val_y = pd.DataFrame({}), pd.Series([]) #initialize to empty data structures

                #define and fit random forest classifier
                clf = RandomForestClassifier(random_state=42, n_jobs=-1) 
                clf.set_params(**current_parameters)
                clf.fit(train_x, train_y)

                #get prediction on training and calculate score
                train_y_pred = clf.predict(train_x)
                train_score = f1_score(train_y, train_y_pred, average='weighted')
                train_accuracy = accuracy_score(train_y, train_y_pred)           

                #LOGGING
                mlflow.log_params(current_parameters)
                mlflow.log_param("no_components", no_components)
                mlflow.log_metric("f1_score_training", train_score)
                mlflow.log_metric("accuracy_training", train_accuracy)

                #get prediction on validation and calculate score
                if len(val_x) > 0:
                    val_y_pred = clf.predict(val_x)
                    val_score= f1_score(val_y, val_y_pred, average='weighted')
                    val_accuracy = accuracy_score(val_y, val_y_pred)
                    mlflow.log_metric("f1_score_validation", val_score)
                    mlflow.log_metric("accuracy_validation", val_accuracy)

                if log_model:
                    mlflow.sklearn.log_model(clf, "best_random_forest_model")
                    mlflow.sklearn.log_model(pca_decomposer, "pca_decomposer")

        logger.info("Trained a random forest model for different hyperparameter combinations")

    except Exception as e:
        logger.error("Error in fitting Random Forest Models: Error - {e}")
        raise Exception(f"{e}")



