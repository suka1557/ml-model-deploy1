import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
from itertools import product


import mlflow
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, precision_score
from dotenv import load_dotenv
import boto3
from src.model_training.train_test_split import get_train_test_split
from src.data_processing.data_loader import DataLoader
from src.feature_engineering.feature_scaler import ScaleInputs
from src.feature_selection.feature_selection_pca import get_selected_components_df
from src.experiment_track.mlflow_setup import set_up_mlflow_tracking

from utils.aws_credentials import load_aws_credentials_into_memory

load_dotenv(os.path.join(PROJECT_ROOT, 'config.env'))
load_dotenv(os.path.join(PROJECT_ROOT, 'secrets.env'))

if __name__ == '__main__':
    data_reader = (DataLoader(file_name=os.getenv("IMAGE_DATA_FILE"), project_root=PROJECT_ROOT))
    image_data = data_reader.read_data()
    image_data, target = data_reader.extract_input_and_target(image_data)

    #scale images
    image_data = ScaleInputs().scale_input(image_data)


    #PARAMETERS TO TRY
    # NO_COMPONENTS, NO_ESTIMATORS
    hyperparameters_dict = {
        "no_components": [30,50,75, 100],
        "n_estimators": [10,20,50, 100]
    }

    #SET UP MLFLOW EXPERIMENT
    #LOAD CREDENTIALS
    load_aws_credentials_into_memory()

    #SET UP TRACKING
    EXPERIMENT_NAME = 'random_forest_test2'
    experiment_id = set_up_mlflow_tracking(experiment_name=EXPERIMENT_NAME)

    print(experiment_id)

    for param_combination in list(product(*hyperparameters_dict.values())):
        current_parameters = dict(zip(hyperparameters_dict.keys(), param_combination))
        print(current_parameters)

        with mlflow.start_run(experiment_id=experiment_id):

            #get pca components
            components_df = get_selected_components_df(image_data, no_components=current_parameters['no_components'])

            #define train and valid set
            train_x,  val_x, train_y, val_y = get_train_test_split(input_df=components_df, target=target, test_size=0.2, maintain_class_balance=True)

            #define and fit random forest classifier
            clf = RandomForestClassifier(n_estimators=current_parameters['n_estimators'], random_state=42)
            clf.fit(train_x, train_y)

            #get prediction on validation
            val_y_pred = clf.predict(val_x)
            train_y_pred = clf.predict(train_x)

            #calculate f1 scores
            val_score= f1_score(val_y, val_y_pred, average='weighted')
            train_score = f1_score(train_y, train_y_pred, average='weighted')

            #print f1 score
            mlflow.log_params(current_parameters)
            mlflow.log_metric("f1_score_validation", val_score)
            mlflow.log_metric("f1_score_training", train_score)


