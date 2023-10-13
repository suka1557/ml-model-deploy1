import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd

import mlflow
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, precision_score
from dotenv import load_dotenv
import boto3
from src.model_training.train_test_split import get_train_test_split
from src.data_processing.data_loader import DataLoader
from src.feature_engineering.feature_scaler import ScaleInputs
from src.feature_selection.feature_selection_pca import get_selected_components_df

load_dotenv(os.path.join(PROJECT_ROOT, 'config.env'))

if __name__ == '__main__':
    data_reader = (DataLoader(file_name=os.getenv("IMAGE_DATA_FILE"), project_root=PROJECT_ROOT))
    image_data = data_reader.read_data()
    image_data, target = data_reader.extract_input_and_target(image_data)

    #scale images
    image_data = ScaleInputs().scale_input(image_data)

    #get pca components
    components_df = get_selected_components_df(image_data, no_components=50)

    #define train and valid set
    train_x,  val_x, train_y, val_y = get_train_test_split(input_df=components_df, target=target, test_size=0.2, maintain_class_balance=True)
    print(train_x.shape, train_y.shape)
    print(val_x.shape, val_y.shape)

    #define and fit random forest classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(train_x, train_y)

    #get prediction on validation
    val_y_pred = clf.predict(val_x)
    train_y_pred = clf.predict(train_x)

    #calculate f1 scores
    val_score= f1_score(val_y, val_y_pred, average='weighted')
    train_score = f1_score(train_y, train_y_pred, average='weighted')

    #print f1 score
    print(f" F1 Score on validation set: {val_score}")
    print(f" F1 Score on training set: {train_score}")


