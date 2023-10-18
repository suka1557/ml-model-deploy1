import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)
import pandas as pd
from src.feature_engineering.feature_scaler import ScaleInputs
from utils.reader import read_yaml
from utils import logger
configs = read_yaml(os.path.join(PROJECT_ROOT, 'config.yaml'))

def get_predictions(input_df: pd.DataFrame, pca_decomposer, rf_model) -> float:
    """
    Function which will be called in the api request to provide predictions on given dataframe

    Arguments:
        input_df - pd.DataFrame: input dataframe of 1 X 784 dimension of numbers

    Returns:
        prediction - int: predicted class value between 0 - 9
    """

    try:    
        #scale inputs
        sc = ScaleInputs(MAX_VALUE=configs['MAX_PIXEL_VALUE'])
        input_df = sc.scale_input(input_df=input_df)
        

        #apply pca
        input_df = pd.DataFrame(pca_decomposer.transform(input_df))

        #make prediction
        prediction = float(rf_model.predict(input_df)[0])

        logger.info("Ran prediction pipeline for given input dataframe")

    except Exception as e:
        logger.error(f"Failed to make predictions: Error - {e}")
        raise Exception(f"{e}")

    return prediction

