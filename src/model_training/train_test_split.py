import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)
from sklearn.model_selection import train_test_split
from typing import Union
import pandas as pd
from utils.logger import logger

def get_train_test_split(input_df: pd.DataFrame, target: pd.Series, test_size: float, 
                         maintain_class_balance: bool) -> Union[
    pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series
] :
    """
    Function to split data into different training set and validation sets

    Arguments:
        input_df: Pandas dataframe containing all the input features
        target: Pandas series containing the target labels
        test_size: percentage of datapoints to keep in validation dataset
        maintain_class_balance: Whether to consider target class distribution while splitting the dataset
    
    Returns:
        train_df, train_y: dataframe and target series for Training
        val_df, val_y: dataframe and target series for Validation
    """
    RANDOM_STATE = 102
    try:
        if maintain_class_balance:
            train_df, val_df, train_y, val_y = train_test_split(input_df, target, 
                                                                test_size=test_size, 
                                                                random_state=RANDOM_STATE, shuffle=True,
                                                                stratify=target)
        else:
            train_df, val_df, train_y, val_y = train_test_split(input_df, target, 
                                                                test_size=test_size, 
                                                                random_state=RANDOM_STATE, shuffle=True)
            
        logger.info(f"""Splitted into train and validation set - train set size: {train_df.shape} - valid set size: {val_df.shape} """)
        
    except Exception as e:
        logger.error(f"Error in splitting into train-val set: Error - {e}")
        raise Exception(f"{e}")

    return train_df,  val_df, train_y, val_y