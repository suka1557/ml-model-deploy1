import os
import numpy as np
import pandas as pd
import sys
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from dotenv import load_dotenv
import boto3

PROJECT_ROOT_PATH = os.path.abspath("./")
sys.path.append(PROJECT_ROOT_PATH)

load_dotenv(os.path.join(PROJECT_ROOT_PATH, "secrets.env" ) )

#Setting up credentials
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["region_name"] = os.getenv("region_name")

#Setting tracking uri
mlflow.set_tracking_uri(os.getenv("MYSQL_TRACKING_URI"))
#Setting artifact uri
experiment_id = mlflow.create_experiment('test_mlflow_8', artifact_location=os.getenv("ARTIFACT_URI"))

#running experiement
mlflow.start_run(run_name="test_run", experiment_id=experiment_id)
mlflow.log_param("my", "param")
mlflow.log_metric("score", 100)
try:
    mlflow.log_artifact(os.path.join(PROJECT_ROOT_PATH, "config.env"))
    print("Loaded file to S3 bucket")
except Exception as e:
    print("Could not load file to S3 bucket")
    print(e)

mlflow.end_run()
