from dotenv import load_dotenv
import mlflow
import os
import mysql.connector

load_dotenv('./secrets.env')

#Set up environment variables
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["region_name"] = os.getenv("AWS_REGION_NAME")
os.environ['MLFLOW_TRACKING_URI'] = os.getenv("MYSQL_URI")

artifact_uri = os.getenv("ARTIFACT_URI")

print('loaded secrets successfully')

mlflow.log_artifact(artifact_uri)

# Try to create an experiment (you can replace 'my_experiment' with your experiment name)
try:
    experiment_id = mlflow.create_experiment("test_experiment")
    print(f"Experiment ID: {experiment_id}")
    print("experiment created successfully")
except Exception as e:
    print(f"Error creating experiment: {e}")