import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)
from dotenv import load_dotenv
from src.data_processing.data_loader import DataLoader
from src.feature_engineering.feature_scaler import ScaleInputs
from src.model_training.model_random_forest_experiment import random_forest_model_experiments
from src.experiment_track.mlflow_extract_results import get_best_experiment_details
from utils.reader import read_yaml

load_dotenv(os.path.join(PROJECT_ROOT, 'config.env'))
load_dotenv(os.path.join(PROJECT_ROOT, 'secrets.env'))
configs = read_yaml(os.path.join(PROJECT_ROOT, 'config.yaml'))

#PARAMETERS TO TRY
# NO_COMPONENTS, NO_ESTIMATORS
hyperparameters_dict = configs['parameter_dict']
# {
#     "no_components": [50,60,80],
#     "n_estimators": [20,50, 100],
#     "max_depth":[2,5,8],    
# }


#SET UP TRACKING
EXPERIMENT_NAME = configs['EXPERIMENT_NAME']

#TRAIN, VAL SPLIT
TRAIN_VALID_SPLIT_RATIO = configs['TRAIN_VALID_SPLIT_RATIO']

#MAINTAIN CLASS BALANCE
MAINTAIN_CLASS_BALANCE = configs['MAINTAIN_CLASS_BALANCE']

#EVALUATION SCORE
EVALUATION_SCORE = configs['EVALUATION_SCORE']



if __name__ == '__main__':
    #Data Ingestion
    data_reader = (DataLoader(file_name=configs['IMAGE_DATA_FILE'], project_root=PROJECT_ROOT))
    image_data = data_reader.read_data()
    image_data, target = data_reader.extract_input_and_target(image_data)

    #Scale inputs - Data Transformation
    sc = ScaleInputs(MAX_VALUE=configs['MAX_PIXEL_VALUE'])
    image_data = sc.scale_input(input_df=image_data)

    # #Run experiments
    random_forest_model_experiments(input_df=image_data,
                                    target=target,
                                    hyperparameters_dict=hyperparameters_dict,
                                    exp_name=EXPERIMENT_NAME,
                                    test_set_size=TRAIN_VALID_SPLIT_RATIO,
                                    class_distributed=MAINTAIN_CLASS_BALANCE)
    
    print('Completed Experiments for all parameter combinations using MLFlow')
    
    #Get best parameters for this experiment to be run on mlflow
    best_parameters_dict = get_best_experiment_details(experiment_name=EXPERIMENT_NAME, evaluation_criteria=EVALUATION_SCORE)

    #MLFLOW Returns all values as string
    #Convert them to float wherever possible
    for k,v in best_parameters_dict.items():
        try: 
            best_parameters_dict[k] = int(v)
        except:
            continue

    #converting single values into iterables lists, which need to bes paased to itertools while doing model fitting
    best_parameters_dict = {k:[v] for k,v in best_parameters_dict.items()}
    print(f"Best Parameters: {best_parameters_dict}")

    # Train a model with best parameters and store the model to S3 bucket
    FINAL_EXPERIMENT_NAME = 'FINAL_' + EXPERIMENT_NAME
    random_forest_model_experiments(input_df=image_data,
                                    target=target,
                                    hyperparameters_dict=best_parameters_dict,
                                    exp_name=FINAL_EXPERIMENT_NAME,
                                    test_set_size=0.0,
                                    class_distributed=MAINTAIN_CLASS_BALANCE,
                                    log_model=True)
    
    print("Trained final model with best parameter combination and saved model to S3 bucket")

    


