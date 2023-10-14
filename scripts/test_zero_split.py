import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)
from dotenv import load_dotenv
from src.data_processing.data_loader import DataLoader
from src.feature_engineering.feature_scaler import ScaleInputs
from src.model_training.train_test_split import get_train_test_split

if __name__ == '__main__':
    data_reader = (DataLoader(file_name=os.getenv("IMAGE_DATA_FILE"), project_root=PROJECT_ROOT))
    image_data = data_reader.read_data()
    image_data, target = data_reader.extract_input_and_target(image_data)

    train_x, val_x, train_y, val_y = get_train_test_split(input_df=image_data, target=target, test_size=0.0, maintain_class_balance=True)
    print(train_x.shape)
    print(val_x.shape)

