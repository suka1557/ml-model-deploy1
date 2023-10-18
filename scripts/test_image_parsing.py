import os
import sys
PROJECT_ROOT = os.path.abspath('./')
sys.path.append(PROJECT_ROOT)


filename = '/Users/sukant.kumar/Downloads/test_image.png'

import cv2 
import pandas as pd

# Load sample image
test_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


# # Format Image
img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)
img_array = (img_resized.flatten())
print(img_array.shape)
img_array  = pd.DataFrame( img_array.reshape(-1,1).T )
img_array.columns = ['pixel'+str(i) for i in range(784)]

#Load models
from src.evaluation.model_loader import load_rf_model_and_pca_decomposer
from src.evaluation.model_prediction import get_predictions


#Load model in global memory
pca, model = load_rf_model_and_pca_decomposer()
print("Loaded model from s3 bucket")

prediction = get_predictions(img_array, pca, model)
print(type(prediction))


