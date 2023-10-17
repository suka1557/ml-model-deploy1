import os
import sys
PROJECT_ROOT = os.path.abspath('./')
sys.path.append(PROJECT_ROOT)


filename = '/Users/sukant.kumar/Downloads/test_image.png'

import cv2 
import matplotlib.pyplot as plt

# Load sample image
test_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


# # Format Image
img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
img_resized = cv2.bitwise_not(img_resized)
img_array = (img_resized.flatten())
print(img_array.shape)
img_array  = img_array.reshape(-1,1).T
print(img_array.shape)

