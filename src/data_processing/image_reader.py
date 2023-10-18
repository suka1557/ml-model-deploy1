import os
import sys
PROJECT_ROOT = os.path.abspath('./')
sys.path.append(PROJECT_ROOT)
from utils.logger import logger
import cv2 
import pandas as pd
import numpy as np

def get_image_to_mnist_dataframe(image_file) -> pd.DataFrame:
    """
    function to get any image file and resize it to 28X28 pixels

    Arguments:
        image_file: file object containing image

    Returns:
        pandas dataframe with shape 1 X 784
    """
    try:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
        img = cv2.bitwise_not(img)

        img_array = (img.flatten())
        img_df  = pd.DataFrame(img_array.reshape(-1,1).T)
        img_df.columns = ['pixel'+str(i) for i in range(784)]
        logger.info("Successfully converted image object to pandas dataframe in MNIST format (1X784) shape")

    except Exception as e:
        logger.error(f"Fail to convert image object to dataframe: Error : {e}")
        raise ValueError(e)


    return img_df

