from src.feature_selection.pca_transformer import PCATransformer as PCA
import pandas as pd
import os, sys
PROJECT_ROOT = (os.path.abspath('./'))
sys.path.append(PROJECT_ROOT)
from ensure import ensure_annotations
from utils.logger import logger

def get_selected_components_df(input_df: pd.DataFrame, no_components: int = 20 ):
    """
    applies PCA on given dataframe and returns top n components as dataframe

    Arguments:
        input_df: Pandas dataframe on which to apply PCA decompositions
        no_components: no of components to be selected 

    Returns:
        dataframe consisting of top n components as columns of the dataframe
    
    """

    pca = PCA(n_components=no_components)
    try:
        pca.fit(input_df)
        principal_components = pca.transform(input_df)
        components_df = pd.DataFrame(principal_components)
    except Exception as e:
        logger.error(f"Error in applying PCA for reducing dimensionality : {e}")
        raise ValueError(e)

    components_df.columns = ["component_"+str(i) for i in range(no_components)]

    return components_df, pca