from sklearn.decomposition import PCA
import pandas as pd

def get_selected_components_df(input_df: pd.DataFrame, no_components: int = 20 ) -> pd.DataFrame:
    """
    applies PCA on given dataframe and returns top n components as dataframe

    Arguments:
        input_df: Pandas dataframe on which to apply PCA decompositions
        no_components: no of components to be selected 

    Returns:
        dataframe consisting of top n components as columns of the dataframe
    
    """

    pca = PCA(n_components=no_components)
    principal_components = pca.fit_transform(input_df)

    components_df = pd.DataFrame(principal_components)
    components_df.columns = ["component_"+str(i) for i in range(no_components)]

    return components_df