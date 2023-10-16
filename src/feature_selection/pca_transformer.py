from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


#Defining a PCA tranformer class which extends sklearn PCA decomposer - to add a dummy predict method 
# This is needed as PCA transformer needs to be stored using mlflow, which requires that the class should have a predict method

class PCATransformer(BaseEstimator):
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
    
    def fit(self, X, y=None):
        self.pca.fit(X)
        return self
    
    def transform(self, X):
        return self.pca.transform(X)
    
    def predict(self, X):
        # Dummy predict method to satisfy MLflow
        return X
