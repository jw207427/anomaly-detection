import pandas as pd
from sklearn.pipeline import Pipeline

from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin

class RandomAddTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column=''):
        self.column = column
        
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # Perform arbitary transformation
        X[self.column] = randint(0, 10, X.shape[0])
        return X
    
class RandomSubtractTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column=''):
        self.column = column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Perform arbitary transformation
        X = X.drop([self.column], axis=1)
        return X
    
    
def get_pipeline(column):
    pipe = Pipeline(
        steps=[
            ("use_custom_transformer", RandomAddTransformer(column=column)),
            ("historic_transform", RandomSubtractTransformer(column=column))
        ]
    )
    return pipe