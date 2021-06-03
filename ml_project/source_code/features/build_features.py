from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

from ..entities.parameters import FeatureParams

__all__ = ['build_transformer']


class CustomTransformer(BaseEstimator, TransformerMixin):
    """add interacions for this features"""
    def __init__(self):
        pass

    def fit(self, X):
        self.X = X
        self.new_columns = sorted(combinations(self.X.columns, r=2))
        return self

    def transform(self, X):
        x_ = X.copy()
        for feature1, feature2 in self.new_columns:
            x_.loc[:, feature1 + '_' + feature2] = x_[feature1] * x_[feature2]
        return x_


def build_custom_transformer_pipeline() -> Pipeline:
    custom_pipeline = Pipeline(
        [
            ('CustomTransformer', CustomTransformer())
        ]

    )
    return custom_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(strategy='mean', missing_values=np.nan)),
            ('scaler', StandardScaler()),
        ]

    )
    return num_pipeline


def build_categorical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
            ('scaler', OneHotEncoder(handle_unknown="ignore")),
        ]

    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
         ('num_pipeline', build_numerical_pipeline(), params.numerical_features),
         ('categorical_pipeline', build_categorical_pipeline(), params.categorical_features),
         ('customer_transformer', build_custom_transformer_pipeline(), params.selected_features)
        ],
    )
    return transformer


def make_features(transformer: ColumnTransformer,
                  df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))
    
    
def extract_target(df: pd.DataFrame, params) -> pd.Series:
    target = df[params.target_col]
    return target
