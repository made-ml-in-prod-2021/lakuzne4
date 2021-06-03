import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer


def make_features(transformer: ColumnTransformer,
                  df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def predict_model(model, data) -> np.ndarray:
    predicts = model.predict(data)
    return predicts
