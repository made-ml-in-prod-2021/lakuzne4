# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from dataclasses import asdict, dataclass


def read_data(input_filepath: str) -> pd.DataFrame:
    data = pd.read_csv(input_filepath)
    return data


def split_train_val_data(input_data: pd.DataFrame, params: dataclass) -> Tuple[pd.DataFrame]:
    train_data, val_data = train_test_split(input_data, **asdict(params))
    return train_data, val_data
