from numbers import Number
from typing import List, Union

from .data_model import DataModelHeartDisease


def check_input_range(
        value: Union[int, float],
        lower: Union[int, float],
        upper: Union[int, float],
        feature_name: str
):
    if not (lower <= value <= upper):
        raise ValueError(
            f"{feature_name} should be in range: [{lower} , {upper}] "
        )


def check_binary_feature(value: int,
                         feature_name: str):
    if value not in [0, 1]:
        raise ValueError(f"{feature_name} is binary feature. It should be 0 or 1")


def validate_row(sample: DataModelHeartDisease):
    try:
        check_input_range(sample.age, 0, 200, "age")
        check_binary_feature(sample.sex, "sex")
        check_input_range(sample.cp, 0, 3, "cp")
        check_input_range(sample.trestbps, 50, 500, "trestbps")
        check_input_range(sample.chol, 50, 1000, "chol")
        check_binary_feature(sample.fbs, "fbs")
        check_input_range(sample.restecg, 0, 2, "restecg")
        check_input_range(sample.thalach, 50, 500, "thalach")
        check_binary_feature(sample.exang, "exang")
        check_input_range(sample.oldpeak, 0, 10, "oldpeak")
        check_input_range(sample.slope, 0, 2, "slope")
        check_input_range(sample.ca, 0, 4, "ca")
        check_input_range(sample.thal, 0, 3, "thal")
        return True, "OK"
    except ValueError as err:
        return False, str(err)
