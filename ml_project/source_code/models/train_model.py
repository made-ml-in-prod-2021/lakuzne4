import pickle
import numpy as np
import pandas as pd
import json
from dataclasses import asdict

from ..entities.input_objects import possible_models_dict, possible_score_funcs
from ..entities.parameters import TrainParams


def train_model(data: pd.DataFrame, target: pd.Series, train_params: TrainParams):
    model = possible_models_dict[train_params.model_factory](**train_params.model_hyperparams)
    model.fit(data, target)
    return model


def predict_model(model, data) -> np.ndarray:
    predicts = model.predict(data)
    return predicts


def evaluate_model(target, predicts, params):
    metrics = {}
    scorers = [possible_score_funcs[scorer_name] for scorer_name in params.scorer_collection]
    for score_func in scorers:
        metrics[score_func.__name__] = score_func(target, predicts)
    return metrics


def serialize_model(model, transformer, where_to_save):
    with open(where_to_save, "wb") as f:
        pickle.dump([model, transformer], f)
    return where_to_save


def save_metrics(metrics: dict, params) -> None:
    to_save = {'result_metrics': metrics,
               'model_parameters': asdict(params)}

    with open(params.evaluation_params.metrics_output_path, "w") as f:
        json.dump(to_save, f, indent=4)
