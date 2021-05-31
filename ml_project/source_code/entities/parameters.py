from dataclasses import dataclass, asdict
from typing import List
import yaml
from marshmallow_dataclass import class_schema


@dataclass
class FeatureParams:
    target_col: str
    numerical_features: list
    categorical_features: list
    selected_features: list


@dataclass
class TrainParams:
    model_factory: str
    model_hyperparams: dict


@dataclass
class EvaluationParams:
    scorer_collection: List[str]
    metrics_output_path: str


@dataclass
class TrainingPipelineParams:
    algorithm_type: str
    input_data_path: str
    splitting_params: dict
    feature_params: FeatureParams
    train_params: TrainParams
    output_model_path: str
    evaluation_params: EvaluationParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_config(config_path: str) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    with open(config_path, "r") as input_stream:
        config = schema.load(yaml.safe_load(input_stream))
        return config


def save_config(config: TrainingPipelineParams, config_path: str) -> None:
    with open(config_path, 'w') as outfile:
        yaml.dump(asdict(config), outfile, default_flow_style=False)
