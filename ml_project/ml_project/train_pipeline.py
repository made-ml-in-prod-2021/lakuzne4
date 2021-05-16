import click
import logging
import sys
import pickle
import pandas as pd
sys.path.append(r"D:\MADE_homeworks\ML_prod\homework1_repo\ml_project")  # for developing

from ml_project.source_code.data.make_dataset import read_data, split_train_val_data
from ml_project.source_code.features.build_features import build_transformer, make_features, extract_target
from ml_project.source_code.models.train_model import (train_model,
                                                       predict_model,
                                                       evaluate_model,
                                                       serialize_model,
                                                       save_metrics)
from ml_project.source_code.entities.parameters import TrainingPipelineParams, read_config


logger = logging.getLogger('train_pipeline_logger')
logger.setLevel(logging.INFO)
logging_handler = logging.FileHandler(
            filename=r'D:\MADE_homeworks\ML_prod\homework1_repo\ml_project\ml_project\logs\logs.txt'
)
logging_handler.setFormatter(logging.Formatter("%(asctime)s;%(levelname)s;%(message)s"))
logger.addHandler(logging_handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    data = read_data(training_pipeline_params.input_data_path)
    logger.info("data is read")

    train_df, val_df = split_train_val_data(
        data,
        training_pipeline_params.splitting_params
    )
    logger.info("split is done")

    clear_train_df = train_df.drop(columns=[training_pipeline_params.feature_params.target_col],
                                   errors='ignore')
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(clear_train_df)
    logger.info("transformer is built")

    train_features = make_features(transformer, clear_train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    logger.info("train features are made")

    model = train_model(train_features, train_target, 
                    training_pipeline_params.train_params
                        )
    logger.info("model is trained")

    clear_val_df = val_df.drop(columns=[training_pipeline_params.feature_params.target_col],
                               errors='ignore')
    val_features = make_features(transformer, clear_val_df)
    val_target = extract_target(val_df,
                                training_pipeline_params.feature_params
                                )
    logger.info("vaidation features are made")
    
    predicts = predict_model(
        model,
        val_features
    )
    logger.info("predictions on validation set are made")

    metrics = evaluate_model(
        val_target,
        predicts,
        training_pipeline_params.evaluation_params
    )
    logger.info("predictions are validated")

    save_metrics(metrics, training_pipeline_params)
    logger.info("metrics saved")

    path_to_model = serialize_model(model, transformer,
                                    training_pipeline_params.output_model_path)
    logger.info("metrics serialised")

    return path_to_model, metrics


@click.group()
def cli_interface():
    pass


@cli_interface.command(name='train')
@click.option('--config_file')
def train_pipeline_from_file(config_file: str):
    training_pipeline_params = read_config(config_file)
    train_pipeline(training_pipeline_params)
    print("pipeline is trained")


@cli_interface.command(name='predict')
@click.option('--data_path')
@click.option('--model_path')
@click.option('--output_path')
def predict_by_model(model_path: str, output_path: str, data_path: str):
    with open(model_path, 'rb') as f:
        model, transformer = pickle.load(f)

    data = pd.read_csv(data_path)
    features = make_features(transformer, data)

    predictions = pd.Series(predict_model(model, features))
    predictions = predictions.to_frame("Predicted")
    predictions.to_csv(output_path, index=False)
    print(f"predictions saved to output_path")


if __name__ == '__main__':
    cli_interface()
