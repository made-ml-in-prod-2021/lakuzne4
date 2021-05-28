import sys
# sys.path.append(r"D:\MADE_homeworks\ML_prod\homework1\ml_project")
import os

from ml_project.source_code.entities.parameters import (FeatureParams,
                                                        TrainParams,
                                                        TrainingPipelineParams,
                                                        EvaluationParams)
from ml_project.train_pipeline import train_pipeline


def test_train_e2e(
    tmpdir
    ):
    my_params = TrainingPipelineParams(
        algorithm_type='LogisticRegression',
        input_data_path=r"D:\MADE_homeworks\ML_prod\homework1\ml_project\ml_project\data\raw\heart.csv",
        splitting_params={'test_size': 0.2},
        feature_params=FeatureParams(target_col='target',
                                     numerical_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                                     categorical_features=['cp', 'restecg', 'slope', 'ca', 'thal'],
                                     selected_features=['cp', 'trestbps', 'restecg']
                                     ),
        train_params=TrainParams(model_factory='LogisticRegression',
                                 model_hyperparams={"C": 1.0}),
        output_model_path=r'D:\MADE_homeworks\ML_prod\homework1\ml_project\tests\test_model_output.pkl',
        evaluation_params=EvaluationParams(scorer_collection=['accuracy_score'],
                metrics_output_path=r"D:\MADE_homeworks\ML_prod\homework1\ml_project\tests\test_metrics.pkl")
                                      )
    real_model_path, metrics = train_pipeline(my_params)
    assert metrics['accuracy_score'] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(my_params.evaluation_params.metrics_output_path)


def test_log_messages(
        tmpdir,
        caplog
):
    my_params = TrainingPipelineParams(
        algorithm_type='LogisticRegression',
        input_data_path=r"D:\MADE_homeworks\ML_prod\homework1\ml_project\ml_project\data\raw\heart.csv",
        splitting_params={'test_size': 0.2},
        feature_params=FeatureParams(target_col='target',
                                     numerical_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                                     categorical_features=['cp', 'restecg', 'slope', 'ca', 'thal'],
                                     selected_features=['cp', 'trestbps', 'restecg']
                                     ),
        train_params=TrainParams(model_factory='LogisticRegression',
                                 model_hyperparams={"C": 1.0}),
        output_model_path=r'D:\MADE_homeworks\ML_prod\homework1\ml_project\tests\test_model_output.pkl',
        evaluation_params=EvaluationParams(scorer_collection=['accuracy_score'],
                metrics_output_path=r"D:\MADE_homeworks\ML_prod\homework1\ml_project\tests\test_metrics.pkl")
                                      )
    real_model_path, metrics = train_pipeline(my_params)

    print(caplog)
