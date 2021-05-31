import os

from source_code.entities.parameters import (FeatureParams,
                                                        TrainParams,
                                                        TrainingPipelineParams,
                                                        EvaluationParams)
from train_pipeline import train_pipeline


TEST_MODEL_PARAMS = dict(
    algorithm_type='RandomForestClassifier',
    input_data_path=r"tests\test_data\heart.csv",
    splitting_params={'test_size': 0.5},
    feature_params=FeatureParams(target_col='target',
                                 numerical_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                                 categorical_features=['cp', 'restecg', 'slope', 'ca', 'thal'],
                                 selected_features=['cp', 'trestbps', 'restecg']
                                 ),
    train_params=TrainParams(model_factory='RandomForestClassifier',
                             model_hyperparams={}),
    output_model_path=r'tests\test_data\test_model_output.pkl',
    evaluation_params=EvaluationParams(scorer_collection=['accuracy_score'],
                    metrics_output_path=r"tests\test_data\test_metrics.pkl")
)


def test_train_e2e(
    tmpdir
    ):
    my_params = TrainingPipelineParams(**TEST_MODEL_PARAMS)
    real_model_path, metrics = train_pipeline(my_params)
    assert metrics['accuracy_score'] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(my_params.evaluation_params.metrics_output_path)


def test_log_messages(
        tmpdir,
        caplog
):
    my_params = TrainingPipelineParams(**TEST_MODEL_PARAMS)
    real_model_path, metrics = train_pipeline(my_params)

    print(caplog)
