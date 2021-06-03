import pytest
from types import SimpleNamespace

from dataclasses import asdict
import warnings
import numpy as np
import pickle
import json

from source_code.models.train_model import (train_model,
                                                       predict_model,
                                                       evaluate_model,
                                                       serialize_model,
                                                       save_metrics
                                                       )
from source_code.features.build_features import build_transformer
from source_code.entities.parameters import (EvaluationParams,
                                                        FeatureParams,
                                                        TrainParams,
                                                        TrainingPipelineParams
                                                        )

from sklearn.base import BaseEstimator

@pytest.fixture
def get_transformer(get_test_file):
    mock_training_pipeline_params = SimpleNamespace(
        feature_params=FeatureParams(target_col='target',
                                     numerical_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                                     categorical_features=['cp', 'restecg', 'slope', 'ca', 'thal'],
                                     selected_features=['cp', 'trestbps', 'restecg']
                                     )
                                                           )
    transformer = build_transformer(mock_training_pipeline_params.feature_params)
    return transformer


@pytest.fixture
def get_fitted_model(get_test_file):
    file_to_read_dir, test_data = get_test_file

    test_params = TrainParams(model_factory='LogisticRegression',
                              model_hyperparams={"C": 2.0})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitted_model = train_model(test_data.drop(columns='target'), test_data['target'], test_params)
    return fitted_model


def test_train_model(get_test_file):
    file_to_read_dir, test_data = get_test_file

    test_params = TrainParams(model_factory='LogisticRegression',
                              model_hyperparams={"C": 2.0})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        test_model = train_model(test_data.drop(columns='target'), test_data['target'], test_params)

    assert isinstance(test_model, BaseEstimator)
    assert hasattr(test_model, 'fit')


def test_predict_model(get_test_file, get_fitted_model):
    file_to_read_dir, test_data = get_test_file
    fitted_model = get_fitted_model

    predicts = predict_model(fitted_model, test_data.drop(columns=['target']))
    assert len(predicts) == test_data.shape[0]


def test_evaluate_model():
    test_target = np.array([0, 1, 0])
    test_prediction = np.array([0, 0, 1])
    test_eval_params = EvaluationParams(scorer_collection=['accuracy_score'],
                                        metrics_output_path='some_path')

    evaluation_result = evaluate_model(test_target, test_prediction, test_eval_params)
    assert isinstance(evaluation_result, dict)
    assert evaluation_result['accuracy_score'] == pytest.approx(0.33, abs=0.01)


def test_serialize_model(tmp_path, get_fitted_model, get_transformer):
    new_dir = tmp_path / 'sub_dir'
    new_dir.mkdir()
    p = new_dir / "my_serialised_file.pkl"

    place_where_model_is_saved = serialize_model(get_fitted_model, get_transformer, p)

    with open(place_where_model_is_saved, "rb") as f:
        model, transformer = pickle.load(f)
        
    assert isinstance(model, BaseEstimator)


def test_save_metrics(tmp_path):
    metrics_dict = {'accuracy_score': 0.1}
    output_path = tmp_path / "sub_dir"
    output_path.mkdir()
    p = output_path / "my_saved_metrics.pkl"

    mock_training_params = TrainingPipelineParams(
        algorithm_type='LogisticRegression',
        input_data_path=r"input_data_path",
        splitting_params={'test_size': 0.2},
        feature_params=FeatureParams(target_col='target',
                                     numerical_features=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                                     categorical_features=['cp', 'restecg', 'slope', 'ca', 'thal'],
                                     selected_features=['cp', 'trestbps', 'restecg']
                                     ),
        train_params=TrainParams(model_factory='LogisticRegression',
                                 model_hyperparams={"C": 1.0}),
        output_model_path=r'output_path',
        evaluation_params=EvaluationParams(scorer_collection=['accuracy_score'],
                        metrics_output_path=str(p)
                                           )
    )

    save_metrics(metrics_dict, mock_training_params)

    with open(p, 'r') as f:
        metrics_to_check = json.load(f)

    assert metrics_to_check['result_metrics'] == metrics_dict
    assert metrics_to_check['model_parameters'] == asdict(mock_training_params)
