import pytest
import yaml
from dataclasses import asdict

from ml_project.source_code.entities.parameters import read_config
from .generate_test_data import get_test_file


@pytest.fixture
def write_yaml_file(tmpdir, get_test_file):
    file_to_write = tmpdir.join("test_config.yaml")

    dict_to_write = {
        'algorithm_type': 'LogisticRegression',
        'input_data_path' : 'get_test_file',
        'splitting_params': {'test_size': 0.2},
        'feature_params': {'target_col': 'target',
                           'numerical_features': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                           'categorical_features': ['cp', 'restecg', 'slope', 'ca', 'thal'],
                           'selected_features': ['cp', 'trestbps', 'restecg']
                           },
        'train_params': {'model_factory': 'LogisticRegression',
                         'model_hyperparams': {"C": 1.0}
                         },
        'output_model_path': 'tmpdir.join("model_result.pkl")',
        'evaluation_params': {'scorer_collection': ['accuracy_score'],
                              'metrics_output_path': 'tmpdir.join("metrics.pkl")'
                              }
    }

    with open(file_to_write, 'w') as file:
        documents = yaml.dump(dict_to_write, file)

    return file_to_write, dict_to_write


def test_read_config(tmpdir, write_yaml_file):
    config_file, written_dict = write_yaml_file
    read_dataclass = read_config(config_path=config_file)
    assert asdict(read_dataclass) == written_dict




