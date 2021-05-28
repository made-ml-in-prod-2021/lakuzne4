import pandas as pd
import numpy as np
from scipy.stats import norm
import pytest
from faker import Faker

TEST_DATA_SIZE = 100

DATA_PARAMS = {'age': {'mean': 54.37, 'std': 9.08, 'min': 29.0, 'max': 77.0},
                 'sex': {'mean': 0.68, 'std': 0.47, 'min': 0.0, 'max': 1.0},
                 'cp': {'mean': 0.97, 'std': 1.03, 'min': 0.0, 'max': 3.0},
                 'trestbps': {'mean': 131.62, 'std': 17.54, 'min': 94.0, 'max': 200.0},
                 'chol': {'mean': 246.26, 'std': 51.83, 'min': 126.0, 'max': 564.0},
                 'fbs': {'mean': 0.15, 'std': 0.36, 'min': 0.0, 'max': 1.0},
                 'restecg': {'mean': 0.53, 'std': 0.53, 'min': 0.0, 'max': 2.0},
                 'thalach': {'mean': 149.65, 'std': 22.91, 'min': 71.0, 'max': 202.0},
                 'exang': {'mean': 0.33, 'std': 0.47, 'min': 0.0, 'max': 1.0},
                 'oldpeak': {'mean': 1.04, 'std': 1.16, 'min': 0.0, 'max': 6.2},
                 'slope': {'mean': 1.4, 'std': 0.62, 'min': 0.0, 'max': 2.0},
                 'ca': {'mean': 0.73, 'std': 1.02, 'min': 0.0, 'max': 4.0},
                 'thal': {'mean': 2.31, 'std': 0.61, 'min': 0.0, 'max': 3.0},
                 'target': {'mean': 0.54, 'std': 0.5, 'min': 0.0, 'max': 1.0}}

FAKER_PARAMETERS = [
    ('age', 29, 80),
    ('sex', 0 , 1),
    ('cp', 0, 3),
    ('trestbps', 90, 220),
    ('chol', 100, 600),
    ('fbs', 0, 1),
    ('restecg', 0, 2),
    ('thalach', 70, 210),
    ('exang', 0, 1),
    ('oldpeak', 0, 7),
    ('slope', 0, 2),
    ('ca', 0, 4),
    ('thal', 0, 3),
    ('target', 0, 1)
]


def create_data_gen_func(sample_size, random_state=None):
    """self-made generator of data generating function
       depending on distribution parameters
    """
    def data_gen(mean_, std_, min_, max_):
        return norm(loc=mean_, scale=std_).rvs(sample_size, random_state=random_state)

    return data_gen


def generate_data(sample_size: int,
                  random_state: int = None,
                  data_params: dict = DATA_PARAMS) -> pd.DataFrame:
    """self made data-generating function"""
    data_gen_func = create_data_gen_func(sample_size=sample_size, random_state=random_state)
    result = []
    for col_name, col_params_dict in data_params.items():
        result.append(pd.Series(data_gen_func(
            **{param_name + "_": param_value for param_name, param_value in col_params_dict.items()}
                                             ),
                                name=col_name
                                )
        )
    result_df = pd.concat(result, axis=1)
    result_df.loc[:, 'target'] = pd.Series(np.random.randint(0, 2, sample_size))

    return result_df


@pytest.fixture
def get_test_file(tmpdir: pytest.fixture):
    file_to_read_dir = tmpdir.join("test_data.csv")
    test_data = generate_data(TEST_DATA_SIZE)
    test_data.to_csv(file_to_read_dir, index=False)
    return file_to_read_dir, test_data


def create_faker_object() -> Faker:
    fake = Faker()
    Faker.seed(0)
    for parameter, min_value, max_value in FAKER_PARAMETERS:
        fake.set_arguments(parameter, {'min_value': min_value, 'max_value': max_value})
    return fake


def generate_fake_data(sample_size: int) -> pd.DataFrame:
    """faker based generator of data """
    fake = create_faker_object()
    header, *data = \
        [row.split(",") for row in fake.csv(header=[i[0] for i in FAKER_PARAMETERS],
                                            data_columns=["{{" + f"pyint: {parameter}" + "}}" for parameter in
                                                          [i[0] for i in FAKER_PARAMETERS]],
                                            num_rows=sample_size
                                            ).split("\r\n")]

    def func(x, additional_func=None):
        if additional_func is None:
            return x.replace('"', '')
        return additional_func(x.replace('"', ''))

    data_matrix = np.vstack([np.array(i) for i in data[:-1]])
    data = np.array([func(i, int) for i in data_matrix.ravel()]).reshape(data_matrix.shape)
    header = [func(i) for i in header]
    return pd.DataFrame(data, columns=header)


@pytest.fixture
def get_faker_test_file(tmpdir: pytest.fixture):
    file_to_read_dir = tmpdir.join("test_data.csv")
    test_data = generate_fake_data(TEST_DATA_SIZE)
    test_data.to_csv(file_to_read_dir, index=False)
    return file_to_read_dir, test_data
