import pytest
import pandas as pd

from ml_project.source_code.data.make_dataset import read_data, split_train_val_data
from tests.generate_test_data import get_test_file, TEST_DATA_SIZE, get_faker_test_file


def test_read_data(get_test_file):
    file_to_read_dir, test_data = get_test_file

    read_data_result = read_data(file_to_read_dir)
    assert read_data_result.shape == test_data.shape
    assert read_data_result.iloc[1, 1] == pytest.approx(test_data.iloc[1, 1], rel=0.0001)


def test_read_faked_data(get_faker_test_file):
    file_to_read_dir, test_data = get_faker_test_file

    read_data_result = read_data(file_to_read_dir)
    assert read_data_result.shape == test_data.shape
    assert read_data_result.iloc[1, 1] == pytest.approx(test_data.iloc[1, 1], rel=0.0001)


def test_split_train_val_data(get_test_file):
    file_to_read_dir, test_data = get_test_file
    test_data_params = {'test_size': TEST_DATA_SIZE // 2}

    test_train_data, test_val_data = split_train_val_data(test_data, test_data_params)

    assert test_train_data.shape[0] == test_val_data.shape[0] == TEST_DATA_SIZE // 2
