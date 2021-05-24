import json


import pytest

from ..rest_service import app, load_model
from ..src.data_model import DataModelHeartDisease
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize():
    load_model()


@pytest.fixture()
def test_data():
    data = [DataModelHeartDisease(id=0), DataModelHeartDisease(id=1)]
    return data


def test_main():
    response = client.get("/")
    assert response.status_code == 200


def test_predict(test_data):
    response = client.post("/predict",
                           data=json.dumps([row.__dict__ for row in test_data]))
    assert response.status_code == 200
    assert len(response.json()) == len(test_data)
    assert response.json()[0]['id'] == 0


def test_validation_types():
    row = DataModelHeartDisease()
    row.age = "20"
    response = client.post("/predict", data=json.dumps(row.__dict__))
    assert response.status_code == 400


def test_validation_binary():
    row = DataModelHeartDisease()
    row.sex = 10
    response = client.post("/predict", data=json.dumps(row.__dict__))
    assert response.status_code == 400


def test_validation_range():
    row = DataModelHeartDisease()
    row.chol = 12000
    response = client.post("/predict", data=json.dumps([row.__dict__]))
    assert response.status_code == 400
