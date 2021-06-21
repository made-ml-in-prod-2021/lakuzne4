from datetime import timedelta
from airflow.models import Variable
from airflow.utils.dates import days_ago

default_args = {
    "owner": "admin",
    "email": ["admin@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

PROD_MODEL_DIR = Variable.get("prod_model_dir") #"2021-06-08"
VOLUME = Variable.get("volume_path")

FULL_PROD_MODEL_PATH = f"/data/models/{PROD_MODEL_DIR}"
DATE_START = days_ago(5)
MODEL_DIR = "/data/models/{{ ds }}"
DATA_DIR = "/data/raw/{{ ds }}"
PREDICTION_DIR = "/data/predictions/{{ds }}"
