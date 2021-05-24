import json
import pandas as pd
import requests

SAMPLE_DATA_PATH = "./ml_project/ml_project/data/interim/sample_for_prediction.csv"

if __name__ == '__main__':
    to_predict = pd.read_csv(SAMPLE_DATA_PATH)
    print(to_predict.head().columns.tolist())
    to_predict.loc[:, 'id'] = pd.Series(list(range(to_predict.shape[0])))
    data = to_predict.to_dict(orient='records')
    response = requests.post("http://localhost:8000/predict", data=json.dumps(data))
    print(f"Response status is {response.status_code}")