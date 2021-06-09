import os
import pandas as pd
import pickle
import click
from sklearn.preprocessing import StandardScaler

PATH_DATA = "data.csv"
PATH_MODEL = "model.pkl"
PATH_PREDICTION = "predictions.csv"

@click.command("predict")
@click.option("--input_dir")
@click.option("--model_dir")
@click.option("--output_dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, PATH_DATA))

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    model = pickle.load(open(os.path.join(model_dir, PATH_MODEL), "rb"))
    predictions = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(predictions).to_csv(os.path.join(output_dir, PATH_PREDICTION))


if __name__ == "__main__":
    predict()