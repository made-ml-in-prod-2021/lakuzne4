import pickle
from typing import List, Optional, Union
import os
import sys

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from fastapi import FastAPI, HTTPException
import uvicorn

# sys.path.append("D:\MADE_homeworks\ML_prod\homework2_repo\online_inference")

from .src.model import (make_features,
                        predict_model
                        )
from .src.data_model import DataModelHeartDisease, ModelResponse
from .src.data_validation import validate_row

DEFAULT_PATH_TO_MODEL = "model_output_baseline.pkl"

model: Optional[LogisticRegression]
transformer: Optional[Pipeline]

app = FastAPI()


def load_object(path: str) -> Pipeline:
    with open(path, "r") as f:
        return pickle.load(f)


def make_predict(
        data: List[DataModelHeartDisease], features: List[str],
        model: LogisticRegression,
        transformer: Pipeline
):
    df = pd.DataFrame(data, columns=features)
    ids = [int(ind) for ind, x in enumerate(data)]

    prediction_features = make_features(transformer, df)
    predictions = pd.Series(predict_model(model,
                                          prediction_features
                                          )
                            ).to_frame("Predicted")
    return [
        ModelResponse(id=id_,
                      pred=prediction)
        for id_, prediction in zip(ids, predictions)
    ]


@app.get("/")
def main():
    return "It's entry point for prediction"


@app.on_event("startup")
def load_model():
    global model, transformer

    model_path = os.getenv("PATH_TO_MODEL",
                           default=DEFAULT_PATH_TO_MODEL)
    if model_path is None:
        raise AssertionError

    with open(model_path, 'rb') as f:
        model, transformer = pickle.load(f)


@app.post("/predict/", response_model=List[ModelResponse])
def predict(request: DataModelHeartDisease):
    if model is None or transformer is None:
        raise HTTPException(
            status_code=400, detail=f"Model is not available"
        )
    if transformer is None:
        raise HTTPException(
            status_code=400, detail=f"Transformer is not available"
        )
    for row in request:
        is_valid, message = validate_row(row)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"row with id = {row.id} is not valid row"
            )
    return make_predict(request.data, request.features, transformer,
                        model)


if __name__ == "__main__":
    uvicorn.run("rest_service:app", host="0.0.0.0", port=os.getenv("PORT", 8000))