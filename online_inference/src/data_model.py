from typing import List, Union
from pydantic import BaseModel, conlist


class DataModelHeartDisease(BaseModel):
    id: int = 0
    age: int = 35
    sex: int = 0
    cp: int = 0
    trestbps: int = 100
    chol: int = 200
    fbs: int = 0
    restecg: int = 0
    thalach: int = 200
    exang: int = 0
    oldpeak: float = 0
    slope: int = 0
    ca: int = 0
    thal: int = 0



class ModelResponse(BaseModel):
    id: str
    predicted: int