from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from joblib import load
import pandas as pd
from typing import List, Dict
from io import BytesIO

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float

class ItemsCollection(BaseModel):
    items: List[Item]

class PredictionResponse(BaseModel):
    prediction: float

class FilePredictionResponse(BaseModel):
    filename: str

def pydantic_model_to_df(model_instance):
    return pd.DataFrame([jsonable_encoder(model_instance)])

@app.post("/predict_item/", response_model=PredictionResponse)
def predict_item(item: Item) -> PredictionResponse:
    model = load('model.joblib')
    df_instance = pydantic_model_to_df(item)
    prediction = model.predict(df_instance).tolist()[0]
    return PredictionResponse(prediction=prediction)

@app.post("/predict_items/", response_model=FilePredictionResponse)
def predict_items(file: UploadFile = File(...)) -> FilePredictionResponse:
    model = load('model.joblib')
    contents = file.file.read()
    df = pd.read_csv(BytesIO(contents))
    predictions = model.predict(df).tolist()
    df['prediction'] = predictions

    output_filename = "predictions_output.csv"
    df.to_csv(output_filename, index=False)

    return FilePredictionResponse(filename=output_filename)