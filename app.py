import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.pipline.predict_pipline import CustomData, PredictPipeline
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
def predict(data: CustomData):
    try:
        data_df = data.get_data_as_dataframe()
        logging.info('Dataframe Gathered')
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data_df)
        logging.info('Prediction Completed')
        return {"prediction": float(prediction[0])}
    except Exception as e:
        logging.info('Exception occured during prediction')
        raise CustomException(e, sys)
