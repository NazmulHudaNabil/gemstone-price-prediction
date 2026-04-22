import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from pydantic import BaseModel, Field
from typing import Annotated

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)


class CustomData(BaseModel):
    carat: Annotated[float, Field(description="Carat weight of the gemstone", example=0.5)]
    cut: Annotated[str, Field(description="Cut quality of the gemstone", example="Ideal")]
    color: Annotated[str, Field(description="Color grade of the gemstone", example="E")]
    clarity: Annotated[str, Field(description="Clarity grade of the gemstone", example="VS1")]
    depth: Annotated[float, Field(description="Depth percentage of the gemstone", example=61.5)]
    table: Annotated[float, Field(description="Table percentage of the gemstone", example=55.0)]
    x: Annotated[float, Field(description="Length of the gemstone in mm", example=5.0)]
    y: Annotated[float, Field(description="Width of the gemstone in mm", example=5.0)]
    z: Annotated[float, Field(description="Height of the gemstone in mm", example=3.0)]

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = self.dict()
            return pd.DataFrame(custom_data_input_dict, index=[0])
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)