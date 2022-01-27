"""Explanation Generator module."""

__version__ = '0.1.0'

from typing import Dict, List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel

import shap
import sys
import joblib
import keras 
import numpy as np
import pandas as pd


app = FastAPI(
    title='Explanation Generator module.',
    docs_url='/documentation',
    redoc_url='/redoc',
    description='Explanation Generator based on ....',
    version=__version__
)


class Parameters(BaseModel):
    '''Parameters for explanation generation'''
    model_path: str
    scaler_path: str


class FeatureExplanationInput(BaseModel):
    '''Data provided for explanation generation'''
    
    parameters: Parameters
    explain_data: Dict[str, List[float]]
    baseline_data: Dict[str, List[float]]
    allsteps_included: bool = False
    timestep_start: int
    timestep_end: int
    first_n: int = 5
    
    
class FeatureExplanationOutput(BaseModel):
    '''Shapley values of features'''
    attributing_features: Dict[str, float]
    offseting_features: Dict[str, float]
        

@app.post('/shap-kernel-explainer-keraslstm', response_model = FeatureExplanationOutput)
async def extract_important_features(time_series_data: FeatureExplanationInput):

    # get model
    model = keras.models.load_model(time_series_data.parameters.model_path)
    # get scaler
    scaler = joblib.load(time_series_data.parameters.scaler_path)
    
    
    # load and preprocess inputX and baselineX
    inputX = pd.DataFrame.from_dict(time_series_data.explain_data)
    col_names = inputX.columns # get the columm names from the data
    inputX = scaler.transform(inputX.values)
    
    baselineX = pd.DataFrame.from_dict(time_series_data.baseline_data)
    baselineX = scaler.transform(baselineX.values)
    
    
    # wrapper function
    f = lambda x: np.mean(np.abs(x.reshape(x.shape[0], 1, x.shape[1]) \
        - model.predict(x.reshape(x.shape[0], 1, x.shape[1])))**2, axis=2)
    

    # calculate shapley values for selected timesteps
    if time_series_data.allsteps_included == True:
        explainer = shap.KernelExplainer(f, baselineX)
        shap_values = explainer.shap_values(inputX)
    else:
        try:
            start = time_series_data.timestep_start
            end = time_series_data.timestep_end
            
            explainer = shap.KernelExplainer(f, baselineX)
            shap_values = explainer.shap_values(inputX[start:end])
        except IndexError:
            "Timestep is out of range."

    # make sure that selected fist_n has the right size
    assert time_series_data.first_n <= inputX.shape[1], "Index is out of range"

    # get mean value of Shapley values for each feature
    feature_importance = np.mean(shap_values[0],axis=0)
    
    # get mean value of Shapley values for each feature, and sort them
    feature_importance = np.mean(shap_values[0],axis=0)
    feature_importance = dict(list(zip(col_names, feature_importance)))
    feature_importance = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)

    # get top n attributing and offsetting features
    top_n = time_series_data.first_n
    attributing = [{key:round(value, 4)} for key, value in feature_importance if value>0][:top_n]
    offseting = [{key:round(value, 4)} for key, value in feature_importance if value<0][-top_n:]
    
    # cover attributing and offsetting dictlists to dict
    attributing_top = {}
    for element in attributing:
        dict_elements = list(element.items())[0]
        attributing_top[dict_elements[0]] = dict_elements[1]
        
    offseting_top = {}
    for element in offseting[::-1]:
        dict_elements = list(element.items())[0]
        offseting_top[dict_elements[0]] = dict_elements[1]
        
    return FeatureExplanationOutput(attributing_features=attributing_top,
                               offseting_features=offseting_top)