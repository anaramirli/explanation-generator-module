"""Explanation Generator module."""

__version__ = '0.0.2'

from typing import Dict, List, Tuple
from fastapi import FastAPI
from pydantic import BaseModel

import shap
import sys
import numpy as np
import pandas as pd

from .core.aeas  import AEAS
from .core.algorithms import isolation_forest, autoencoder
from .core.tools import missingdata_handler, normalizer

app = FastAPI(
    title='Explanation Generator module.',
    docs_url='/documentation',
    redoc_url='/redoc',
    description='Explanation Generator based on ....',
    version=__version__
)


class Parameters(BaseModel):
    '''Parameters for explanation generation'''
    allsteps_included: bool = False
    timestep_start: int
    timestep_end: int


class FeatureExplanationInput(BaseModel):
    '''Data provided for explanation generation'''
    run_data: Dict[str, List[float]]
    first_n: int = 5
    parameters: Parameters
    
    
class FeatureExplanationOutput(BaseModel):
    '''Shapley values of features'''
    shapley_features: Dict[str, float]

        
class AEASInput(BaseModel):
    '''Data provided for AEAS (Autoencoder Anomaly with SHAP) explanation generator'''
    run_data: Dict[str, List[float]]
    num_anomalies_to_explain: int = 5
    parameters: Parameters        
        
        
class AEASOutput(BaseModel):
    '''
    Features that contribute to the anomaly for a given time step with highest reconstruction error
    '''
    timeseries_anomaly_features: Dict[int, List[Tuple]]
    

@app.post('/isolationtree-explanation-generator', response_model = FeatureExplanationOutput)
async def extract_important_features(time_series_data: FeatureExplanationInput):

    '''Apply Isolation Tree to extract featuer imporortance of most strongest features'''

    inputX = pd.DataFrame.from_dict(time_series_data.run_data)

    # train model with whole data
    model = isolation_forest(inputX)

    # calculate shapley values for selected timesteps
    if time_series_data.parameters.allsteps_included == True:
        shap_values = shap.TreeExplainer(model).shap_values(inputX)
    else:
        try:
            shap_values = shap.TreeExplainer(model).shap_values(inputX[time_series_data.parameters.timestep_start: time_series_data.parameters.timestep_end])
        except IndexError:
            "Timestep is out of range."

    # make sure that selected fist_n has the right size
    assert time_series_data.first_n <= inputX.shape[1], "Index is out of range"

    # get mean value of Shapley values for each feature
    feature_importance = np.abs(shap_values).mean(0)

    # get most importat features by sorting mean Shapley  values
    feature_importance = pd.DataFrame(list(zip(inputX.columns, feature_importance)), columns=['col_name','mean_shapley_vals'])
    feature_importance.sort_values(by=['mean_shapley_vals'], ascending=False,inplace=True)

    # convert panda series containing strong features to dictionary
    strongest_feature = feature_importance[:time_series_data.first_n].set_index(feature_importance.columns[0])[feature_importance.columns[1]].to_dict()

    # add the aggregated values of the remaning features to the first n strongest features as "Rest"
    if time_series_data.first_n != feature_importance.shape[0]:
        strongest_feature['Rest'] = np.sum(feature_importance.iloc[time_series_data.first_n:, 1])

    return FeatureExplanationOutput(shapley_features=strongest_feature)



@app.post('/aeas-generator', response_model = AEASOutput)
async def extract_anomaly_features(time_series_data: AEASInput):

    '''
        Apply AEAH to extract Features that contribute to the anomaly for
        a given time step with highest reconstruction error
    '''

    inputX = pd.DataFrame.from_dict(time_series_data.run_data)
    
    # deal with missing data and normalize data
    inputX = missingdata_handler(inputX)
    inputX = normalizer(inputX)

    # train model with whole data
    model = autoencoder(inputX)
    
    # prepare anomaly explanation model
    explain_anomolies = AEAS(time_series_data.num_anomalies_to_explain)

    # calculate shapley values for selected timesteps
    if time_series_data.parameters.allsteps_included == True:
        explaining_features = explain_anomolies.explain_unsupervised_data(x_train=inputX,
                                                                          x_explain=inputX,
                                                                          autoencoder=model,
                                                                          return_shap_values=True)
    else:
        try:
            # prepare the explainer for given range
            x_explain = inputX[time_series_data.parameters.timestep_start: time_series_data.parameters.timestep_end]
            explaining_features = explain_anomolies.explain_unsupervised_data(x_train=inputX,
                                                                              x_explain=x_explain,
                                                                              autoencoder=model,
                                                                              return_shap_values=True)
        except IndexError:
            "Timestep is out of range."

    return AEASOutput(timeseries_anomaly_features=explaining_features)