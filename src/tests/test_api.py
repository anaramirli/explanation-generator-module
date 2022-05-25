"""Test the webserver built with FastAPI"""

from fastapi.testclient import TestClient
from ..main import app
import os
import json


client = TestClient(app)

# constants
model_name = 'keras_mvts_lstm.h5'
scaler_name = 'mvts_scaler.gz'


# upload files from mvts training (model)
def test_upload_model_files():


    # # download files (to the local machine) from Multivariate Aggregator container
    # wget.download(MVTS_URL+'/data/'+model_name)
    # wget.download(MVTS_URL+'/data/'+scaler_name)

    # paths to test data
    twd = os.path.dirname(__file__)

    model_path = os.path.join(
        twd,
        os.path.join('data', model_name))
        

    scaler_path = os.path.join(
        twd,
        os.path.join('data', scaler_name))
    
    # open/load the files
    model = open(model_path, 'rb')
    scaler = open(scaler_path, 'rb')


    model_response = client.post(
        "/upload-model/", files={"file": (model_name, model)}
    )

    scaler_response = client.post(
         "/upload-model/", files={"file": (scaler_name, scaler)} 
    )
    # delete files (from the local machine) after uploading
    model.close()
    scaler.close()

    assert model_response.status_code==200, "Upload-Model Fail Reason: {}\n".format(model_response.reason)
    assert scaler_response.status_code==200, "Upload-Scaler Fail Reason: {}\n".format(scaler_response.reason)


def test_isolationtree_explanation_generator():
    """Tests for of explaining feature importance."""
    response = client.post(
        '/keras-shap-kernel-explainer',
        json= {
          "paths": {
            "model": model_name,
            "scaler": scaler_name
          },
          "explain_data": {
            "data": {
                "A1": [616.7353, 611.1519, 626.6752, 637.2719, 642.9557, 610.7206],
                "A2": [665.1294, 665.1294, 665.1294, 665.1294, 664.2964, 664.2964],
                "A3": [636.7812, 636.7812, 636.7812, 636.7812, 636.7812, 636.7812],
                "A4": [321.3449, 320.9327, 321.784, 322.1602, 318.5836, 318.0116],
                "A5": [24.9016, 24.6329, 24.7058, 25.1163, 24.6315, 24.431]
                }
          },
          "baseline_data": {
            "data": {
                "A1": [616.7353, 611.1519],
                "A2": [665.1294, 665.1294],
                "A3": [636.7812, 636.7812],
                "A4": [321.3449, 320.9327],
                "A5": [24.9016, 24.6329]
                }
          },
          "allsteps_included": False,
          "timestep_start": 1,
          "timestep_end": 5,
          "first_n_attribution": 2
        }     
    )

    assert response.status_code == 200

    try:
        reponse = json.loads(response.text)
        reponse['attributing_features']
        reponse['offseting_features']
    except:
        assert False, "Key Error in the response data"


# returns list of files in data/. 
def test_list_files():

    response = client.get('/list-model-files')
    assert response.status_code==200, "Response Fail Reason: {}\n".format(response.reason)

    try:
        json.loads(response.text)['files']
    except:
        assert False, "Key Error in the response data"