"""Test the webserver built with FastAPI"""

import sys
from fastapi.testclient import TestClient

# sys.path.append("..")
from ..main import app

client = TestClient(app)


def test_isolationtree_explanation_generator():
    """Tests for of explaining feature importance."""
    response = client.post(
        '/shap-kernel-explainer-keraslstm',
        json= {
          "paths": {
            "model": "../trained models/lstm keras/model",
            "scaler": "../trained models/lstm keras/scaler/minmaxscaler.gz"
          },
          "explain_data": {
            "data": {
                "A1": [616.7353, 611.1519, 626.6752, 637.2719, 642.9557, 610.7206],
                "A2": [665.1294, 665.1294, 665.1294, 665.1294, 664.2964, 664.2964],
                "A3": [636.7812, 636.7812, 636.7812, 636.7812, 636.7812, 636.7812],
                "A4": [321.3449, 320.9327, 321.784, 322.1602, 318.5836, 318.0116],
                "A5": [24.9016, 24.6329, 24.7058, 25.1163, 24.6315, 24.431],
                "A6": [289.6083, 289.0702, 289.8628, 290.1725, 286.9941, 286.294],
                "A7": [0.8744, 0.8744, 0.8744, 0.8744, 0.7968, 0.8902],
                "A8": [408.0314, 408.0314, 408.0314, 408.0314, 408.0314, 408.0314],
                "A9": [19.0921, 19.0921, 19.0921, 19.0921, 19.0921, 19.0921]
                }
          },
          "baseline_data": {
            "data": {
                "A1": [616.7353, 611.1519],
                "A2": [665.1294, 665.1294],
                "A3": [636.7812, 636.7812],
                "A4": [321.3449, 320.9327],
                "A5": [24.9016, 24.6329],
                "A6": [289.6083, 289.07024],
                "A7": [0.8744, 0.8744],
                "A8": [408.0314, 408.0314],
                "A9": [19.0921, 19.0921]
                }
          },
          "allsteps_included": False,
          "timestep_start": 1,
          "timestep_end": 5,
          "first_n_attribution": 2
        }     
    )

    assert response.status_code == 200
    assert response.json() == {
              "attributing_features": {
                "A1": 0.9302,
                "A5": 0.0011
              },
              "offseting_features": {
                "A2": -0.1620,
                "A7": -0.0158
              }
            }