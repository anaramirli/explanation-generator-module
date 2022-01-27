# Explanation Generator Module

#### Dir description
```
+-- src
|   +-- core
|   +-- test
|   +-- trained models/lstm keras
|   +-- main.py
+--- Dockerfile
+--- requirements.txt
```
    
* *core:* core/utility functions folder
* *test:* test unit folder
* *trained models/lstm keras:* folder for trained models that will be used in xai generators
* *Dockerfile:* docker container file

## Build
```sh
$ docker build . -t explanation-generator
```

## Run

With docker

```sh
$ docker run -p 8080:8080 explanation-generator
```


## Documentation
* Swagger: http://localhost:8080/documentation
* ReDoc: http://localhost:8080/redoc

## Use

### 1. shap-kernel-explainer-keraslstm

Request body
```
curl -X 'POST' \
  'http://localhost:8080/shap-kernel-explainer-keraslstm' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "parameters": {
    "model_path": "src/trained models/lstm keras/model",
    "scaler_path": "src/trained models/lstm keras/scaler/minmaxscaler.gz"
  },
    "explain_data": {
	"A1": [616.7353, 611.1519, 626.6752, 637.2719, 642.9557, 610.7206],
	"A2": [665.1294, 665.1294, 665.1294, 665.1294, 664.2964, 664.2964],
	"A3": [636.7812, 636.7812, 636.7812, 636.7812, 636.7812, 636.7812],
	"A4": [321.3449, 320.9327, 321.784, 322.1602, 318.5836, 318.0116],
	"A5": [24.9016, 24.6329, 24.7058, 25.1163, 24.6315, 24.431],
	"A6": [289.6083, 289.0702, 289.8628, 290.1725, 286.9941, 286.294],
	"A7": [0.8744, 0.8744, 0.8744, 0.8744, 0.7968, 0.8902],
	"A8": [408.0314, 408.0314, 408.0314, 408.0314, 408.0314, 408.0314],
	"A9": [19.0921, 19.0921, 19.0921, 19.0921, 19.0921, 19.0921]},

    "baseline_data": {
	"A1": [616.7353, 611.1519],
	"A2": [665.1294, 665.1294],
	"A3": [636.7812, 636.7812],
	"A4": [321.3449, 320.9327],
	"A5": [24.9016, 24.6329],
	"A6": [289.6083, 289.07024],
	"A7": [0.8744, 0.8744],
	"A8": [408.0314, 408.0314],
	"A9": [19.0921, 19.0921]},

  "allsteps_included": false,
  "timestep_start": 1,
  "timestep_end": 5,
  "first_n": 2
}'
```

Response
```
{
  "attributing_features": {
    "A1": 0.9302,
    "A5": 0.0011
  },
  "offseting_features": {
    "A2": -0.162,
    "A7": -0.0158
  }
}
```

## Request bodies (Schemas)
Here the structure of each request body used in main.py has been explained.

### 1. Parameters

```Python
class Parameters(BaseModel):
    '''Parameters for explanation generation'''
    model_path: str
    scaler_path: str    
```
This is the main request body for accessing the pre-trained networks

* **model_path**: path to the saved trained model (to the directory where a Keras model is saved using joblib)
* **scaler_path**: path to the scaler values (path of the joblib dumped file (.gz)), will be used to scale the new data.

### 2. FeatureExplanationInput

```Python
class FeatureExplanationInput(BaseModel):
    '''Data provided for explanation generation'''
    
    parameters: Parameters
    explain_data: Dict[str, List[float]]
    baseline_data: Dict[str, List[float]]
    allsteps_included: bool = False
    timestep_start: int
    timestep_end: int
    first_n: int = 5
```

This is the request body is used for an explanation of the general feature importance using kernel shap method on a keras lstm model. In the code examples, this request body has been used with /shap-kernel-explainer-keraslstm.

* **parameters**: are a class type of the main request of Parameters.
* **explain_dat**: is the structure of data accepted with an HTTP request. It can hold any time-series data, with the 2-D domain. It stands for the data we want to explain.
* **baseline_data**: is the structure of data accepted with an HTTP request. It can hold any time-series data, with the 2-D domain. This is a small subset, preferably from a training set, used as baseline value for explanation model.
* **allstep_included**: default value being False means only timesteps between timestep_start and timestep_end will be explained. If True, than all timesteps of the given data will be explained.
* **timestep_start**: timestep_end used to select the desired timestep range if allstep_includes has been set False.
* **first_n**: account for the number of n most strong features we want to explain.

### 3. FeatureExplanationInputOutput

```Python
class FeatureExplanationOutput(BaseModel):
    '''Shapley values of features'''
    attributing_features: Dict[str, float]
    offseting_features: Dict[str, float]
```
This is a response body which will be used for most of the xai APIs.

* **attributing_features**: list of top *first_n* features that contribute to the reconstruction loss, "str" stand for the feature name, "float" depicts the corresponding attribution values.
* **offseting_features**: list of top *first_n* features that reduce the reconstruction loss, "str" stands for the feature name, "float" depicts the corresponding attribution values.

## To Do
Integrated Gradient endpoint will be developed.

## New Release
1. Update `__version__` in `src/main.py` with a new commit.
2. Tag this commit.
