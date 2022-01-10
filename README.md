# Explanation Generator Module

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

### 1. isolationtree-explanation-generator

Request body
```
curl -X 'POST' \
  'http://localhost:8080/isolationtree-explanation-generator' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{ "run_data": {
	"A1": [616.7353, 611.1519, 626.6752, 637.2719, 642.9557, 610.7206],
	"A2": [665.1294, 665.1294, 665.1294, 665.1294, 664.2964, 664.2964],
	"A3": [636.7812, 636.7812, 636.7812, 636.7812, 636.7812, 636.7812],
	"A4": [321.3449, 320.9327, 321.784, 322.1602, 318.5836, 318.0116],
	"A5": [24.9016, 24.6329, 24.7058, 25.1163, 24.6315, 24.431],
	"A6": [289.6083, 289.0702, 289.8628, 290.1725, 286.9941, 286.294],
	"A7": [0.8744, 0.8744, 0.8744, 0.8744, 0.7968, 0.8902],
	"A8": [408.0314, 408.0314, 408.0314, 408.0314, 408.0314, 408.0314],
	"A9": [19.0921, 19.0921, 19.0921, 19.0921, 19.0921, 19.0921],
	"B1": [635.8683, 637.0881, 636.0352, 634.4209, 632.9593, 637.18],
	"B2": [634.0299, 635.8535, 636.883, 637.4162, 637.4162, 634.9201],
	"B3": [691.7459, 698.7901, 703.62399, 706.7784, 708.6975, 694.8862],
	"B4": [511.7529, 476.8244, 454.6953, 440.3727, 430.2531, 492.8985],
	"C1": [779.6705, 779.6705, 779.6705, 779.6705, 779.6705, 779.6705],
	"C2": [777.9381, 777.9381, 777.9381, 777.9381, 777.9381, 777.9381],
	"C3": [758.9238, 758.4783, 758.4783, 758.4783, 758.4783, 758.9803],
	"C4": [136.2615, 136.7904, 136.0211, 135.6675, 135.2121, 135.7022],
	"C5": [15.87, 15.9627, 16.0212, 15.9628, 15.9291, 15.7457],
	"C6": [274.0298, 274.8324, 273.3234, 272.7507, 271.6059, 272.5491], 
	"C7": [-0.3762, -0.3762, -0.3053, -0.3053, -0.3053, -0.3053],
	"C8": [109.6026, 109.6026, 109.6026, 110.3097, 109.6026, 109.6026],
	"D1": [696.0388, 693.87, 693.87, 693.87, 693.87, 695.0573],
	"D2": [671.8816, 671.4991, 671.4991, 671.4991, 671.4991, 671.4991],
	"D3": [634.2017, 632.0032, 632.0032, 631.0006, 631.0006, 633.0731],
	"D4": [468.764, 487.7069, 487.5909, 491.4545, 492.1715, 490.2036],
	"E1": [605.5621, 605.1485, 605.1485, 604.6945, 604.6945, 605.5035],
	"E2": [615.0891, 615.0891, 615.0891, 615.0891, 615.0891, 615.0891],
	"E3": [590.2991, 590.2991, 590.2991, 590.2991, 590.2991, 590.2991],
	"E4": [819.8063, 809.7016, 817.7061, 816.6172, 820.4639, 817.4162],
	"X1": [0.8217, 0.7416, 0.5517, 0.3218, -0.0288, 0.7465],
	"X2": [0.3313, 0.3313, 0.3313, 0.3313, 0.3313, 0.3313]}, 
	
	"first_n": 5, 
	
    "parameters": {
        "allsteps_included": false,
        "timestep_start": 1,
        "timestep_end": 4}
}'
```

Response
```
{
   "shapley_features":
   {
	"A5":0.06360128205686923,
	"C4":0.05783355715129527,
	"C8":0.05015626154668473,
	"C5":0.04617671917708279,
	"C3":0.04209151016301943,
	"Rest":0.4093395347533801
   }
}
```

### 2. aeas-generator

Request body
```
curl -X 'POST' \
  'http://localhost:8080/aeas-generator' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{ "run_data": {
	"A1": [616.7353, 611.1519, 626.6752, 637.2719, 642.9557, 610.7206],
	"A2": [665.1294, 665.1294, 665.1294, 665.1294, 664.2964, 664.2964],
	"A3": [636.7812, 636.7812, 636.7812, 636.7812, 636.7812, 636.7812],
	"A4": [321.3449, 320.9327, 321.784, 322.1602, 318.5836, 318.0116],
	"A5": [24.9016, 24.6329, 24.7058, 25.1163, 24.6315, 24.431],
	"A6": [289.6083, 289.0702, 289.8628, 290.1725, 286.9941, 286.294],
	"A7": [0.8744, 0.8744, 0.8744, 0.8744, 0.7968, 0.8902],
	"A8": [408.0314, 408.0314, 408.0314, 408.0314, 408.0314, 408.0314],
	"A9": [19.0921, 19.0921, 19.0921, 19.0921, 19.0921, 19.0921],
	"B1": [635.8683, 637.0881, 636.0352, 634.4209, 632.9593, 637.18],
	"B2": [634.0299, 635.8535, 636.883, 637.4162, 637.4162, 634.9201],
	"B3": [691.7459, 698.7901, 703.62399, 706.7784, 708.6975, 694.8862],
	"B4": [511.7529, 476.8244, 454.6953, 440.3727, 430.2531, 492.8985],
	"C1": [779.6705, 779.6705, 779.6705, 779.6705, 779.6705, 779.6705],
	"C2": [777.9381, 777.9381, 777.9381, 777.9381, 777.9381, 777.9381],
	"C3": [758.9238, 758.4783, 758.4783, 758.4783, 758.4783, 758.9803],
	"C4": [136.2615, 136.7904, 136.0211, 135.6675, 135.2121, 135.7022],
	"C5": [15.87, 15.9627, 16.0212, 15.9628, 15.9291, 15.7457],
	"C6": [274.0298, 274.8324, 273.3234, 272.7507, 271.6059, 272.5491], 
	"C7": [-0.3762, -0.3762, -0.3053, -0.3053, -0.3053, -0.3053],
	"C8": [109.6026, 109.6026, 109.6026, 110.3097, 109.6026, 109.6026],
	"D1": [696.0388, 693.87, 693.87, 693.87, 693.87, 695.0573],
	"D2": [671.8816, 671.4991, 671.4991, 671.4991, 671.4991, 671.4991],
	"D3": [634.2017, 632.0032, 632.0032, 631.0006, 631.0006, 633.0731],
	"D4": [468.764, 487.7069, 487.5909, 491.4545, 492.1715, 490.2036],
	"E1": [605.5621, 605.1485, 605.1485, 604.6945, 604.6945, 605.5035],
	"E2": [615.0891, 615.0891, 615.0891, 615.0891, 615.0891, 615.0891],
	"E3": [590.2991, 590.2991, 590.2991, 590.2991, 590.2991, 590.2991],
	"E4": [819.8063, 809.7016, 817.7061, 816.6172, 820.4639, 817.4162],
	"X1": [0.8217, 0.7416, 0.5517, 0.3218, -0.0288, 0.7465],
	"X2": [0.3313, 0.3313, 0.3313, 0.3313, 0.3313, 0.3313]}, 
	
	"num_anomalies_to_explain": 1, 
	
    "parameters": {
        "allsteps_included": false,
        "timestep_start": 1,
        "timestep_end": 4}
}'
```

Response
```
{
  "timeseries_anomaly_features": {
    "1": [
      ["C7", 0.010373665747700564],
      ["E4", 0.006709346377488344],
      ["A1",0.006652534677852057],
      ["C6",0.002724539310819373],
      ["C4", 0.0026822643054557347],
      ["B2",0.001827991356478799],
      ["D1",0.0018144925011582117],
      ["C5",0.0017605594797516985],
      ["B1",0.0016457709914967063],
      ["A2",0.018202290039275516],
      ["C8",0.011963846012465171],
      ["X1",0.011328377632369546],
      ["C3",0.010913324700011905],
      ["D2", 0.009005796532144117],
      ["D3",0.008195828315337071]
    ]
  }
}
```

## Note

## New Release
1. Update `__version__` in `src/main.py` with a new commit.
2. Tag this commit.
