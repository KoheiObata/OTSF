# OTSF

This repository provides code and resources for evaluating Online Time Series Forecasting (OTSF) models, focusing on realistic adaptation to non-stationarity (concept drift).

## Outstanding Features
### Corrected & Challenging Datasets
Commonly used datasets frequently lack significant concept drift, making it difficult to assess true online adaptation capabilities.

__Our Solution:__
We provide scripts and methodologies for:
Correcting existing real-world datasets (ETTh2, ECL, Traffic) by selecting non-stationary features or removing anomalies.
Generating artificial datasets (SynA, SynB) with explicit, controllable concept drifts using SARIMA processes.
These datasets ensure that online adaptation is genuinely necessary for good performance.


### The One-Path Learning Protocol
The prevalent offline pre-training paradigm doesn't reflect real-world scenarios where models often need to learn from scratch.

__Our Solution:__
Models are initialized once and learn sequentially through the entire data stream (Train/Warm-up -> Valid/Tuning -> Test/Evaluation) in a single pass, adhering to the "Update-then-Predict" cycle with proper handling of label lag.
Hyperparameters are tuned strictly based on validation performance.

### Supported Methods
__Online Learning__
- **Naive**: Online Gradient Descent.
- **ER**: Experience Replay.
- **EWC**: Elastic Weight Consolidataion.

__Others__
- **Linear**: Linear model updated with recurrent least square.
- **Chronos**: Foundataion model for time series forecasting.


## Usage
This project uses Python 3.10.14.
Please install the required libraries from `requirements.txt`.
Please refer to the directory `scripts`. \
`run_all.sh` will reproduce all the experiments on the paper.

