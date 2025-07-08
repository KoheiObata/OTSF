#%%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import os
import random
import time
import math
# %%
DATAPATH='/opt/home/kohei/OnlineTimeSeriesForecasting/OnlineTSF/dataset'
dataset='ETTm1'

df = pd.read_csv(f'{DATAPATH}/{dataset}.csv')
