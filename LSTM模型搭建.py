import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt

# For Evalution we will use these library

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

# For PLotting we will use these library

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots




#划分数据集





#Visualization
plt.figure(figsize=(15, 5));
plt.plot(maindf.Open.values, color='red', label='open')
plt.plot(maindf.Close.values, color='green', label='low')
plt.plot(maindf.Low.values, color='blue', label='low')
plt.plot(maindf.High.values, color='black', label='high')
plt.plot(maindf.ISI.values, color='orange', label='ISI')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('price/volume')
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(15, 5));
plt.plot(maindf.Close.values, color='green', label='Close')
plt.plot(maindf.ISI.values, color='orange', label='ISI')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('price/volume')
plt.legend(loc='best')
plt.show()

