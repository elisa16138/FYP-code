# First we will import the necessary Library 

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

# Load some required libraries
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import keras
import matplotlib.pyplot as plt
import math
import time
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from sklearn import metrics
import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping



from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from sklearn.metrics import mean_squared_error, r2_score
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.offline as pyo
cf.go_offline()
pyo.init_notebook_mode()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
print(__version__)
import os
for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))











maindf = pd.read_csv('C:/Users/Eclipsa/Desktop/FYP/2021-2022股票价格数据.csv')

print('Total number of days present in the dataset: ',maindf.shape[0])
print('Total number of fields present in the dataset: ',maindf.shape[1])

maindf.shape
maindf.head()
maindf.tail()
maindf.info()
maindf.describe()

#Checking for Null Values
#print('Null Values:',maindf.isnull().values.sum())
#print('NA values:',maindf.isnull().values.any())
maindf.drop_duplicates(inplace=True)
maindf.isna().sum

# If dataset had null values we can use this code to drop all the null values present in the dataset

# maindf=maindf.dropna()
# print('Null Values:',maindf.isnull().values.sum())
# print('NA values:',maindf.isnull().values.any())
# Final shape of the dataset after dealing with null values


#Preprocessing
plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
plt.plot(maindf.Open.values, color='red', label='Open')
plt.plot(maindf.Close.values, color='green', label='Close')
plt.plot(maindf.Low.values, color='blue', label='Low')
plt.plot(maindf.High.values, color='black', label='High')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
#plt.show()

plt.subplot(1,2,2);
plt.plot(maindf.Volume.values, color='black', label='Volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best');


# Scalling
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x = maindf[['Open', 'Low', 'High', 'Volume']].copy()
y = maindf['Close'].copy()

x[['Open', 'Low', 'High', 'Volume']] = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y.values.reshape(-1, 1))


# Splitting

def load_data(X, seq_len, train_size=0.8):
    amount_of_features = X.shape[1]
    X_mat = X.values
    sequence_length = seq_len + 1
    data = []

    for index in range(len(X_mat) - sequence_length):
        data.append(X_mat[index: index + sequence_length])

    data = np.array(data)
    train_split = int(round(train_size * data.shape[0]))
    train_data = data[:train_split, :]

    x_train = train_data[:, :-1]
    y_train = train_data[:, -1][:, -1]

    x_test = data[train_split:, :-1]
    y_test = data[train_split:, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return x_train, y_train, x_test, y_test


window = 22
x['Close'] = y
X_train, y_train, X_test, y_test = load_data(x, window)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)








#Visualization
plt.figure(figsize=(15, 5));
plt.plot(maindf.Open.values, color='red', label='open')
plt.plot(maindf.Close.values, color='green', label='low')
plt.plot(maindf.Low.values, color='blue', label='low')
plt.plot(maindf.High.values, color='black', label='high')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('price/volume')
plt.legend(loc='best')
plt.show()



#LSTM architecture
model = Sequential()
# First LSTM layer with Dropout regularisation
model.add(LSTM(units=50, input_shape=(window,5),return_sequences=True))
model.add(Dropout(0.2))
# Second LSTM layer
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
# Third LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Fourth LSTM layer
model.add(LSTM(units=50))
model.add(Dropout(0.5))
# The output layer
model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
callbacks_list = [earlystop]

# Compiling the RNN
model.compile(optimizer='adam',loss='mean_squared_error')
# Fitting to the training set
start = time.time()
LSTM=model.fit(X_train,y_train,epochs=100, batch_size=35, validation_split=0.05, verbose=1,callbacks=callbacks_list)
print ('compilation time : ', time.time() - start)


model.summary()



import matplotlib.pyplot as plt

loss = LSTM.history['loss']
val_loss = LSTM.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()


plt.show()















trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

trainPredict = y_scaler.inverse_transform(trainPredict)
trainY = y_scaler.inverse_transform([y_train])
testPredict = y_scaler.inverse_transform(testPredict)
testY = y_scaler.inverse_transform([y_test])

plot_predicted = testPredict.copy()
plot_predicted = plot_predicted.reshape(321, 1)
plot_actual = testY.copy()
plot_actual = plot_actual.reshape(321, 1)
print(plot_actual.shape)
print(plot_predicted.shape)

plt.figure(figsize=(20,7))

plt.plot(pd.DataFrame(plot_predicted), label='Predicted')
plt.plot(pd.DataFrame(plot_actual), label='Actual')
plt.legend(loc='best')
plt.show()

trainScore = metrics.mean_squared_error(trainY[0], trainPredict[:,0]) ** .5
print('Train Score: %.2f RMSE' % (trainScore))
testScore = metrics.mean_squared_error(testY[0], testPredict[:,0]) ** .5
print('Test Score: %.2f RMSE' % (testScore))

prices = maindf.Close.values.astype('float32')
prices = prices.reshape(len(prices), 1)


trainPredictPlot = np.empty_like(prices)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[22:len(trainPredict)+22, :] = trainPredict

testPredictPlot = np.empty_like(prices)
testPredictPlot[:, :] = np.nan
testPredictPlot[(len(prices) - testPredict.shape[0]):len(prices), :] = testPredict

print(trainPredict)



plt.figure(figsize=(20,7))
plt.plot(pd.DataFrame(prices, columns=["close"]).close, label='Actual')
plt.plot(pd.DataFrame(trainPredictPlot, columns=["close"]).close, label='Training')
plt.plot(pd.DataFrame(testPredictPlot, columns=["close"]).close, label='Testing')
plt.legend(loc='best')
plt.show()

model.save('./Final_model.h5')


from keras_sequential_ascii import keras2ascii
keras2ascii(model)








