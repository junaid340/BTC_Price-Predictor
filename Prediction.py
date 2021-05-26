# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:17:50 2021

@author: Zigron
"""
#import Libraries
import numpy as np
import pandas
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,BatchNormalization, Input
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

#Load Test Data
Xtest = np.load('Xtest.npy')
Ytest = np.load('Ytest.npy')

#Build Model
def lstm_layer (hidden1) :
    
    model = Sequential()
    
    # add input layer
    model.add(Input(shape = (500, 2, )))
    
    # add rnn layer
    model.add(LSTM(hidden1, activation = 'tanh', return_sequences = False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # add output layer
    model.add(Dense(1, activation = 'linear'))
    
    model.compile(loss = "mean_squared_error", optimizer = 'adam')
    
    return model

model = lstm_layer(256)
model.summary()

#Load Weights
model = load_model('./bit_model_lstm.h5')

#Predictions
pred = model.predict(Xtest)

pred = pred.reshape(-1)
print('MSE : ' + str(mean_squared_error(Ytest, pred)))

#Visualize Predictions
plt.figure(figsize = (20,7))
plt.plot(Ytest[2040:2060])
plt.plot(pred[2040:2060])
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Closing Price vs Time (using SimpleRNN)')
plt.legend(['Actual price', 'Predicted price'])
plt.show()