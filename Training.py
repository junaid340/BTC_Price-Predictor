# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:56:26 2021

@author: Zigron
"""

#import libraries
import numpy as np
import pandas
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,BatchNormalization, Input
from keras.callbacks import ModelCheckpoint

#import dataset
bit_df = pandas.read_csv('./btc_data.csv')
bit_df.head()

#visualize Features
info = [[col, bit_df[col].count(), bit_df[col].max(), bit_df[col].min()] for col in bit_df.columns]
print(tabulate(info, headers = ['Feature', 'Count', 'Max', 'Min'], tablefmt = 'orgtbl'))

#Drop NA values from the dataset
bit_df = bit_df.dropna()
bit_df = bit_df[bit_df['Timestamp'] > (bit_df['Timestamp'].max()-650000)]
bit_df = bit_df.reset_index(drop = True)
bit_df.head()

bit_df = bit_df.drop(['Timestamp', 'Low', 'High', 'Volume_(BTC)', 'Weighted_Price'], axis = 1)
info = [[col, bit_df[col].count(), bit_df[col].max(), bit_df[col].min()] for col in bit_df.columns]
print(tabulate(info, headers = ['Feature', 'Count', 'Max', 'Min'], tablefmt = 'orgtbl'))

#Visualize the recent Trends
plt.figure(figsize = (20,10))
plt.subplot(2,1,1)
plt.plot(bit_df['Open'].values[bit_df.shape[0]-500:bit_df.shape[0]])
plt.xlabel('Time period')
plt.ylabel('Opening price')
plt.title('Opening price of Bitcoin for last 500 timestamps')

plt.subplot(2,1,2)
plt.plot(bit_df['Volume_(Currency)'].values[bit_df.shape[0]-500:bit_df.shape[0]])
plt.xlabel('Time period')
plt.ylabel('Volume Traded')
plt.title('Volume traded of Bitcoin for last 500 timestamps')
plt.show()

#Create Arrays & Scalling Data
X = np.array(bit_df.drop(['Close'], axis = 1))
y = np.array(bit_df['Close'])

X = StandardScaler().fit_transform(X)

t = np.reshape(y, (-1,1))
y = StandardScaler().fit_transform(t)
y = y.reshape(-1)

#Creating Time Series Data for LSTMS
length = 500
X_temp = []
y_temp = []
for i in range(length,X.shape[0]) :
    X_temp.append(X[i-length: i])
    y_temp.append(y[i])
X_temp = np.array(X_temp)
y_temp = np.array(y_temp)

#Spliting Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size = 0.2, random_state = 1)

#Saving Test Date for Prediction
np.save('Xtest.npy', X_test)
np.save('Ytest.npy', y_test)

#Creating Model
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

#Training and Svaing the Model
checkp = ModelCheckpoint('./bit_model_lstm.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)

model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data = (X_test, y_test), callbacks = [checkp])

