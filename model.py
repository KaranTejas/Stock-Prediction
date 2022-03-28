# !pip install yfinance
import keras
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

newModel=Sequential()
newModel.add(LSTM(50,return_sequences=True,input_shape=(100,50)))
newModel.add(LSTM(50,return_sequences=True))
newModel.add(LSTM(50))
newModel.add(Dense(1))
newModel.summary()
newModel.compile(loss='mean_squared_error',optimizer='adam')

newModel.save('Save_Model')