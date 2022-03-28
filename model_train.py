import keras
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

ticker = yf.Ticker('AAPL')
df = ticker.history(period='max')
df1 = df.reset_index()['Close']
df2 = pd.DataFrame(df1)

scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:1],df1[training_size:len(df1),:1]

def create_dataset(dataset, time_step = 1, input_size = 1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-input_size+1):
		a = []
		for j in range(time_step) :
			a.append(dataset[i+j:(i+j+input_size), 0])
		dataX.append(a)
		dataY.append(dataset[i + time_step + input_size - 1, 0])
	return array(dataX), array(dataY)
 
time_step = 100
input_size = 50
X_train, y_train = create_dataset(train_data, time_step, input_size)
X_test, y_test = create_dataset(test_data, time_step, input_size)

savedModel = keras.models.load_model('Save_Model')

savedModel.evaluate(X_test ,y_test )

savedModel.fit(X_test, y_test, epochs=30, verbose=1)

savedModel.evaluate(X_test ,y_test )

savedModel.save('Save_Model')

train_predict = savedModel.predict(X_train)
test_predict = savedModel.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

l = len(df2) - len(test_predict)
test = df2[l:]
train = df2[:l]
test['Predictions'] = test_predict