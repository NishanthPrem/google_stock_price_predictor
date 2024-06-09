import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = df_train.iloc[:, 1:2].values

# Feature Scaling using Normalization

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(train_set)

# 60 timesteps are used (Data from past 3 months)

X_train, y_train = [], []
for i in range(60, len(train_set_scaled)):
    X_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the data

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

regressor.add(
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


#%%
df_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_price = df_test.iloc[:, 1:2].values
df_total = pd.concat((df_test['Open'], df_train['Open']), axis=0)

inputs = df_total[len(df_total) - len(df_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

#%%
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = regressor.predict(X_test)



