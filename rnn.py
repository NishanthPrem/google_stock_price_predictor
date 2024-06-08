import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = df_train.iloc[:, 1:2].values

#Feature Scaling using Normalization

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(train_set)

