import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.externals import joblib

dataset = pd.read_csv("Dataset_accuracy/Dataset.csv")

target = dataset.real_distance
train = dataset.drop(['real_distance'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=2)

reg = ensemble.GradientBoostingRegressor()
reg.fit(x_train, y_train)
prediction = reg.predict(x_test)
mse = mean_squared_error(y_test, prediction)
joblib.dump(reg, 'gradient_boosting_model.pkl')


