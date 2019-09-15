import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, tree, svm, ensemble

dataset = pd.read_csv("Dataset_accuracy/Dataset.csv")

target = dataset.real_distance
train = dataset.drop(['real_distance'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=2)

regressors = [
    linear_model.LinearRegression(),
    linear_model.Ridge(),
    linear_model.Lasso(),
    linear_model.ElasticNet(),
    linear_model.BayesianRidge(),
    linear_model.RANSACRegressor(),
    svm.SVR(),
    ensemble.GradientBoostingRegressor(),
    tree.DecisionTreeRegressor(),
    ensemble.RandomForestRegressor()
]

for reg in regressors:
    reg.fit(x_train, y_train)
    
    name = reg.__class__.__name__
    print("*"*30)
    print(name)
    
    predictions = reg.predict(x_test)
    
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error: {}".format(mse))