#!/usr/bin/python3

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from time import time
import numpy as np
import joblib

#setting
dataset_file = 'dataset.npy'
test_data_len = 10000

dataset = np.load(dataset_file)
param_train = dataset[:-test_data_len, :-1]
param_test = dataset[-test_data_len:,:-1]

label_train = dataset[:-test_data_len,-1]
label_test = dataset[-test_data_len:, -1]

#print('param data', np.shape(param_train), np.shape(param_test))
#print('label data', np.shape(label_train), np.shape(label_test))

start_time = time()
print('trying regression linear')
reg = LinearRegression()
reg.fit(param_train, label_train)

print('predicting with random forest')
reg_prediction = reg.predict(param_test)
end_time = time()
reg_rmse_err = mean_squared_error(label_test, reg_prediction)

print('linear regression RMSE -> %s, processing time -> %s'%(reg_rmse_err, end_time-start_time))

##########################################

start_time = time()
print('trying MLP')
mlp = MLPRegressor(hidden_layer_sizes = (5,2))
mlp.fit(param_train, label_train)

print('predicting with MLP')
mlp_prediction = mlp.predict(param_test)
end_time = time()
mlp_rmse_err = mean_squared_error(label_test, mlp_prediction)

print('MLP RMSE -> %s, processing time -> %s'%(mlp_rmse_err, end_time-start_time))

##########################################

start_time = time()
print('trying random forest')
random_forest = RandomForestRegressor()
random_forest.fit(param_train, label_train)

print('predicting with random forest')
random_forest_prediction = random_forest.predict(param_test)
end_time = time()
random_forest_rmse_err = mean_squared_error(label_test, random_forest_prediction)

print('random forest RMSE -> %s, processing time -> %s'%(random_forest_rmse_err, end_time-start_time))

##########################################

start_time = time()
print('trying gradient boost')
g_boost = GradientBoostingRegressor()
g_boost.fit(param_train, label_train)

print('predicting with gradient boost')
g_boost_prediction = g_boost.predict(param_test)
end_time = time()
g_boost_rmse_err = mean_squared_error(label_test, g_boost_prediction)

print('gradient boost RMSE -> %s, processing time -> %s'%(g_boost_rmse_err, end_time-start_time))

##############################################

start_time = time()
print('trying SVR')
svr = SVR()
svr.fit(param_train, label_train)

print('predicting with SVR')
svr_prediction = svr.predict(param_test)
end_time = time()
svr_rmse_err = mean_squared_error(label_test, svr_prediction)

print('SVR RMSE -> %s, processing time -> %s'%(svr_rmse_err, end_time-start_time))

#save the best model
joblib.dump(random_forest, 'random_forest.pkl')
