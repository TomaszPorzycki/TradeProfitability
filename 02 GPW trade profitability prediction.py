# -*- coding: utf-8 -*-
"""
@author: Tomasz Porzycki
Trade profitability prediction for trades based on various indicators signals
Mainly used MACD, simple moving average, expontential moving average
Assumptions:
    1) Trade is profitable if, profit >0
    2) Buy / sell happen the following day of the signal
    3) Buy / sell are taken 10% from the open price towards close price

Machine learning models
- Binary classification: 1 - profit, 0 - loss
- A separate model for each company / ticker
- Model is trained vs optimal precision

    1) Linear Support Vector Classifier
    2) Decision Tree Classifier
    3) Random Forest Classifier
    4) Gradient Boosting Classifier
    5) XGBoost Classifier
    6) Keras classifier
        
Machine learning features:
    - indicators: macd, rsi, sts, trix, wiliamsR, cci, roc, ult, fi, mfi, bop, eom, ao, sma, ema
    - features:
        signal day volume vs previous days average volume
        signal day close price vs previous days close price, previous days average price, signal day open price
        signal day close price vs previos days price range
        signal day close price vs purchase day open price
"""

#Import libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#pd.set_option("display.max_columns", None)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.metrics import Precision

#Upload the data
analysis_result = pd.read_csv('003_analysis_result.csv', index_col=0, parse_dates=[2,9])

#PREPARE DATA FOR MODELLING
#Clean the data
tickers = list(analysis_result['Ticker'].unique())
tickers.remove('GIF')
tickers.remove('HUB')
analysis_result = analysis_result[analysis_result['Ticker'].isin(tickers)]

#Setup dataframe columns for analysis
columns_original = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
columns_index = ['macd_hist', 'rsi_index', 'sts_index', 'trix_index', 'wiliamsR', 'cci_index', 'roc_index', 'ult_index', 'fi_index', 'mfi_index', 'bop_index', 'eom_index', 'obv_index', 'ao_index', 'atr_index', 'vr_index', 'adi_index']
columns_signals = ['macd_signal', 'rsi_signal', 'sts_signal', 'trix_signal', 'wiliamsR_signal', 'cci_signal', 'roc_signal', 'ult_signal', 'fi_signal', 'mfi_signal', 'bop_signal', 'eom_signal', 'ao_signal']
columns_sma = ['sma10_5', 'sma20_5', 'sma20_10', 'sma50_10', 'sma50_20', 'sma100_20', 'sma100_50', 'sma200_50', 'sma200_100']
columns_ema = ['ema10_5', 'ema20_5', 'ema20_10', 'ema50_10', 'ema50_20', 'ema100_20', 'ema100_50', 'ema200_50', 'ema200_100']
columns_sma_signal = ['sma10_5_signal', 'sma20_5_signal', 'sma20_10_signal', 'sma50_10_signal', 'sma50_20_signal', 'sma100_20_signal', 'sma100_50_signal', 'sma200_50_signal', 'sma200_100_signal']
columns_ema_signal = ['ema10_5_signal', 'ema20_5_signal', 'ema20_10_signal', 'ema50_10_signal', 'ema50_20_signal', 'ema100_20_signal', 'ema100_50_signal', 'ema200_50_signal', 'ema200_100_signal']
columns_features = ['p_volume_1', 'p_volume_5', 'p_volume_10', 'p_volume_20',
                    'p_day_close', 'p_day_high',
                    'p_day_close1', 'p_day_close2', 'p_day_close3', 'p_day_close4', 'p_day_close5',
                    'p_day_open1',
                    'p_avg_close3', 'p_avg_close5', 'p_avg_close10', 'p_avg_close20',
                    'p_range_close3', 'p_range_open3', 'p_range_close5', 'p_range_open5', 'p_range_close10', 'p_range_open10',
                    'p_max_close5', 'p_max_close10', 'p_max_close20', 'p_max_high5', 'p_max_high10', 'p_max_high20']
columns_features_purchase_open = ['ps_open_close']
columns_analysis = ['type', 'Date_purchase', 'Open_purchase', 'Close_purchase', 'invested_stocks', 'invested_money', 'profit', 'trade_days', 'profitable']
columns_analysis_today = ['type']
columns_features_analysis = ['profitable_1', 'profitable_2', 'profitable_3', 'previous_exit1', 'previous_exit2', 'previous_exit3']
signals_for_analysis = ['sma50_20_signal','sma50_10_signal','sma20_5_signal','sma20_10_signal','sma10_5_signal','sma100_50_signal','macd_signal']

#Define features and target dataframe columns for machine learning model
target_columns = ['profitable'] #'profit'
feature_columns = columns_analysis_today + columns_features_analysis + columns_sma + columns_ema + columns_index + columns_features_purchase_open + columns_features + columns_sma_signal + columns_ema_signal + columns_signals

#Split data into data and target
data = analysis_result[feature_columns]
target = analysis_result[target_columns]
data.reset_index(inplace=True, drop=True)
target.reset_index(inplace=True, drop=True)

#Split categorical data into columns (one-hot encoding)
data = pd.concat([data, pd.get_dummies(data['type'], prefix='type')], axis=1, join="inner")
data = data.drop(['type'], axis=1)

#Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(data, target, stratify=target, random_state=0, test_size=0.2)

#Scale data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Create a dataframe to store predictons
summary_prediction = pd.DataFrame()
summary = []

#MODELLING

#Linear Support Vector Classifier

#Class ratio - setup ration due to unbalanced data (60-70% trades are non profitable)
ratioSVC = round(y_train.sum() / y_train.count(), 2)
#Train model
lsvc = LinearSVC(C=10, class_weight={1:ratioSVC, 0:(1-ratioSVC)})
lsvc.fit(X_train, y_train)
#Accuracy
print("SVC accuracy on training set: %f" % lsvc.score(X_train, y_train))
print("SVC accuracy on test set: %f" % lsvc.score(X_test, y_test))
#Prediction
y_pred = lsvc.predict(X_test)
matrix= confusion_matrix(y_test, y_pred)
disp = plot_confusion_matrix(lsvc, X_test, y_test, cmap=plt.cm.Blues)
print(disp.confusion_matrix)
plt.show()
#Print Precision / Recall
print('Precision: ', round(matrix[1,1] / (matrix[0,1]+matrix[1,1])*100,1))
print('Recall: ', round(matrix[1,1] / (matrix[1,0]+matrix[1,1])*100,1))
#Store predictions
summary_prediction['svc'] = y_pred

#Decision Tree Classifier
#Modelling
tree = DecisionTreeClassifier(random_state=0, criterion= 'gini', max_depth=12, splitter='best', class_weight={1:ratioSVC, 0:(1-ratioSVC)})
tree.fit(X_train, y_train)
print("DT accuracy on training set: %f" % tree.score(X_train, y_train))
print("DT accuracy on test set: %f" % tree.score(X_test, y_test))
#Prediction
y_pred = tree.predict(X_test)
matrix= confusion_matrix(y_test, y_pred)
disp = plot_confusion_matrix(tree, X_test, y_test, cmap=plt.cm.Blues)
print(disp.confusion_matrix)
plt.show()

print('Precision: ', round(matrix[1,1] / (matrix[0,1]+matrix[1,1])*100,1))
print('Recall: ', round(matrix[1,1] / (matrix[1,0]+matrix[1,1])*100,1))
#Store prediction
summary_prediction['tree'] = y_pred

#Random Forest Classifier
#Modelling
forest = RandomForestClassifier(n_estimators=20, random_state=0, criterion='entropy', max_depth=12, class_weight={1:ratioSVC, 0:(1-ratioSVC)})
forest.fit(X_train, y_train)
print("accuracy on training set: %f" % forest.score(X_train, y_train))
print("accuracy on test set: %f" % forest.score(X_test, y_test))
#Prediction
y_pred = forest.predict(X_test)
matrix= confusion_matrix(y_test, y_pred)
#class_names = churn.target_names
disp = plot_confusion_matrix(forest, X_test, y_test, cmap=plt.cm.Blues)
print(disp.confusion_matrix)
plt.show()

print('Precision: ', round(matrix[1,1] / (matrix[0,1]+matrix[1,1])*100,1))
print('Recall: ', round(matrix[1,1] / (matrix[1,0]+matrix[1,1])*100,1))
#Store predictions
summary_prediction['forest'] = y_pred

#Gradient Boosting Classifier
#Modelling
gbrt = GradientBoostingClassifier(random_state=0, n_estimators=100, max_depth=4)
gbrt.fit(X_train, y_train)

print("accuracy on training set: %f" % gbrt.score(X_train, y_train))
print("accuracy on test set: %f" % gbrt.score(X_test, y_test))

y_pred = gbrt.predict(X_test)
matrix= confusion_matrix(y_test, y_pred)
disp = plot_confusion_matrix(gbrt, X_test, y_test, cmap=plt.cm.Blues)
print(disp.confusion_matrix)
plt.show()

print('Precision: ', round(matrix[1,1] / (matrix[0,1]+matrix[1,1])*100,1))
print('Recall: ', round(matrix[1,1] / (matrix[1,0]+matrix[1,1])*100,1))
#Store predictions
summary_prediction['gbrt'] = y_pred

#XGBoost Classifier
#Modelling
xgb = XGBClassifier(learning_rate = 0.3,
                    gamma = 5,
                    max_depth=6,
                    min_child_weight=1,
                    n_estimators=10,
                    alpha=5,
                    reg_lambda=1,
                    scale_pos_weight= round(analysis_result[analysis_result['profitable'] == 0].shape[0] / analysis_result[analysis_result['profitable'] == 1].shape[0])
                   )
xgb.fit(X_train, y_train)

print("accuracy on training set: %f" % xgb.score(X_train, y_train))
print("accuracy on test set: %f" % xgb.score(X_test, y_test))

y_pred = xgb.predict(X_test)
matrix= confusion_matrix(y_test, y_pred)
disp = plot_confusion_matrix(xgb, X_test, y_test, cmap=plt.cm.Blues)
print(disp.confusion_matrix)
plt.show()

print('Precision: ', round(matrix[1,1] / (matrix[0,1]+matrix[1,1])*100,1))
print('Recall: ', round(matrix[1,1] / (matrix[1,0]+matrix[1,1])*100,1))
#Store predictions
summary_prediction['xgb'] = y_pred

#Keras
#Modelling
# Define the keras model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=Precision())

# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=int(X_train.shape[0]/30))

#Evaluate accuracy
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

#Predict
y_pred = model.predict_classes(X_test)
y_pred = np.reshape(y_pred,-1)
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

print('Precision: ', round(matrix[1,1] / (matrix[0,1]+matrix[1,1])*100,1))
print('Recall: ', round(matrix[1,1] / (matrix[1,0]+matrix[1,1])*100,1))

#Store predictions
summary_prediction['keras'] = y_pred


#SUMMARY
#Compare predictions vs test data
coefficient = 1 #[x/sum(summary_forecast) for x in summary_forecast]
summary_prediction[['svc','tree','gbrt','forest','xgb','keras']] = summary_prediction[['svc','tree','gbrt','forest','xgb','keras']]*coefficient
summary_prediction['final_pred'] = summary_prediction.sum(axis=1)
summary_prediction['final_pred'] = summary_prediction['final_pred'].apply(lambda x: 1 if x>=5 else 0)
summary_prediction['original'] = y_test.values

matrix = confusion_matrix(summary_prediction['original'], summary_prediction['final_pred'])
print(matrix)

print('Precision: ', round(matrix[1,1] / (matrix[0,1]+matrix[1,1])*100,1))
print('Recall: ', round(matrix[1,1] / (matrix[1,0]+matrix[1,1])*100,1))