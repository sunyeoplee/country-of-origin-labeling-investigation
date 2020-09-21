# title: Machine learning

# import libraries
import sklearn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import xgboost
import lightgbm as lgb
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import matplotlib
import warnings

warnings.filterwarnings(action='ignore')

# set directory
os.chdir('your-path-to-data')

with open('machine-learning-input-data.pickle', 'rb') as f:
    data_original = pkl.load(f)
data = data_original
data.info()

# remove missing data
data = data.dropna(axis=0)

# split into training, validation, test set (repeat for each month)
data_train = data[(data['year'] <= 2018) & (data['month'] == 1)] # loop over months from 1 to 12
data_test = data[(data['year'] <= 2019) & (data['month'] == 1)] # loop over months from 1 to 12

x_train = data_train[:, 3:-1]
y_train = pd.DataFrame(data_train.iloc[:,-1], index = data_train.index, columns = ['label']) # reserve index
x_train, x_val, y_train, y_val = train_test_split(x_Train y_train, test_size = 0.2, random_state=0) # split into trainig, validation set

x_test = data_train[:, 3:-1]
y_test = pd.DataFrame(data_train.iloc[:,-1], index = data_train.index, columns = ['label']) # reserve index

# scaling
scaler_x = MinMaxScaler(feature_range=(0,1)) # normalize the features
scaler_x.fit(x_train)
x_train_scaled = scaler_x.transform(x_train)
x_val_scaled = scaler_x.transform(x_val)
x_test_scaled = scaler_x.transform(x_test)

scaler_y = MinMaxScaler(feature_range=(0,1)) # normalize the features
scaler_y.fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# compare model performance across different models
## AdaBoost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

regr = AdaBoostRegressor(random_state=0, n_estimators=50)
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.4f" % (rmse))

## Bagging
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor

regr = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)\
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.4f" % (rmse))

## ExtraTrees
from sklearn.ensemble import ExtraTreesRegressor

regr = ExtraTreesRegressor(n_estimators=100, random_state=0)
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.4f" % (rmse))

## Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

regr = GradientBoostingRegressor(n_estimators=100, random_state=0)\
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.6f" % (rmse))

## Stacking
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

estimators = [('lr', RidgeCV()), ('svr', LinearSVR(random_state=0))]
regr = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=100, random_state=0))
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.4f" % (rmse))

## Voting
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

r1 = LinearRegression()
r2 = RandomForestRegressor(n_estimators=100, random_state=0)
regr = VotingRegressor([('lr', r1), ('rf', r2)])
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.6f" % (rmse))

## Histogram-based Gradient Boosting
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

regr = HistGradientBoostingRegressor(random_state = 0)
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.4f" % (rmse))

## LightGBM
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

regr = lgb.LGBMRegressor(random_state=0, n_jobs=-1)
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.4f" % (rmse))

## XGBoost
import xgboost as xgb

regr = XGBRegressor(random_state = 0, n_jobs = -1)
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val_scaled, y_pred))
print("RMSE: %.4f" % (rmse))

# model evaluation (ExtraTrees is chosen from validation)
from sklearn.ensemble import ExtraTreesRegressor

regr = ExtraTreesRegressor(n_estimators=100, random_state=0)
regr.fit(x_train_scaled, y_train_scaled)

y_pred = regr.predict(x_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred))
print("RMSE: %.6f" % (rmse))

y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=['label'])
y_pred_real = scaler_y.inverse_transform(y_pred_df) # inverse the transformation
y_pred_real

# combine observed label and predicted label
final_result = pd.concat([y_test.reset_index(), pd.DataFrame(y_pred_real)], axis = 1)
final_result.rename(columns = {0 : 'y_pred'}, inplace = True)

# store the result in a csv file
final_result.to_csv('your-file-name', encoding='cp949')

# store the model
with open('your-model-name.pkl', 'wb') as f:
    pkl.dump(regr, f)

# visualize feature importance
regr.feature_importances_

matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # set font
matplotlib.rcParams['axes.unicode_minus'] = False    # prevent font break

plt.figure(figsize=(10,10))
colors = ['red','orangered', 'orange', 'gold', 'khaki','aquamarine', 'darkturquoise', 'steelblue', 'royalblue', 'purple']

(pd.Series(regr.feature_importances_, index=x_train.columns)
   .nlargest(35)
   .plot(kind='barh', color = colors).invert_yaxis())

plt.title('feature importance', fontsize = 24, pad = 15)
plt.ylabel('feature', fontsize = 15)
plt.yticks(fontsize = 15)
# plt.savefig('your-path-to-graph', dpi=200, facecolor='#eeeeee', bbox_inches='tight')
plt.show()




##########################################################################################
# for all months
## the codes above is looped over months.

model_list = ['month', 'AdaBoost', 'BaggingRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor',
              'StackingRegressor', 'VotingRegressor', 'HistGradientBoostingRegressor', 'LightGBM', 'XGBoost']
all_rmse = pd.DataFrame(columns=model_list)
all_rmse['month'] = np.arange(1, 13)
for i in all_rmse['month'].values:
    print("Process : {} | Total : 12".format(i), end='\r')

    dataset_train = dataset[(dataset['year'] <= 2018) & (dataset['month'] == i)]
    dataset_test = dataset[(dataset['year'] == 2019) & (dataset['month'] == i)]
    x_train = dataset_train.iloc[:, 3:-1]
    y_train = pd.DataFrame(dataset_train.iloc[:, -1], index=dataset_train.index,
                           columns=['label'])
    x_test = dataset_test.iloc[:, 3:-1]
    y_test = pd.DataFrame(dataset_test.iloc[:, -1], index=dataset_test.index,
                          columns=['label'])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                      random_state=0)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x.fit(x_train)
    x_train_scaled = scaler_x.transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    x_val_scaled = scaler_x.transform(x_val)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    y_val_scaled = scaler_y.transform(y_val)

    # AdaBoost
    regr_ada = AdaBoostRegressor(random_state=0, n_estimators=50)
    regr_ada.fit(x_train_scaled, y_train_scaled)
    y_pred_ada = regr_ada.predict(x_val_scaled)
    rmse_ada = np.sqrt(mean_squared_error(y_val_scaled, y_pred_ada))
    all_rmse.iloc[i - 1, 1] = rmse_ada

    # Bagging
    regr_bag = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)\
    regr.fit(x_train_scaled, y_train_scaled)
    y_pred_bag = regr_bag.predict(x_val_scaled)
    rmse_bag = np.sqrt(mean_squared_error(y_val_scaled, y_pred_bag))
    all_rmse.iloc[i - 1, 2] = rmse_bag

    # ExtraTrees
    regr_et = ExtraTreesRegressor(n_estimators=100, random_state=0) \
    regr.fit(x_train_scaled, y_train_scaled)
    y_pred_et = regr_et.predict(x_val_scaled)
    rmse_et = np.sqrt(mean_squared_error(y_val_scaled, y_pred_et))
    all_rmse.iloc[i - 1, 3] = rmse_et

    # Gradient Boosting
    regr_gbr = GradientBoostingRegressor(n_estimators=100, random_state=0)
    regr.fit(x_train_scaled, y_train_scaled)
    y_pred_gbr = regr_gbr.predict(x_val_scaled)
    rmse_gbr = np.sqrt(mean_squared_error(y_val_scaled, y_pred_gbr))
    all_rmse.iloc[i - 1, 4] = rmse_gbr

    # Stacking
    estimators = [('lr', RidgeCV()), ('svr', LinearSVR(random_state=0))]
    regr_sr = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=100, random_state=0))
    regr.fit(x_train_scaled, y_train_scaled)
    y_pred_sr = regr_sr.predict(x_val_scaled)
    rmse_sr = np.sqrt(mean_squared_error(y_val_scaled, y_pred_sr))
    all_rmse.iloc[i - 1, 5] = rmse_sr

    # Voting
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=100, random_state=0)
    regr_vr = VotingRegressor([('lr', r1), ('rf', r2)])
    regr.fit(x_train_scaled, y_train_scaled)
    y_pred_vr = regr_vr.predict(x_val_scaled)
    rmse_vr = np.sqrt(mean_squared_error(y_val_scaled, y_pred_vr))
    all_rmse.iloc[i - 1, 6] = rmse_vr

    # Histogram-based Gradient Boosting
    regr_hgbr = HistGradientBoostingRegressor(random_state=0)
    regr.fit(x_train_scaled, y_train_scaled)
    y_pred_hgbr = regr_hgbr.predict(x_val_scaled)
    rmse_hgbr = np.sqrt(mean_squared_error(y_val_scaled, y_pred_hgbr))
    all_rmse.iloc[i - 1, 7] = rmse_hgbr

    # LightGBM
    regr_lgbm = lgb.LGBMRegressor(random_state=0, n_jobs=-1)
    regr_lgbm.fit(x_train_scaled, y_train_scaled)
    y_pred_lgbm = regr_lgbm.predict(x_val_scaled)
    rmse_lgbm = np.sqrt(mean_squared_error(y_val_scaled, y_pred_lgbm))
    all_rmse.iloc[i - 1, 8] = rmse_lgbm

    # XGBoost
    regr_xgb = XGBRegressor(random_state=0, n_jobs=-1)
    regr_xgb.fit(x_train_scaled, y_train_scaled)
    y_pred_xgb = regr_xgb.predict(x_val_scaled)
    rmse_xgb = np.sqrt(mean_squared_error(y_val_scaled, y_pred_xgb))
    all_rmse.iloc[i - 1, 9] = rmse_xgb

all_rmse

fig, ax = plt.subplots(figsize = (25, 15))
x = list(map(lambda x: str(x) + 'month', all_rmse['month']))
line1 = ax.plot(x, all_rmse['AdaBoost'], color = 'red')
line2 = ax.plot(x, all_rmse['Bagging'], color = 'orange')
line3 = ax.plot(x, all_rmse['ExtraTrees'],color = 'gold')
line4 = ax.plot(x, all_rmse['GradientBoosting'],color = 'aquamarine')
line5 = ax.plot(x, all_rmse['Stacking'], color = 'darkturquoise')
line6 = ax.plot(x, all_rmse['Voting'], color = 'purple')
line7 = ax.plot(x, all_rmse['HistGradientBoosting'], color = 'steelblue')
line8 = ax.plot(x, all_rmse['LightGBM'], color = 'blue')
line9 = ax.plot(x, all_rmse['XGBoost'], color = 'gray')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylim(0, 0.15)
ax.legend(['AdaBoost RMSE',
            'Bagging RMSE',
            'ExtraTrees RMSE',
            'GradientBoosting RMSE',
            'Stacking RMSE',
            'Voting RMSE',
            'HistGradientBoosting RMSE',
            'LightGBM RMSE',
            'XGBoost RMSE'], loc = 0, fontsize = 'large')
fig.suptitle('머신러닝모델 RMSE 비교', fontsize = 30, y=0.915)
plt.savefig('./머신러닝_시각화/머신러닝모델_성능비교.png', dpi=200, facecolor='#eeeeee', bbox_inches='tight')
plt.show()