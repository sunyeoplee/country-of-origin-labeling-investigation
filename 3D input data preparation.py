# title: 3D input data preparation

# import libraries
import numpy as np
import pandas as pd
import pickle
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# set directory
os.chdir('your-path-to-data')

# import 2D data
with open('your-data.pickle', 'rb') as f:
    data=pickle.load(f)

data.shape # (92614, 36)

# scale before making 3D data
scaler_x = MinMaxScaler(feature_range=(0,1)) # normalize the features
scaler_x.fit(data.iloc[:, 4:])
data_scaled = scaler_x.transform(data2.iloc[:, 4:])
data_scaled

# save a list of unique grid ID
grid_id_list = data['id'].unique().tolist()

# count the number of unique grids
len(grid_id_list) # 16235

# a function for adjusting month value
def length_adjust(str_) : # if month, day is single number, convert to two-numbers (ex. 6 → 06, 7 → 07)
    return (2-len(str_)) * '0' + str_

def date_convertor(date) : # combine dates (e.g., '2010-01-01' → '20100101')
    year = length_adjust(str(date.year))
    month = length_adjust(str(date.month))
    yyyymm = year+month
    return yyyymm # return 'str' type

year_month = pd.date_range(start='20100101', end='20191231', freq='M')  # turn 20100101 ~ 20191231 into 201001 ~ 201912
year_month = list(map(date_convertor, year_month))  # len(year_month) : 120
temp_columns = data2.columns[4:]  # 36

# check if feature data is 120 x 36 for each label
for i in range(16235):  # the number of unique grids
    globals()['master_table_{}'.format(i+1)] = pd.DataFrame(np.full(shape = (len(year_month), len(temp_columns)),
                                                                    fill_value=-1),
                                                            index = year_month, columns = temp_columns)
master_table_1.shape # (120, 36)

# create 2D feature dataframes for each grid
total_input = []
for i in range(16235): # the number of unique grids
    print('Process : {} | Total : 16235'.format(i+1), end='\r')
    master_table_clip = globals()['master_table_{}'.format(i+1)]   # store each master_table in master_table_clip
    clip = data[(data['id']==grid_id_list.iloc[i])]   # store unique grids in clip
    for j in range(len(clip)):
        mm = length_adjust(str(clip.iloc[j,2]))  # adjust the month value (e.g., 6 → 06, 7 → 07)
        ym = str(clip.iloc[j,1]) + mm    # combine year and month (e.g., 2010 + 01 → 201001)
        master_table_clip.loc[ym, :] = data_scaled[clip.index[j]]
    total_input.append(master_table_clip.values[:108, :]) # 108 is the maximum length of time sequences

len(total_input) # 16235 = the number of unique grids

# stack the 2D feature dataframes
total_input_stacked = total_input[0]
for s in range(len(total_input)-1):
    print('process:{}'.format(s+1), end='\r')
    total_input_stacked = np.vstack((total_input_stacked, total_input[s+1]))

real_total_input = total_input_stacked.reshape(-1, 108, 36)
real_total_input.ndim # 3

real_total_input.shape # (16235, 108, 36)

# save the 3D feature
np.save("Deep_input_X_scaled_full_36.npy", real_total_input)

# create 2D label dataframes without scaling. Scaling can be done later.
for i in range(16235):
    print('Process : {} | Total : 16235'.format(i+1), end='\r')
    master_table_clip = globals()['master_table_{}'.format(i+1)] # store each master_table in master_table_clip
    clip = data2[(data2['id']==unique_input['id'].iloc[i])] # store unique grids in clip
    for j in range(len(clip)):  # clip의 행의 개수
        mm = length_adjust(str(clip.iloc[j,2])) # adjust the month value (e.g., 6 → 06, 7 → 07)
        ym = str(clip.iloc[j,1]) + mm  # combine year and month (e.g., 2010 + 01 → 201001)
        master_table_clip.loc[ym, :] = clip.values[j][-1:]  # save only the label

#
y_grid = [] # final 3D label dataframe

for i in range(16235):
    sum_ = 0  # final label
    print('Process : {} | Total : 16235'.format(i + 1), end='\r')
    master_table_clip = globals()['master_table_{}'.format(i + 1)]

    # final label value counts
    unique, counts = np.unique(master_table_clip.iloc[-12:, 0].values, return_counts=True)  # 2019년 총적발건수
    cnt_dict = dict(zip(unique, counts))  # final label {value : count}
    del cnt_dict[-1.0]  # remove missing values

    # final label count
    if len(cnt_dict) == 0:  # if a grid had all missing values
        # add all labels by year and divide by the year available
        count = 0  # available year number
        for k in range(9):
            master_table_year = master_table_clip.iloc[12 * k:12 * (k + 1), :]  # per year
            jucval_list = list(master_table_year['label'].values)
            if all(jucval_list) != -1:  # if not all years were missing
                count += 1
                for char in jucval_list:
                    if char >= 0:  # if it is not missing,
                        sum_ += char  # add it to the final label
        y_grid.append(sum_ / count)
    else:  # if at least one year is not missing
        # count final label
        list_ = []
        for a in cnt_dict:
            list_.append(a * cnt_dict[a])  # unique value of a label * the number of the unique values
        sum_ += np.sum(list_) # add them all to get the final label
        y_grid.append(sum_)

# store in an array
y_grid = np.array(y_grid)
y_grid = y_grid.reshape(16235, 1)
y_grid.shape # (16235, 1)

# store in a dataframe and scale it
y_grid_df = pd.DataFrame(y_grid, columns=['label'])

scaler_y = MinMaxScaler(feature_range=(0,1))  # normalize the label
scaler_y.fit(y_grid_df)
y_scaled = scaler_y.transform(y_grid_df)

# save the 3D label
np.save("Deep_input_Y_MSE_scaled.npy", y_scaled)