# title: CNN-LSTM

# import libraries
import numpy as np
import pandas as pd
import pickle
import os
import glob
from random import shuffle
#import geopandas


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D, \
    GlobalMaxPool1D, MaxPool1D, AvgPool1D, LSTM, Bidirectional, Masking
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization

import wandb
from wandb.keras import WandbCallback

# set directory
os.chdir("your-path-to-data")

# import data
cnnlstmdata_x = np.load('Deep_input_X_scaled_36.npy')
cnnlstmdata_y = np.load('Deep_input_Y_MSE_scaled.npy') # continuous
# cnnlstmdata_y = np.load('Deep_input_Y_binary.npy') # binary

x = cnnlstmdata_x
y = cnnlstmdata_y

# split into training, validation, and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape, y_train.shape, 'train examples')
# print(x_val.shape, y_val.shape, 'validation examples')
print(x_test.shape, y_test.shape, 'test examples')

# set hyperparameters
n_features = x.shape[2]

# build a model
model = Sequential()
## convolutional layer
model.add(Masking(mask_value=-1, input_shape=(108, n_features))) # subsequent layers must be able to take masking function
model.add(Conv1D(filters=10, kernel_size=12, padding='valid', strides=1, input_shape=(108, n_features)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool1D())
## LSTM layer
model.add(LSTM(units=10, activation='tanh'))
model.add(Dropout(0.3))
## output layer
model.add(Dense(units=1, activation='linear')) # continuous
# model.add(Dense(units=1, activation='sigmoid')) # binary

model.summary()

# compile the model
rmse = tf.keras.metrics.RootMeanSquaredError()
binary = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')

opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                               amsgrad=False, name="Adam") # add decay
model.compile(loss='mse', optimizer=opt, metrics=[rmse])
# model.compile(loss=binary, optimizer=opt, metrics=['acc'])


# fit the model
model_path = 'your-path-to-model'
model_name = 'your-model-name.h5'
model_location = os.path.join(model_path, model_name)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                  mode='min', restore_best_weights=False)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_location, monitor='val_loss', save_best_only=True)
callbacks_list = [early_stop, model_checkpoint]
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1,
                    callbacks=callbacks_list)

# evaluate the model
saved_model = load_model(model_location)
train_metrics = saved_model.evaluate(x_train, y_train, verbose=0)
test_metrics = saved_model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_metrics[1], test_metrics[1]))