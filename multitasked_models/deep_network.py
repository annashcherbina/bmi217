#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:01:20 2017

@author: davidcohniii
"""

import h5py

from keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D

from keras.layers.core import Dropout, Flatten, Dense, Activation

from keras.models import Sequential

from keras.optimizers import Adam

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.layers.advanced_activations import PReLU

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l1, l2, activity_l1, activity_l2

from keras.constraints import maxnorm

import os

from math import floor

from numpy.random import randint, seed

import numpy

def data_io(file_name, train_examples):
    data = h5py.File(file_name, 'r')
    x_data = data['X']['sequence']
    y_data = data['Y']['output']
    if(file_name == 'train_data.hdf5'):
        x_data = x_data[0:train_examples, :, : , :]
        y_data = y_data[0:train_examples, :]
    return x_data, y_data

def model_development(x_train, y_train):
    
    deep_network = Sequential()
    deep_network.add(Convolution2D(250, 4, 11, border_mode = 'valid', input_shape = (1, 4, 2000), dim_ordering = 'th'))
    deep_network.add(BatchNormalization(mode = 0, axis = 1))
    deep_network.add(PReLU())
    deep_network.add(AveragePooling2D(pool_size = (1, 8), border_mode = 'valid', dim_ordering = 'th'))

    deep_network.add(Convolution2D(250,1, 9, border_mode = 'valid', dim_ordering = 'th'))
    deep_network.add(PReLU())
    deep_network.add(AveragePooling2D(pool_size=(1,8)))
    
    deep_network.add(Convolution2D(250,1,7, border_mode = 'valid', dim_ordering = 'th'))
    deep_network.add(PReLU())
    deep_network.add(AveragePooling2D(pool_size=(1,8)))
    
    deep_network.add(Flatten())
    deep_network.add(Dense(750, activity_regularizer=activity_l1(0.00001),W_constraint=maxnorm(m=7)))
    deep_network.add(PReLU())
    deep_network.add(Dropout(0.3))

    deep_network.add(Dense(61))    
    deep_network.add(Activation("sigmoid"))
    
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    deep_network.compile(loss="binary_crossentropy", optimizer = adam)
    
    return deep_network

def model_evaluation(x_train, y_train, x_validation, y_validation, x_test, y_test, deep_network, results_directory, train_batch_size):
    checkpoint = ModelCheckpoint(filepath = "optimal_deep_learning_model8.hdf5", verbose = 1, save_best_only = True)
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose=1)
    tensorboard = TensorBoard(histogram_freq=0, write_graph=True, write_images=True)
    csvlogger = CSVLogger('training_results_model8.csv', append = False)
    
    deep_network.fit(x_train, y_train, batch_size = 100, nb_epoch = 15, shuffle = True, show_accuracy= True, validation_data = (x_validation, y_validation), callbacks = [checkpoint, early_stop, tensorboard, csvlogger])
    print("Model Fitting Complete!")
    deep_network_results = deep_network.evaluate(x_test, y_test, show_accuracy = True)
    print(deep_network_results)
    return deep_network_results

train_examples = 300000

train_batch_size = 100000

data_directory = '/data/deeplearning/multitasked_model'

results_directory = '/data/deep_learning_hdf5_files'

#data_directory = '/Users/davidcohniii/Documents/multitasked_model'

#results_directory = '/Users/davidcohniii/Documents/BMI_217_Final_Project'

os.chdir(data_directory)

x_train, y_train = data_io('train_data.hdf5', train_examples)

x_validation, y_validation = data_io('valid_data.hdf5', train_examples)

x_test, y_test = data_io('test_data.hdf5', train_examples)


os.chdir(results_directory)

deep_network = model_development(x_train, y_train)
deep_network_results = model_evaluation(x_train, y_train, x_validation, y_validation, x_test, y_test, deep_network, results_directory, train_batch_size)
