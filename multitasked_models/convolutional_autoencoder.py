#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:01:20 2017

@author: davidcohniii
"""

import h5py

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D

from keras.models import Sequential

from keras.optimizers import Adadelta

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import pdb

import os

from math import floor

import numpy

def data_io(file_name):
    data = h5py.File(file_name, 'r')
    x_data = data['X']['sequence']
    if(file_name == 'train_data.hdf5'):
        x_data = x_data[0:(train_examples), :, 0:3, :]
    else:
        x_data = x_data[:, :, 0:3, :]
    return x_data

def model_development(x_train):
    convolutional_autoencoder = Sequential()
    convolutional_autoencoder.add(Convolution2D(64, 3, 3, activation = 'relu', border_mode = 'same', input_shape = (1, 3, 2000)))
    convolutional_autoencoder.add(MaxPooling2D((1, 3)))
    convolutional_autoencoder.add(Convolution2D(64, 3, 3, activation = 'relu', border_mode = 'same'))
    convolutional_autoencoder.add(MaxPooling2D((1, 3)))
    convolutional_autoencoder.add(Convolution2D(64, 3, 3, activation = 'relu', border_mode = 'same'))
    convolutional_autoencoder.add(MaxPooling2D((1, 3)))
    convolutional_autoencoder.add(Convolution2D(64, 3, 3, activation = 'relu', border_mode = 'same'))
    convolutional_autoencoder.add(UpSampling2D((1, 3)))
    convolutional_autoencoder.add(Convolution2D(64, 3, 3, activation = 'relu', border_mode = 'same'))
    convolutional_autoencoder.add(UpSampling2D((1, 3)))
    convolutional_autoencoder.add(Convolution2D(64, 3, 3, activation = 'relu', border_mode = 'same'))
    convolutional_autoencoder.add(UpSampling2D((1, 3)))
    convolutional_autoencoder.add(Convolution2D(64, 3, 3, activation = 'sigmoid', border_mode = 'same'))
    ada_delta = Adadelta(lr = 1.0, rho = 0.9, epsilon = 1e-08)
    convolutional_autoencoder.compile(optimizer = ada_delta, loss = 'hinge')
    return convolutional_autoencoder

def model_evaluation(x_train, x_validation, x_test, convolutional_autoencoder, results_directory):
    checkpoint = ModelCheckpoint(filepath = results_directory + "optimal_CAE_model.hdf5", verbose = 1, save_best_only = True)
    early_stop = EarlyStopping(monitor='val_loss', patience = 3, verbose=1)
    convolutional_autoencoder.fit(x_train, x_train, nb_epoch = 10, batch_size = 128, shuffle = True, validation_data = (x_validation, x_validation), callbacks = [checkpoint, early_stop])
    CAE_results = convolutional_autoencoder.evaluate(x_test, x_test, batch_size = 64)
    print(CAE_results)
    return CAE_results

train_examples = 300000

data_directory = '/data/deeplearning/multitasked_model'

results_directory = '/home/bmi217_admin'

os.chdir(data_directory)

x_train = data_io('train_data.hdf5')

x_validation = data_io('valid_data.hdf5')

x_test = data_io('test_data.hdf5')

os.chdir(results_directory)

convolutional_autoencoder = model_development(x_train)

CAE_results = model_evaluation(x_train, x_validation, x_test, convolutional_autoencoder, results_directory)
