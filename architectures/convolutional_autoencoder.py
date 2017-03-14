#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:01:20 2017

This script is responsible for creating the Convolutional Autoencoder model (CAE) that performs dimensionality
reduction by training a deep neural network (DNN) to reproduce an inputted sequence. The CAE consists of a series of convolution and max pooling layers to downsample the data. For our experiments, we used either one or two sets of convolutional and max pooling layers. From there, the data is passed through an equivalent number of convolutional and upsampling layers. Finally, the data is passed through a single filter convolutional layer, with a sigmoid activation, to be able to regenerate a sequence of equivalent dimensions to the input. This script handles the data I/O, model construction, model training and model evaluation (in terms of test loss) on a test set.

To effectively run my code, the results and data directories (data_directory and results_directory, respectively) should be modified based on the user's computer/server where the model is being used.

@author: davidcohniii
"""
# Packages used in CAE script
import h5py

# DNN packages from Keras deep learning library
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D

from keras.models import Sequential

from keras.optimizers import Adadelta

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.layers.advanced_activations import PReLU

import os

# This function performs handles the I/O of our training, test, and validation chromatin accessibility data sets. To reduce the computation time associated with model training, this function also reduces the number of training examples used to a defined constant.
def data_io(file_name):
    data = h5py.File(file_name, 'r')
    # Read in sequence data
    x_data = data['X']['sequence']
    if(file_name == 'train_data.hdf5'):
        # Reduce training data set size for computation purposes
        x_data = x_data[0:(train_examples), :, :, :]
    return x_data

# This function is responsible for the construction of the CAE model, as outlined above. In order to test different model constructions, this function has been modified, based on those model constructions.
def model_development():
    convolutional_autoencoder = Sequential()
    # First Convolution/Pooling Layer, using 50 4x4 convolutional kernels, a 1x4 pooling size, and a parametric rectified linear unit (PReLU) activation
    convolutional_autoencoder.add(Convolution2D(50, 4, 4, border_mode = 'same', input_shape = (1, 4, 2000), dim_ordering = 'th'))
    convolutional_autoencoder.add(PReLU())
    convolutional_autoencoder.add(MaxPooling2D(pool_size = (1, 4), border_mode = 'same', dim_ordering = 'th'))
    # Second Convolution/Pooling Layer, using 50 1x4 convolutional kernels, a 1x4 pooling size, and a PReLU activation
    convolutional_autoencoder.add(Convolution2D(50, 1, 4, border_mode = 'same', dim_ordering = 'th'))
    convolutional_autoencoder.add(PReLU())
    convolutional_autoencoder.add(MaxPooling2D(pool_size = (1, 4), border_mode = 'same' ,dim_ordering = 'th'))
    # First Convolution/Upsampling Layer, using 50 1x4 convolutional kernels and a 1x4 upsampling size
    convolutional_autoencoder.add(Convolution2D(50, 1, 4, border_mode = 'same', dim_ordering = 'th'))
    convolutional_autoencoder.add(UpSampling2D((1, 4), dim_ordering = 'th'))
    # Second Convolution/Upsampling Layer, using 50 1x4 convolutional kernels and a 1x4 upsampling size
    convolutional_autoencoder.add(Convolution2D(50, 4, 4, border_mode = 'same', dim_ordering = 'th'))
    convolutional_autoencoder.add(UpSampling2D((1, 4), dim_ordering = 'th'))
    # Final Convolutional Layer to regenerate sequence of equivalent dimensions to input
    convolutional_autoencoder.add(Convolution2D(1, 1, 4, activation = 'sigmoid', border_mode = 'same', dim_ordering = 'th'))
    # Definition of Adadelta optimizer and binary cross entropy loss function for CAE
    ada_delta = Adadelta(lr = 0.1, rho = 0.9, epsilon = 1e-08)
    convolutional_autoencoder.compile(optimizer = ada_delta, loss = 'binary_crossentropy')
    return convolutional_autoencoder

# This function is responsible for training the CAE, and evaluating its performance (in terms of test loss) on a held out test set. This function also stores the architecture and weights of the best CAE model in a .hdf5 file, while outputting training/validation loss results to a .csv file. In the event that model performance does not improve (as defined by validation loss) for three epochs, the model training will end prematurely to save computation time.
def model_evaluation(x_train, x_validation, x_test, convolutional_autoencoder):
    # Saving model archictecture and weights to .hdf5 file
    checkpoint = ModelCheckpoint("optimal_CAE_model8.hdf5", verbose = 1, save_best_only = True)
    # Terminating model training if validation loss does not improve for three epochs
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1)
    # Visualization to monitor training vs. validation loss
    tensorboard = TensorBoard(histogram_freq = 0, write_graph = True, write_images = True)
    # Outputting training and validation loss results, for each epoch, to a .csv file
    csvlogger = CSVLogger('training_results_cae8.csv', append = False)
    # Fitting CAE model
    convolutional_autoencoder.fit(x_train, x_train, nb_epoch = 15, shuffle = True, validation_data = (x_validation, x_validation), callbacks = [checkpoint, early_stop, tensorboard, csvlogger])
    # Evaluating CAE model on test set
    CAE_results = convolutional_autoencoder.evaluate(x_test, x_test)
    print(CAE_results)
    return CAE_results

# Number of Training Examples to use in model training
train_examples = 300000
# Directory where sequence data is stored
data_directory = '/data/deeplearning/multitasked_model'
# Directory where CAE model results should be stored
results_directory = '/home/bmi217_admin'

os.chdir(data_directory)

x_train = data_io('train_data.hdf5')

x_validation = data_io('valid_data.hdf5')

x_test = data_io('test_data.hdf5')

os.chdir(results_directory)

convolutional_autoencoder = model_development()

CAE_results = model_evaluation(x_train, x_validation, x_test, convolutional_autoencoder)
