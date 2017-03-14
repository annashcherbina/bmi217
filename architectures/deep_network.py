#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:01:20 2017

This script is responsible for constructing Basset-like convolutional neural network (CNN) models to predict chromatin accessibility for sequence data. These CNN models consist a series of convolution and Max pooling layers, before flattening the data to pass it through a series of fully-connected layers to generate a series of predictions based on the size of the label vector. This script is responsible for data I/O, model development, and model evaluation (in terms of test loss) on a test.

To effectively run my code, the results and data directories (data_directory and results_directory, respectively) should be modified based on the user's computer/server where the model is being used.

@author: davidcohniii
"""

# Libraries used in construction and evaluation of CNN model

import h5py

# Keras Packages used in CNN Model Development

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

# This function is responsible for training, test, or validation data I/O. To save computation time, the number of training set examples are reduced.
def data_io(file_name, train_examples):
    data = h5py.File(file_name, 'r')
    # I/O One-hot encoded Sequence Data
    x_data = data['X']['sequence']
    # I/O Binary Chromatin Accessibility Labels
    y_data = data['Y']['output']
    if(file_name == 'train_data.hdf5'):
        # Reduce Number of Training examples to reduce computation times
        x_data = x_data[0:train_examples, :, : , :]
        y_data = y_data[0:train_examples, :]
    return x_data, y_data

# This function is responsible for the construction of the CNN model, as outlined above. In order to test different model constructions, this function has been modified, based on those model constructions.
def model_development():
    deep_network = Sequential()
    # First Convolution/Pooling Layer, using 300 4x19 convolutional kernels, a 1x3 max pooling size, and a parametric rectified linear unit (PReLU) activation
    deep_network.add(Convolution2D(300, 4, 19, border_mode = 'valid', input_shape = (1, 4, 2000), dim_ordering = 'th', W_regularizer = l2(0.00001)))
    # Application of batch normalization, as a regularizer to prevent overfitting
    deep_network.add(BatchNormalization(mode=0, axis=1))
    deep_network.add(PReLU())
    deep_network.add(MaxPooling2D(pool_size = (1, 3), border_mode = 'valid', dim_ordering = 'th'))
    # Second Convolution/Pooling Layer, using 200 1x11 convolutional kernels, a 1x4 max pooling size, and a PReLU activation
    deep_network.add(Convolution2D(200,1,11, W_constraint=maxnorm(m=7), W_regularizer = l2(0.00001))
    # Application of batch normalization, as a regularizer to prevent overfitting
    deep_network.add(BatchNormalization(mode=0, axis=1))
    deep_network.add(PReLU())
    deep_network.add(MaxPooling2D(pool_size=(1,4)))

    # Second Convolution/Pooling Layer, using 200 1x7 convolutional kernels, a 1x4 max pooling size, and a PReLU activation
    deep_network.add(Convolution2D(200,1,7, W_constraint=maxnorm(m=7), W_regularizer = l2(0.00001)))
    # Application of batch normalization, as a regularizer to prevent overfitting
    deep_network.add(BatchNormalization(mode=0, axis=1))
    deep_network.add(PReLU())
    deep_network.add(MaxPooling2D(pool_size=(1,4)))
    
    # First Fully Connected Layer (with size 1000), and dropout of 50% to reduce overfitting
    deep_network.add(Flatten())
    deep_network.add(Dense(1000,activity_regularizer=activity_l1(0.00001),W_constraint=maxnorm(m=7)))
    deep_network.add(PReLU())
    deep_network.add(Dropout(0.5))
                     
    # Second Fully Connected Layer (with size 1000), and dropout of 50% to reduce overfitting
    deep_network.add(Dense(1000,activity_regularizer=activity_l1(0.00001),W_constraint=maxnorm(m=7)))
    deep_network.add(PReLU())
    deep_network.add(Dropout(0.5))
    
    # Final Fully Connected Layer, with a sigmoid activation, to generate model label predictions
    deep_network.add(Dense(61))    
    deep_network.add(Activation("sigmoid"))
    
    # Compile CNN model with Adam optimizer and Binary Cross Entropy Loss
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    deep_network.compile(loss = "binary_crossentropy", optimizer = adam)
    
    return deep_network
                     
# This function is responsible for training the CNN, and evaluating its performance (in terms of test loss) on a held out test set. This function also stores the architecture and weights of the best CNN model by epoch in a .hdf5 file, while outputting training/validation loss results to a .csv file. In the event that model performance does not improve (as defined by validation loss) for three epochs, the model training will end prematurely to save computation time.
def model_evaluation(x_train, y_train, x_validation, y_validation, x_test, y_test, deep_network):
    # Saving model archictecture and weights to .hdf5 file
    checkpoint = ModelCheckpoint(filepath = "optimal_deep_learning_model5.hdf5", verbose = 1, save_best_only = True)
    # Terminating model training if validation loss does not improve for three epochs
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose=1)
    # Visualization to monitor training vs. validation loss
    tensorboard = TensorBoard(histogram_freq=0, write_graph=True, write_images=True)
    # Outputting training and validation loss results, for each epoch, to a .csv file
    csvlogger = CSVLogger('training_results_model5.csv', append = False)
    # Fitting CNN model
    deep_network.fit(x_train, y_train, batch_size = 100, nb_epoch = 15, shuffle = True, validation_data = (x_validation, y_validation), callbacks = [checkpoint, early_stop, tensorboard, csvlogger])
    print("Model Fitting Complete!")
    # Evaluating CNN model performance on test set
    deep_network_results = deep_network.evaluate(x_test, y_test)
    print(deep_network_results)
    return deep_network_results
                     
# Number of Training Examples to use in model training
train_examples = 300000
                     
# Directory where sequence data is stored
data_directory = '/data/deeplearning/multitasked_model'
                     
# Directory where CAE model results should be stored
results_directory = '/data/deep_learning_hdf5_files'

#data_directory = '/Users/davidcohniii/Documents/multitasked_model'

#results_directory = '/Users/davidcohniii/Documents/BMI_217_Final_Project'

os.chdir(data_directory)

x_train, y_train = data_io('train_data.hdf5', train_examples)

x_validation, y_validation = data_io('valid_data.hdf5', train_examples)

x_test, y_test = data_io('test_data.hdf5', train_examples)


os.chdir(results_directory)

deep_network = model_development()
deep_network_results = model_evaluation(x_train, y_train, x_validation, y_validation, x_test, y_test, deep_network)
