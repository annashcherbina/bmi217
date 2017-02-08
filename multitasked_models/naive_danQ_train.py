import argparse
import numpy as np
import h5py
import scipy.io

np.random.seed(1337) # for reproducibility

import keras;
from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop;
from keras.constraints import maxnorm;
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, activity_l1, activity_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path')
    parser.add_argument('valid_path')
    parser.add_argument('test_path')

    return parser.parse_args()

def loadData(args):
    trainmat = h5py.File(args.train_path)
    validmat = h5py.File(args.valid_path)
    testmat = h5py.File(args.test_path)

    x_train = trainmat['X']['sequence']
    y_train = trainmat['Y']['output']
    x_valid = validmat['X']['sequence']
    y_valid = validmat['Y']['output']
    x_test = testmat['X']['sequence']
    y_test = testmat['Y']['output']
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def createModel():
    print "Creating model..."

    model = Sequential()
    model.add(Convolution2D(300,4,19,input_shape=(1,4,2000)))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(1,3)))

    model.add(Convolution2D(200,1,11,W_constraint=maxnorm(m=7)))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(1,4)))

    model.add(Convolution2D(200,1,7,W_constraint=maxnorm(m=7)))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(1,4)))

    model.add(Flatten())
    model.add(Dense(1000,activity_regularizer=activity_l1(0.00001),W_constraint=maxnorm(m=7)))
    model.add(PReLU())
    model.add(Dropout(0.3))

    model.add(Dense(1000,activity_regularizer=activity_l1(0.00001),W_constraint=maxnorm(m=7)))
    model.add(PReLU())
    model.add(Dropout(0.3))

    model.add(Dense(61))
    model.add(Activation("sigmoid"))

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    print "Compiling model..."
    model.compile(loss="binary_crossentropy",optimizer=adam)
    return model

def fitAndEvaluate(model, x_train, y_train, x_valid, y_valid, x_test, y_test):
    print "Running only one epoch"

    checkpointer = ModelCheckpoint(filepath="naive_basset_best.hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.fit(x_train, 
        y_train, 
        batch_size=100, 
        nb_epoch=2, 
        shuffle="batch", 
        show_accuracy=True, 
        validation_data=(x_valid, y_valid), 
        callbacks=[checkpointer,earlystopper])

    results = model.evaluate(x_test, 
        y_test,
        show_accuracy=True)

    print results

def main():
    args = parseArgs()
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadData(args)
    model = createModel()
    fitAndEvaluate(model, x_train, y_train, x_valid, y_valid, x_test, y_test)

if __name__ == "__main__":
    main()