# A program that trains a model provided by a generator
# specified through the command line.  The model_path
# command line argument must link to a valid Python
# module with a function createModel(), that returns
# the Keras model to train.

# Imports
import argparse
import imp
import numpy as np
import h5py
import scipy.io

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

global modelModule

# Parses the arguments passed by train_naive.sh, specifically the
# path to the model generator, the training dataset path, the
# validation dataset path, and the test dataset path.
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('train_path')
    parser.add_argument('valid_path')
    parser.add_argument('test_path')
    return parser.parse_args()

# Loads into memory the  validation data, training data, and
# test data.
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

# Trains the network on the training data, and saves the best model
# after every epoch.  After training, performs a quick evaluation of
# the trained model.
def fitAndEvaluate(model, model_path, 
            x_train, y_train, x_valid, y_valid, x_test, y_test):
    NUM_EPOCHS = 5
    print "Running %d epochs" % (NUM_EPOCHS)

    model_output_path = "best_model_%s.hdf5" % (model_path[:-3])
    checkpointer = ModelCheckpoint(filepath=model_output_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tensorboard = TensorBoard(histogram_freq=0, write_graph=False, write_images=True)
    csvlogger = CSVLogger('training_results', append = True)

    model.fit(x_train, 
        y_train, 
        batch_size=100, 
        nb_epoch=NUM_EPOCHS, 
        shuffle="batch", 
        show_accuracy=True, 
        validation_data=(x_valid, y_valid), 
        callbacks=[checkpointer,earlystopper, csvlogger, tensorboard])

    results = model.evaluate(x_test, 
        y_test,
        show_accuracy=True)

    print results

# Main function
def main():
    args = parseArgs()
    model_path=args.model_path
    modelModule = imp.load_source('module.name', args.model_path)
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadData(args)
    model = modelModule.createModel()
    fitAndEvaluate(model, args.model_path, x_train, y_train, x_valid, y_valid, x_test, y_test)

if __name__ == "__main__":
    main()
    
