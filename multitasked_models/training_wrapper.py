import argparse
import imp
import numpy as np
import h5py
import scipy.io

global modelModule
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('train_path')
    parser.add_argument('valid_path')
    parser.add_argument('test_path')
    return parser.parse_args()

def loadData(args):
    trainmat = h5py.File(args.train_path)
    validmat = h5py.File(args.valid_path)
    testmat = h5py.File(args.test_path)

    x_train = trainmat['X']['sequence'][0:100000]
    y_train = trainmat['Y']['output'][0:100000]
    x_valid = validmat['X']['sequence']
    y_valid = validmat['Y']['output']
    x_test = testmat['X']['sequence']
    y_test = testmat['Y']['output']
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def fitAndEvaluate(model, model_path, 
            x_train, y_train, x_valid, y_valid, x_test, y_test):
    print "Running only one epoch"

    model_output_path = "best_model_%s.hdf5" % (model_path[:-3])
    checkpointer = ModelCheckpoint(filepath=model_output_path, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tensorboard = TensorBoard(histogram_freq=0, write_graph=False, write_images=True)
    csvlogger = CSVLogger('training_results', append = True)

    model.fit(x_train, 
        y_train, 
        batch_size=100, 
        nb_epoch=1, 
        shuffle="batch", 
        show_accuracy=True, 
        validation_data=(x_valid, y_valid), 
        callbacks=[checkpointer,earlystopper, csvlogger, tensorboard])

    results = model.evaluate(x_test, 
        y_test,
        show_accuracy=True)

    print results

def main():
    args = parseArgs()
    model_path=args.model_path
    modelModule = imp.load_source('module.name', args.model_path)
    x_train, y_train, x_valid, y_valid, x_test, y_test = loadData(args)
    model = modelModule.createModel()
    fitAndEvaluate(model,model_path,x_train, y_train, x_valid, y_valid, x_test, y_test)

if __name__ == "__main__":
    main()
    
