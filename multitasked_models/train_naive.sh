# train_naive.sh
# --------------
# Trains a neural network through training_wrapper.py.
# 
# Requires training_wrapper.py in the same directory.
# 
# A bash script that takes a filename of a model generator
# as a parameter.
# 
# The absolute paths to the HDF5 files containing the training
# data, the validation data, and the test data should be
# changed as necessary.

#THEANO_FLAGS=device=gpu,floatX=float32 python training_wrapper.py $1 /data/deeplearning/multitasked_model/train_data.hdf5 /data/deeplearning/multitasked_model/valid_data.hdf5 /data/deeplearning/multitasked_model/test_data.hdf5

CUDA_VISIBLE_DEVICES=0 python training_wrapper.py $1 /data/deeplearning/multitasked_model/train_data.hdf5 /data/deeplearning/multitasked_model/valid_data.hdf5 /data/deeplearning/multitasked_model/test_data.hdf5
