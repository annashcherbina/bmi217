#THEANO_FLAGS=device=gpu,floatX=float32 python training_wrapper.py $1 /data/deeplearning/multitasked_model/train_data.hdf5 /data/deeplearning/multitasked_model/valid_data.hdf5 /data/deeplearning/multitasked_model/test_data.hdf5

CUDA_VISIBLE_DEVICES=0 python training_wrapper.py $1 /data/deeplearning/multitasked_model/train_data.hdf5 /data/deeplearning/multitasked_model/valid_data.hdf5 /data/deeplearning/multitasked_model/test_data.hdf5
