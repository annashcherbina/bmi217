from keras import backend as K
import h5py
import numpy as np


#real data
data=h5py.File("/srv/scratch/annashch/deeplearning/gecco/inputs/gecco.sampled.one.output/test_data.hdf5")
#just look at first 100k examples
X=data['X']['sequence'][0:100000]

#scrambled data
X_scrambled=np.random.shuffle(X)
print("got real & scrambled data")

#load the model



#The layer number
n = 3
# with a Sequential model
get_nth_layer_output = K.function([model.layers[0].input],
                                  [model.layers[n].output])
layer_output = get_nth_layer_output([X])[0]
