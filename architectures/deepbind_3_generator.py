# A generator that provides a Keras model for training,
# according to the guidelines in train_naive.py
def createModel():
    import numpy as np
    np.random.seed(1337) # for reproducibility

    # imports
    import keras;
    from keras.models import Sequential
    from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.optimizers import Adadelta, SGD, RMSprop;
    from keras.layers.advanced_activations import PReLU
    from keras.constraints import maxnorm;
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l1, l2, activity_l1, activity_l2
    
    print "Creating model..."

    model = Sequential()
    # convolution layer with 16 filters of size 4x16
    model.add(Convolution2D(16,4,16,input_shape=(1,4,2000), bias=True))
    # ReLU rectification activation
    model.add(Activation("relu"))
    # maxpooling to decrease feature set
    model.add(MaxPooling2D(pool_size=(1,4)))
    # dropout to prevent overfitting
    model.add(Dropout(0.5))    
    model.add(Flatten())
    # fully connected layer and a sigmoidal activation
    model.add(Dense(61, bias=True))
    model.add(Activation("sigmoid"))
    
    # training optimizer
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    print "Compiling model..."
    model.compile(loss="binary_crossentropy",optimizer=adam)
    return model
