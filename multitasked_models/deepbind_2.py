def createModel():
    import numpy as np
    np.random.seed(1337) # for reproducibility

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
    model.add(Convolution2D(16,4,16,input_shape=(1,4,2000), bias=True))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1,4)))
    model.add(Dropout(0.0))    
    model.add(Flatten())
    model.add(Dense(61, bias=True))
    model.add(Activation("sigmoid"))
    
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    print "Compiling model..."
    model.compile(loss="binary_crossentropy",optimizer=adam)
    return model
