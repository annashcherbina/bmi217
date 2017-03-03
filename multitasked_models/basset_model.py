def createModel():
    import numpy as np
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
