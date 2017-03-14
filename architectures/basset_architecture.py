def getModelGivenModelOptions(self, options):
    np.random.seed(1234)
    import keras;
    from keras.models import Sequential
    from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.optimizers import Adadelta, SGD, RMSprop;
    from keras.constraints import maxnorm;
    from keras.layers.advanced_activations import PReLU
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l1, l2, activity_l1, activity_l2


    model = Sequential()
    model.add(Convolution2D(300,4,19,input_shape=(1,4,int(options.seq_length)), W_learning_rate_multiplier=10.0))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(1,3)))

    model.add(Convolution2D(200,1,11,W_learning_rate_multiplier=5.0,W_constraint=maxnorm(m=7)))
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

    model.add(Dense(options.num_labels))
    model.add(Activation("sigmoid"))

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="binary_crossentropy",optimizer=adam)
    # model.save_weights('/mnt/lab_data/kundaje/users/pgreens/projects/hematopoiesis/models/saved_weights/dec_18_yaml_150K_neg_no_hema_diff_peaks_Leuk_75M_BassetModel_10xlearnrate_noweightnorml1.h5')
    if options.init_weights_file!=None:
        model.save_weights(options.init_weights_file)

    return model;
