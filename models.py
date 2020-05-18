import config
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, BatchNormalization, ReLU, Dropout, Conv2D, Conv2DTranspose, Flatten, UpSampling2D, UpSampling3D, MaxPooling2D, MaxPool3D, ZeroPadding2D, Cropping2D, Cropping3D, Reshape, Conv3D, Conv3DTranspose

from tensorflow.keras.layers import Dense, Input, TimeDistributed, LayerNormalization, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
import os
def modular(filters, latentDim, path, batch=False, dropout=False):

    checkpoint_dir = os.path.dirname(path)
    model = Sequential()
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    model.add(InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1)))
    for f in filters:

        model.add(Conv2D(f, (3, 3), strides=2, activation = 'relu', padding="same"))
        if(dropout): model.add(Dropout(0.2))
        if(batch): model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2, 2)))
    if(latentDim is not None):
        # model.add(Flatten())
        model.add(Conv2D(latentDim, (1,1), strides=1, activation = 'relu', padding="same"))
    # model.add(Flatten())
    # model.add(Dense(latentDim))
    for f in reversed(filters):
        # apply a CONV_TRANSPOSE => RELU => BN operation
        model.add(Conv2DTranspose(f, (3, 3), activation = 'relu', strides=2, padding="same"))
        if(dropout): model.add(Dropout(0.2))
        if(batch): model.add(BatchNormalization())

    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    # model.add(Cropping2D((1)))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer = Adam())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                # monitor='val_loss',
                                                save_weights_only=True,
                                                # save_best_only=True,
                                                verbose=1,
                                                save_freq='epoch')

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print('latestdasdasdas')
    print(latest)
    if latest is not None:
         model.load_weights(latest)
         print('weights loaded')


    return model, cp_callback

def final2D(path):
    checkpoint_dir = os.path.dirname(path)
    model = Sequential()
    model.add(InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1)))
    model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10)))


    model.add(Conv2DTranspose(64, (3, 3), strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(128, (3, 3), strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    model.summary()

    model.compile(loss='mean_squared_error', optimizer = Adam())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                # monitor='val_loss',
                                                save_weights_only=True,
                                                # save_best_only=True,
                                                verbose=1,
                                                save_freq='epoch')

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print('latestdasdasdas')
    print(latest)
    if latest is not None:
         model.load_weights(latest)
         print('weights loaded')
    model.save(path+'model.h5')


    return model, cp_callback


def final2DStacked(path):
    checkpoint_dir = os.path.dirname(path)
    model = Sequential()
    model.add(InputLayer(input_shape=(config.NUM_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)))

    model.add(Conv2D(128, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10), data_format="channels_first"))


    model.add(Conv2DTranspose(64, (3, 3), strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(128, (3, 3), strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10), data_format="channels_first"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(config.NUM_CHANNELS, (3, 3), activation='sigmoid', padding='same'), data_format="channels_first")
    model.summary()

    model.compile(loss='mean_squared_error', optimizer = Adam())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                # monitor='val_loss',
                                                save_weights_only=True,
                                                # save_best_only=True,
                                                verbose=1,
                                                save_freq='epoch')

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print('latestdasdasdas')
    print(latest)
    if latest is not None:
         model.load_weights(latest)
         print('weights loaded')
    model.save(path+'model.h5')


    return model, cp_callback

def final3D(path):
    checkpoint_dir = os.path.dirname(path)
    model = Sequential()
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    model.add(InputLayer(input_shape=(config.NUM_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH, 1)))

    model.add(Conv3D(128, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool3D(pool_size=(1,2,2)))
    

    model.add(Conv3D(64, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPool3D(pool_size=(2,2,2)))


    model.add(Conv3D(32, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv3D(64, 3, padding="same", stride=(2,2,2), kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())
    # model.add(UpSampling3D(size=(2,2,2)))

    model.add(Conv3D(128, 3, padding="same",stride=(1,2,2), kernel_regularizer=tf.keras.regularizers.l2(1e-10)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(UpSampling3D(size=(1,2,2)))
    model.add(Conv3D(1, 3, activation='sigmoid', padding='same'))

    model.compile(loss='mean_squared_error', optimizer = Adam())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                # monitor='val_loss',
                                                save_weights_only=True,
                                                # save_best_only=True,
                                                verbose=1,
                                                save_freq='epoch')

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print('latestdasdasdas')
    print(latest)
    if latest is not None:
        model.load_weights(latest)
        print('weights loaded')
    model.save(path+'model.h5')


    return model, cp_callback

def modular2d3dims(filters, latentDim, path, batch=False, dropout=False, filter_size=3):

    checkpoint_dir = os.path.dirname(path)
    model = Sequential()
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    model.add(InputLayer(input_shape=(config.NUM_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH)))
    for f in filters:
        # print('assssdad')
        model.add(Conv2D(f, filter_size, strides=2, activation = 'relu', padding="same", data_format="channels_first"))
        if(dropout): model.add(Dropout(0.2))
        if(batch): model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2, 2)))
    if(latentDim is not None):
        # model.add(Flatten())
         model.add(Conv2D(latentDim, (1,1), strides=1, activation = 'relu', padding="same", data_format="channels_first"))
         if(batch): model.add(BatchNormalization())
    for f in reversed(filters):
        # apply a CONV_TRANSPOSE => RELU => BN operation
        model.add(Conv2DTranspose(f, filter_size, activation = 'relu', strides=2, padding="same", data_format="channels_first"))
        if(dropout): model.add(Dropout(0.2))
        if(batch): model.add(BatchNormalization())

    # model.add(Reshape((20, config.IMG_HEIGHT, config.IMG_WIDTH)))
    model.add(Conv2D(config.NUM_CHANNELS, filter_size, activation='sigmoid', padding='same', data_format="channels_first"))
    # model.add(Cropping2D((1)))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer = Adam())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                # monitor='val_loss',
                                                save_weights_only=True,
                                                # save_best_only=True,
                                                verbose=1,
                                                save_freq='epoch')

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print('latestdasdasdas')
    print(latest)
    if latest is not None:
         model.load_weights(latest)
         print('weights loaded')


    return model, cp_callback

def conv3d(filters, latentDim, path, batch=False, dropout=False, filter_size=3):

    checkpoint_dir = os.path.dirname(path)
    model = Sequential()
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    model.add(InputLayer(input_shape=(config.NUM_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH, 1)))

    
    for f in filters:
        model.add(Conv3D(f, filter_size, strides=2, activation = 'relu', padding="same"))
        if(dropout): model.add(Dropout(0.2))
        if(batch): model.add(BatchNormalization())
        # model.add(MaxPool3D(pool_size=(1,2,2)))
    if(latentDim is not None):
        # model.add(Flatten())
         model.add(Conv3D(latentDim, 1, strides=1, activation = 'relu', padding="same"))
         if(batch): model.add(BatchNormalization())
    # model.add(Flatten())
    # model.add(Dense(latentDim))
    for f in reversed(filters):
        # apply a CONV_TRANSPOSE => RELU => BN operation
        model.add(Conv3DTranspose(f, filter_size, activation = 'relu', strides=2, padding="same"))
        if(dropout): model.add(Dropout(0.2))
        if(batch): model.add(BatchNormalization())

    model.add(Conv3D(1, filter_size, activation='sigmoid', padding='same'))
    if(config.NUM_CHANNELS%(2**len(filters))!=0):
        dim=config.NUM_CHANNELS
        for i in range(len(filters)):
            if(dim%2!=0):
                dim=int(dim/2)
                dim+=1
            else:
                dim=int(dim/2)
        print(dim)
        croppingFactor=int((dim*(2**len(filters))-config.NUM_CHANNELS)/2)
        model.add(Cropping3D(cropping=((croppingFactor,croppingFactor),(0,0),(0,0))))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer = Adam())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                # monitor='val_loss',
                                                save_weights_only=True,
                                                # save_best_only=True,
                                                verbose=1,
                                                save_freq='epoch')

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print('latestdasdasdas')
    print(latest)
    if latest is not None:
         model.load_weights(latest)
         print('weights loaded')


    return model, cp_callback


def modularMaxpool(filters, latentDim, path):

    checkpoint_dir = os.path.dirname(path)
    model = Sequential()
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    # model.add(ZeroPadding2D(padding=(1),input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(InputLayer(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1)))

    for f in filters:
        print('assssdad')
        model.add(Conv2D(f, (3, 3), activation = 'relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    if(latentDim !=0):
        # model.add(Flatten())
         model.add(Conv2D(latentDim, (1,1), strides=1, activation = 'relu', padding="same"))
    # model.add(Flatten())
    # model.add(Dense(latentDim))
    for f in reversed(filters):
        # apply a CONV_TRANSPOSE => RELU => BN operation
        model.add(Conv2DTranspose(f, (3, 3), activation = 'relu', strides=2, padding="same"))

    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    # model.add(Cropping2D((1)))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer = Adam())
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                # monitor='val_loss',
                                                save_weights_only=True,
                                                # save_best_only=True,
                                                verbose=1,
                                                save_freq='epoch')

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('latestdasdasdas')
    print(latest)
    if latest is not None:
         model.load_weights(latest)
         print('weights loaded')


    return model, cp_callback