"""Algorithms for explanation generator."""

import numpy as np

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.ensemble import IsolationForest


def isolation_forest(input):
    '''
    params:
        input (pd.DataFrame): input data to trian the model
    '''
    
    outliers_fraction= 0.01
    rdm = np.random.RandomState(10)
    model = IsolationForest(contamination = outliers_fraction, random_state=rdm)
    model.fit(input)
    
    return model


def autoencoder(input, nb_epoch=100, batch_size=64):
    '''
    A 6-layer conventional autoencoder model.

    params:
        input (pd.DataFrame): input data to train the model
        nb_epoch (int): number of epoch the model will perform
        batch_size (int): size of each batch of data enter to the model
    
    output:
        model: trained autoencoder
    '''

    input_dim = input.shape[1]
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(units=128, activation='relu')(input_layer)
    encoded = Dense(units=64, activation='relu')(encoded)
    encoded = Dense(units=32, activation='relu')(encoded)
    encoded = Dense(units=16, activation='relu')(encoded)
    decoded = Dense(units=32, activation='relu')(encoded)
    decoded = Dense(units=64, activation='relu')(decoded)
    decoded = Dense(units=128, activation='relu')(decoded)
    decoded = Dense(units=input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    autoencoder.fit(input, input, epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_split=0.1, verbose=0, callbacks=[earlystopper])

    return autoencoder





