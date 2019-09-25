#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.activations import relu, softmax
from keras.optimizers import Adam, Nadam
from keras.callbacks import EarlyStopping
from keras_tqdm import TQDMNotebookCallback
import talos as ta
from talos.metrics.keras_metrics import fmeasure_acc

def createmodel(optimizer='adam',
                activation='relu',
                initmode='uniform',
                dropout=0.2):
    """Constructs Sequential NN model based on Talos optimization parameters

    Inputs
    ------
    optimizer:  string | Default: 'adam'
                optimizer to use for deep learning model

    activation: string | Default: 'relu'
                activation to use for deep learning model

    initmode:   string | Default: 'uniform'
                type of model initialization

    dropout:    float between 0 and 1 | Default: 0.2
                Dropout rate to apply to layer input

    Returns
    -------
    model:      sequential model based on input parameters
    """

    model = Sequential()
    model.add(Dense(64,
                    activation=activation,
                    kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation=activation,
                    kernel_initializer=kernel_initializer))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
