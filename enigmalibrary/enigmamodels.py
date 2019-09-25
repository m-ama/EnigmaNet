#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from keras.optimizers import SGD, Adam, RMSprop

def CreateSequentialModel(input_dim=None,
                          optimizer='Adam',
                          activation='relu',
                          initmode='uniform',
                          dropout=0.2,
                          units_L1=5,
                          units_L2=10,
                          L2_reg_penalty=0.01):
    """Constructs Sequential NN model based on Talos optimization parameters

    Inputs
    ------
    optimizer:  string | Default: 'adam'
                optimizer to use for deep learning model

    learnrate:  float between 0 and 1 | Default: 0.1
                learning rate of optimizer

    activation: string | Default: 'relu'
                activation to use for deep learning model

    initmode:   string | Default: 'uniform'
                type of model initialization

    dropout:    float between 0 and 1 | Default: 0.2
                Dropout rate to apply to layer input

    units_L1:   int >= 0 | Default: 5
                number of units in first dense layer

    units_L2:   int >= 0 | Default: 10
                number of units in second dense layer

    L2_penalty: float between 0 and 1 | Default: 0.01
                penalty on second dense layer's kernel regularizer

    Returns
    -------
    model:      sequential model based on input parameters
    """
    if input_dim is None:
        raise Exception('Please enter number of input dimensions to '
                        'initialize a Sequential model')
    model = Sequential()
    model.add(Dense(input_dim=input_dim,
                    units=units_L1,
                    activation=activation,
                    kernel_initializer=initmode))
    model.add(Dropout(dropout))
    model.add(Dense(units=units_L2,
                    activation=activation,
                    kernel_initializer=initmode,
                    kernel_regularizer=regularizers.l2(L2_reg_penalty)))
    model.add(Dense(units=1,
                    activation=activation))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

