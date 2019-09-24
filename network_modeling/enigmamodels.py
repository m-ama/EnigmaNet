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
def createmodel(x_train, y_train, params):
    """Constructs Sequential NN model based on Talos optimization parameters

    Inputs
    ------
    x_train     Training dataset
    y_train     Training class labels
    params:     Talos gridsearch parameters

    Returns
    -------
    model:      Sequential model based on input parameters
    history:    Fitted model based on training set
    """
    # Initialising the ANN
    model = Sequential()

    # Add initial layer
    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))

    # Adding dropout to prevent overfitting
    model.add(Dropout(params['dropout']))

    # Adding hidden layers
    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))

    # Adding the output layer
    model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    # criterion loss and optimizer
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](),
                  metrics=['acc', fmeasure_acc])

    # Fitting the ANN to the Training set
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(x_train, y_train,
                        validation_data=[x_test, y_test],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=False,
                        callbacks=[TQDMNotebookCallback(leave_inner=False,
                                                        leave_outer=True)])
    return history, model