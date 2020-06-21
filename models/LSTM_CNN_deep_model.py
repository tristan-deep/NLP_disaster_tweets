"""
=============================================================================
    Eindhoven University of Technology
==============================================================================
    Source Name   : LSTM_CNN_deep_model.py

    Author(s)     : Raoul Melaet
    Date          : Sat Jun 20 18:29:31 2020
==============================================================================
Model with convolutional modules + LSTM placed after that.

Hparam search conclusions:
*Tested models with varying amount of Dropout, conv filter size, number of conv modules and number of conv filters.
 the number of lstm_out = 20, as this was the best value for the separate LSTM model, and otherwise it would increase
 the amount of models in the hparam search by a lot.

*Best model with:
    -Dropout        = 0.5
    -Optimizer      = Adam
    -Conv_filt_size = 1
    -Conv_num_mods  = 1
    -Conv_num_filts = 16
*Metrics:
    -Train_acc  = 0.84
    -Val_acc    = 0.77
    -Train_loss = 0.38
    -Val_loss   = 0.58

This model is performing very similar to the model with just convolutional modules, however it is showing slightly more
overfit. This can be due to the LSTM modules, like we saw in the LSTM model.
"""

import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, LeakyReLU, AvgPool2D, MaxPooling1D, GlobalMaxPooling1D, UpSampling2D, ReLU, MaxPooling2D, BatchNormalization, \
    Reshape, Softmax, Activation, Flatten, Lambda, Conv2DTranspose, Dropout, Embedding
from tensorflow.keras.losses import MSE, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np



def create_model(max_words=10000, embedding_vecor_length=32, max_length=500, dropout = 0.2, lstm_out = 20, conv_num_modules = 2, conv_num_filters = 32, conv_filter_size = 1, l2_reg = 0, optimizer = 'adam', print_summary=True):
    model = Sequential()
    model.add(Embedding(max_words, embedding_vecor_length, input_length=max_length))
    
    for conv_num_module in np.arange(1,conv_num_modules+1):
        model.add(Conv1D(conv_num_filters*conv_num_module, conv_filter_size, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D())
        model.add(Dropout(dropout))
        

    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_out))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, name='out_layer'))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer=optimizer,
                   metrics = ['accuracy'])

    if print_summary:
        print(model.summary())
    return model


