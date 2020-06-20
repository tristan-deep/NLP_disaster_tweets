"""
=============================================================================
    Eindhoven University of Technology
==============================================================================
    Source Name   : LSTM_model.py

    Author(s)     : Raoul Melaet
    Date          : Sat Jun 20 18:32:14 2020
==============================================================================
This is just a basic LSTM model.

Hparam search conclusions:
*Best model with:
    -Dropout    = 0.2
    -LSTM_out   = 20
    -Optimizer  = RMSprop
*Metrics:
    -Train_acc  = 0.95
    -Val_acc    = 0.72
    -Train_loss = 0.13
    -Val_loss   = 1.91

Model does overfit quite a lot. Simpler/more complex models does not seem to reduce the overfit and also the optimizer does not seem to make a large difference.
Small size of the dataset could also play a role with the overfit.
"""

import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, LeakyReLU, AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Softmax, Activation, Flatten, Lambda, Conv2DTranspose, Dropout, Embedding
from tensorflow.keras.losses import MSE, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def create_model(max_words=10000, embedding_vecor_length=32, max_length=500, lstm_out=100, dropout = 0.5, optimizer = 'adam', print_summary=True, ):
    model = Sequential()
    model.add(Embedding(max_words, embedding_vecor_length, input_length = max_length))
    model.add(LSTM(lstm_out))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, name='out_layer'))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer=optimizer,
                   metrics = ['accuracy'])
    
    if print_summary:
        print(model.summary())
    return model
