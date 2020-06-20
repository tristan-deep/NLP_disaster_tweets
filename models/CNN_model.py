"""
=============================================================================
    Eindhoven University of Technology
==============================================================================
    Source Name   : CNN_model.py

    Author(s)     : Raoul Melaet
    Date          : Sat Jun 20 18:28:12 2020
==============================================================================
Model with convolutional modules.

Theoretical motivation:
Convnets make sense to use for computer vision, and comparing to that, in vision the filters slide over local patches
in the image, but in NLP we typically use filters that slide over full rows of the vector (in which the words or 
tokenized words are. 

The conv_filter_size (window size) determines the size of the window that hovers over the input sentence. The idea is
that the kernel is going to hover over a certain area and apply the convolution operation.

Hparam search conclusions:
*Tested models with varying amount of Dropout, conv filter size, number of conv modules and number of conv filters. 
*Best model with:
    -Dropout        = 0.5
    -Optimizer      = Adam
    -Conv_filt_size = 1
    -Conv_num_mods  = 2
    -Conv_num_filts = 16  
*Metrics:
    -Train_acc  = 0.77
    -Val_acc    = 0.76
    -Train_loss = 0.49
    -Val_loss   = 0.52

Compared to the LSTM model, the overfit reduced by a lot, and overall performance increased. Furthermore, we could conclude that
a very simple 2 module conv net, is performing better at predicting the NLP sequence data, than a more complex one. 
"""
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, LeakyReLU, AvgPool2D, MaxPooling1D, GlobalMaxPooling1D, UpSampling2D, ReLU, MaxPooling2D, BatchNormalization, \
    Reshape, Softmax, Activation, Flatten, Lambda, Conv2DTranspose, Dropout, Embedding
from tensorflow.keras.losses import MSE, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np



def create_model(max_words=10000, embedding_vecor_length=32, max_length=500, dropout = 0.2, lstm_out = 20, conv_num_modules = 2, conv_num_filters = 32, conv_filter_size = 1, optimizer = 'adam', print_summary=True, ):
    model = Sequential()
    model.add(Embedding(max_words, embedding_vecor_length, input_length=max_length))

    for conv_num_module in np.arange(1,conv_num_modules+1):
        model.add(Conv1D(conv_num_filters*conv_num_module, conv_filter_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D())
        model.add(Dropout(dropout))

    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, name='out_layer'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    if print_summary:
        print(model.summary())
    return model


