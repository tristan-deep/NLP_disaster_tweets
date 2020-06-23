"""
=============================================================================
    Eindhoven University of Technology
==============================================================================
    Source Name   : Bidirectional_LSTM_model.py

    Author(s)     : Raoul Melaet
    Date          : Sat Jun 20 18:27:14 2020
==============================================================================
Bidirectional LSTM Model

Theoretical motivation:
    Model with bidirectional LSTM's. The bidirectionality of the LSTM will basically create two LSTM's one from the past to 
    future and the other one from future to past. In this way you preserve information from the future, and using the two 
    hidden states combined you are able preserve information from both past and future.

    What they are suited for is a very complicated question but BiLSTMs show very good results as they can understand 
    context better, and this is what we actually want with our tweet sentiment NLP model ;) I will try to explain through an 
    example;

    Lets say we try to predict the next word in a sentence, on a high level what a unidirectional LSTM will see is
        "The boys went to ...."

    And will try to predict the next word only by this context, with bidirectional LSTM you will be able to see information 
    further down the road for example
        Forward LSTM:
            "The boys went to ..."
        Backward LSTM:
            "... and then they got out of the pool"


Hparam search conclusions:
*Tested models with varying amount of Dropout, conv filter size, number of conv modules and number of conv filters. 
 the number of lstm_out = 20, as this was the best value for the separate LSTM model, and otherwise it would increase
 the amount of models in the hparam search by a lot.

*Best model with:
    -Dropout        = 0.5
    -Optimizer      = Adam
    -LSTM out       = 20
*Metrics:
    -Train_acc  = 0.96
    -Val_acc    = 0.72
    -Train_loss = 0.10
    -Val_loss   = 2.57

This model is performing very similar to the model plain LSTM model, and it is not really improving our performance.
Maybe our sequences are simply too short or there is not enough data available such that the network is actually able
to do meaningful predictions? idk.
"""

import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, LeakyReLU, AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Softmax, Activation, Flatten, Lambda, Conv2DTranspose, Dropout, Embedding, Bidirectional
from tensorflow.keras.losses import MSE, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop


def create_model(max_words=10000, embedding_vecor_length=32, max_length=500, lstm_out=100, conv_num_modules = 2, conv_num_filters = 32, conv_filter_size = 1, dropout = 0.5, optimizer = 'adam', print_summary=True):
    model = Sequential()
    model.add(Embedding(max_words, embedding_vecor_length, input_length = max_length))
    model.add(Bidirectional(LSTM(lstm_out)))
    #model.add(Dense(256))
    #model.add(Activation('relu'))
    model.add(Dropout(dropout))


    model.add(Dense(1, name='out_layer'))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer=optimizer,
                   metrics = ['accuracy'])
    
    if print_summary:
        print(model.summary())
    return model
