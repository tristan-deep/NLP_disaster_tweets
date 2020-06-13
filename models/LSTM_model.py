import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, LeakyReLU, AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Softmax, Activation, Flatten, Lambda, Conv2DTranspose, Dropout, Embedding
from tensorflow.keras.losses import MSE, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def create_model(max_words=10000, embedding_vecor_length=32, max_length=500, lstm_out=100, print_summary=True):
    # create the model
    # model = Sequential()
    # model.add(Embedding(max_words, embedding_vecor_length, input_length=max_length))
    # model.add(LSTM(lstm_out))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # if print_summary:
    #     print(model.summary())
        
    model = Sequential()
    model.add(Embedding(max_words, embedding_vecor_length, input_length = max_length))
    model.add(LSTM(lstm_out))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, name='out_layer'))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam',
                   metrics = ['accuracy'])
    
    if print_summary:
        print(model.summary())
    return model
