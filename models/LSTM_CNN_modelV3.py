import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, LeakyReLU, AvgPool2D,  MaxPooling1D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Conv1D, Softmax, Activation, Flatten, Lambda, Conv2DTranspose, Dropout, Embedding, Bidirectional
from tensorflow.keras.losses import MSE, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

def create_model(max_words=10000, embedding_vecor_length=32, max_length=500, lstm_out=100, dropout = 0.5, optimizer = 'adam', print_summary=True, ):
    model = Sequential()
    model.add(Embedding(max_words, embedding_vecor_length, input_length = max_length))
    model.add((Conv1D(filters=32, kernel_size=5, padding='same', activation = 'relu')))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_out))
    #model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, name='out_layer'))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer=optimizer,
                   metrics = ['accuracy'])
    
    if print_summary:
        print(model.summary())
    return model
