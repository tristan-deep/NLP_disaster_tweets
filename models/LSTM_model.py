import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

def create_model(max_words=10000, embedding_vecor_length=32, max_length=500, lstm_out=100, print_summary=True):
    # create the model
    model = Sequential()
    model.add(Embedding(max_words, embedding_vecor_length, input_length=max_length))
    model.add(LSTM(lstm_out))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    if print_summary:
        print(model.summary())
    return model
