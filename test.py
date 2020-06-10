'''
For now this is just a file where I tried some stuff. 

'''
import numpy as np
import pandas as pd
import string
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, LeakyReLU, AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Softmax, Activation, Flatten, Lambda, Conv2DTranspose, Dropout, Embedding
from tensorflow.keras.losses import MSE, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

from DataLoader import LoadTweets

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split()])# if word.lower() not in STOPWORDS])


def built_LSTM(embed_dim=50, lstm_out=64, print_summary=True) :


    model = Sequential()
    model.add(Embedding(max_words, embed_dim, input_length = max_len))
    model.add(LSTM(lstm_out))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, name='out_layer'))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam',\
                   metrics = ['accuracy'])
      
    if print_summary == True :
        print(model.summary())
    
    return model

if __name__ == '__main__' :
    # simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']
    # vect = CountVectorizer()
    # vect.fit(simple_train)
    # vect.get_feature_names()
    # simple_train_dtm = vect.transform(simple_train)
    # simple_train_dtm.toarray()
    
    # df = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
    
    # the model will remember only the top 2000 most common words
    max_words = 2000
    max_len = 50
    
    # set text_only to true so only text is fed into Tokenizer.
    tokenizer_gen = LoadTweets(split='train', batch_size=32, shuffle=False, text_only=True)
      
    token = Tokenizer(num_words=max_words, lower=True, split=' ')
    token.fit_on_texts(tokenizer_gen)
    sequences = token.texts_to_sequences(tokenizer_gen)
    train_sequences_padded = pad_sequences(sequences, maxlen=max_len)
    

    model = built_LSTM(embed_dim = 50, lstm_out = 64)