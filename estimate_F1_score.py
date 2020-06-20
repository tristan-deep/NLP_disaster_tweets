# This file estimates the f1-score of the model based on training set. 
# The 'cv' parameter in cross_validate_score indicates K-fold parameter for cross validation
# Remember to change the path to the csv files
# In this file, whole training dataset is used

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow.keras as keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn import feature_extraction, model_selection
from keras.wrappers.scikit_learn import KerasClassifier
from models.LSTM_model import create_model  

# remember to change these parameters
vocabulary_size = 10000
max_length = 500
vector_length = 16 
lstm_out = 64

def tokenizer_tweets(max_words=vocabulary_size):
  data = pd.read_csv(Path('nlp-getting-started', 'train' + '.csv'))
  list_IDs = list(range(len(data))) 
  all_text = data.text.to_list()

  # Create Tokenizer Object
  tokenizer = Tokenizer(num_words=max_words, filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n', lower=False, split=" ")

  # Train the tokenizer to the texts
  tokenizer.fit_on_texts(all_text)

  return tokenizer


tokenizer = tokenizer_tweets(vocabulary_size)

data = pd.read_csv(Path('nlp-getting-started', 'train' + '.csv'))
X = {'keyword'  : data.keyword.to_list(), 
      'location' : data.location.to_list(),
      'text'     : data.text.to_list()}

X_text = X['text']
# Convert list of strings into list of lists of integers
X_sequences = tokenizer.texts_to_sequences(X_text)
X_pad = sequence.pad_sequences(X_sequences, maxlen=max_length)

y = np.array(data.target.to_list())

def model_fn(max_word=vocabulary_size,length=vector_length,max_length=max_length,lstm_out=lstm_out):
    return create_model(vocabulary_size,vector_length,max_length,lstm_out)


model = KerasClassifier(build_fn=model_fn, epochs=30, batch_size=32, verbose=0)
model.fit(X_pad,y)

scores = model_selection.cross_val_score(model, X_pad, y, cv=3, scoring="f1")
print('F1-score: ')
print(scores)


