"""## DataLoader"""

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from pathlib import Path
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

def TokenizeTweets(max_words=10000):
  data = pd.read_csv(Path('dataset', 'train' + '_set.csv'))
  list_IDs = list(range(len(data))) 
  all_text = data.text.to_list()

  # Create Tokenizer Object
  tokenizer = Tokenizer(num_words=max_words, filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n', lower=False, split=" ")

  # Train the tokenizer to the texts (training data)
  tokenizer.fit_on_texts(all_text)

  return tokenizer

class LoadTweets(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, tokenizer, split, batch_size=32, n_classes=2, shuffle=True, vocabulary_size = 10000, max_length=500):
        'Initialization'
        self.batch_size = batch_size
        self.split = split
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data = pd.read_csv(Path('dataset', self.split + '_set.csv'))
        self.all_text = self.data.text.to_list()

        # Train the tokenizer to the texts
        self.tokenizer = tokenizer

        # not using ids but rather rows in csv file for convenience
        self.list_IDs = list(range(len(self.data)))

        self.vocabulary_size = vocabulary_size

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization

        # Generate data, we can change this later when we know exactly
        # in what kind of format we think data should be fed into network.
        data = self.data.loc[list_IDs_temp, :]

        X = {'keyword': data.keyword.to_list(),
             'location': data.location.to_list(),
             'text': data.text.to_list()}

        X_text = X['text']
        # Convert list of strings into list of lists of integers
        X_sequences = self.tokenizer.texts_to_sequences(X_text)
        # Truncate and pad input sequences
        X_pad = sequence.pad_sequences(X_sequences, maxlen=self.vocabulary_size)

        if self.split == 'train':
            y = data.target.to_list()
            y = [np.expand_dims(y, axis=1)]
            #y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        elif self.split == 'val':
            y = data.target.to_list()
            y = [np.expand_dims(y, axis=1)]
            # y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        elif self.split == 'test':
            # for test set there are no labels
            y = None
            
        return X_pad, y

if __name__ == '__main__':
    max_words =10000
    max_length = 500
    tokenizer = TokenizeTweets(max_words=max_words)

    gen = LoadTweets(tokenizer, split='train', batch_size=1, shuffle=False, max_words = max_words, max_length=max_length)

    # example that prints all the tweets of the first batch
    batch = gen[0]  # first of len(gen) batches

    X = batch[0]  # 0 -> keywords/location/text, 1 -> target
    text_first_batch = gen.tokenizer.sequences_to_texts(X)
    print(text_first_batch)
