from DataLoader import LoadTweets, TokenizeTweets
from models.LSTM_model import create_model
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
from pathlib import Path

import numpy as np

#Make a folder to save model weights, and
run = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = Path("logs/"+run)

#check if the directory to save the model in exists
os.makedirs(os.path.dirname(logdir), exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=logdir)

"""Create model"""
max_length = 100
vocabulary_size = 5000
embedding_vecor_length = 32
lstm_out = 100

# Create Tokenizer Object
tokenizer = TokenizeTweets(vocabulary_size)
gen = LoadTweets(tokenizer, split='train', batch_size=1, shuffle=False, vocabulary_size=vocabulary_size, max_length=max_length)
#print gen.tokenizer.index_word.find("http")
# example that prints all the tweets of the first batch

#X = gen
#X = {'text'     : X.text.to_list()}
#X_text = X['text']
#X_sequences = gen.tokenizer.sequences_to_texts(X_text)

#data = pd.read_csv(Path('nlp-getting-started', 'train' + '.csv'))
#X = {'keyword'  : data.keyword.to_list(),
#     'location' : data.location.to_list(),
#     'text'     : data.text.to_list()}

#X_text = X['text']
# Convert list of strings into list of lists of integers
#X_sequences = tokenizer.texts_to_sequences(X_text)
#X_pad = sequence.pad_sequences(X_sequences, maxlen=max_length)

#y = np.array(data.target.to_list())

check_str = 'http'

all_words = []

for k in range(0,len(gen.all_text)):
    sentence = gen.all_text[k].split()
    all_words.extend(sentence)

len_all_words = len(all_words)

num_http = 0
for k in range(0,len_all_words):
    if check_str in all_words[k]:
        num_http = num_http+1
print(num_http)
url_div_twords_all = num_http/len_all_words
print(url_div_twords_all)

num_url_limited = 0
for k in range(1,gen.indexes.shape[0]):
    batch = gen[k]  # first of len(gen) batches
    X = batch[0]  # 0 -> keywords/location/text, 1 -> target
    text_first_batch = gen.tokenizer.sequences_to_texts(X)[0]
    text_first_batch.split()
    if check_str in text_first_batch:
        num_url_limited = num_url_limited+1

url_div_twords_limited = num_url_limited/vocabulary_size
print(url_div_twords_limited)