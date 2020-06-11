from DataLoader import LoadTweets, tokenizer_tweets
from models.LSTM_model import create_model
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np


logdir = '\logs'
run = datetime.now().strftime("%Y%m%d-%H%M%S")
#check if the directory to save the model in exists
os.makedirs(os.path.dirname(logdir), exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=logdir)

"""Create model"""
max_words = 10000
max_length = 500
embedding_vecor_length = 32
lstm_out = 100

tokenizer = tokenizer_tweets(max_words)
gen = LoadTweets(tokenizer, split='train', batch_size = 32, shuffle=False, max_length=max_length)
model = create_model(max_words, embedding_vecor_length, max_length, lstm_out)

"""## Train model"""
model.fit(gen, epochs=1, callbacks = [tensorboard_callback])
