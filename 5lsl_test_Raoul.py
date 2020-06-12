from DataLoader import LoadTweets, tokenizer_tweets
from models.LSTM_model import create_model
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np


#Make a folder to save model weights, and
logdir = "logs"
run = datetime.now().strftime("%Y%m%d-%H%M%S")

#check if the directory to save the model in exists
os.makedirs(os.path.dirname(logdir+'/'), exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=logdir+"\\"+run)


"""Create model"""
max_length = 500
max_words = 10000
embedding_vecor_length = 32
lstm_out = 100
gen = LoadTweets(split='train', batch_size = 32, shuffle=False, max_words = max_words, max_length=max_length)
model = create_model(max_words, embedding_vecor_length, max_length, lstm_out)



"""## Train model"""
model.fit(gen, epochs=10, callbacks = [tensorboard_callback])
