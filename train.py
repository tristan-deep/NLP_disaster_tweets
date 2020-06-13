from DataLoader import LoadTweets, TokenizeTweets
from models.LSTM_model import create_model

from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
from pathlib import Path
import numpy as np



if __name__ == '__main__' :

    #Make a folder to save model weights, and
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = Path("logs/"+run)
    
    #check if the directory to save the model in exists
    os.makedirs(os.path.dirname(logdir), exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=logdir)
    
    
    """Create model"""
    max_length = 500
    max_words = 10000
    embedding_vecor_length = 32
    lstm_out = 100
    tokenizer = TokenizeTweets(max_words=max_words)
    
    train_gen = LoadTweets(tokenizer,split='train', batch_size = 32, shuffle=False, max_words = max_words, max_length=max_length)
    val_gen = LoadTweets(tokenizer, split='val', batch_size = 32, shuffle=False, max_words = max_words, max_length=max_length)
    model = create_model(max_words, embedding_vecor_length, max_length, lstm_out)
    
    
    """## Train model"""
    model.fit(train_gen, validation_data = val_gen, epochs=15, callbacks = [tensorboard_callback])