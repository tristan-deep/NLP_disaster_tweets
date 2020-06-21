from DataLoader import LoadTweets, TokenizeTweets
from models.LSTM_CNN_deep_model import create_model

from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
from pathlib import Path
import numpy as np



if __name__ == '__main__' :
    save_model = True
    model_name = 'LSTM_CNN_Deep_model.h5'
    
    #Make a folder to save model weights, and
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = Path("logs/"+run)
    
    #check if the directory to save the model in exists
    os.makedirs(os.path.dirname(logdir), exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=logdir)
    
    
    """Create model"""
    vocabulary_size = 10000
    max_length = 100

    embedding_vecor_length = 32
    lstm_out = 20
    tokenizer = TokenizeTweets(vocabulary_size=vocabulary_size)
    
    train_gen = LoadTweets(tokenizer, split='train',
                           batch_size = 32, shuffle=False,
                           vocabulary_size=vocabulary_size, max_length=max_length)
    val_gen = LoadTweets(tokenizer, split='val',
                         batch_size = 32, shuffle=False,
                         vocabulary_size=vocabulary_size, max_length=max_length)
    
    model = create_model(vocabulary_size, embedding_vecor_length, max_length, dropout=0.5, lstm_out=20, conv_num_filters = 16)
    
    
    """## Train model"""
    model.fit(train_gen, validation_data = val_gen, epochs=10, callbacks = [tensorboard_callback])
    
    if save_model == True :
        PATH = os.path.join('weights', model_name)
        model.save(PATH)