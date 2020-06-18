from DataLoader import LoadTweets, TokenizeTweets

from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
from pathlib import Path
import numpy as np

#Import the required models
import models.LSTM_model as LSTM_model
import models.Bidirectional_LSTM_model as Bidirectional_LSTM_model
import models.LSTM_CNN_model as LSTM_CNN_model
import models.LSTM_CNN_model as LSTM_CNN_modelV2
import models.LSTM_CNN_model as LSTM_CNN_modelV3
import models.Bidirectional_LSTM_CNN_modelV2 as Bidirectional_LSTM_CNN_modelV2



import models.Bidirectional_LSTM_CNN_model as Bidirectional_LSTM_CNN_model

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp




if __name__ == '__main__' :
    save_model = True

    """Create model"""
    vocabulary_size = 1000
    max_length = 100

    embedding_vecor_length = 100
    tokenizer = TokenizeTweets(vocabulary_size=vocabulary_size)
    
    train_gen = LoadTweets(tokenizer, split='train',
                           batch_size = 256, shuffle=False,
                           vocabulary_size=vocabulary_size, max_length=max_length)
    val_gen = LoadTweets(tokenizer, split='val',
                         batch_size = 256, shuffle=False,
                         vocabulary_size=vocabulary_size, max_length=max_length)

    #Initialze the hyperparams you want to train with.
    HP_MODEL_NAME = hp.HParam('model_name', hp.Discrete(['Bidirectional_LSTM_CNN_model']))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.05, 0.1, 0.2, 0.5]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['RMSprop', 'adam']))
    HP_LSTM_OUT = hp.HParam('lstm_output_size', hp.Discrete([10, 20, 50, 100]))

    for model_name in HP_MODEL_NAME.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                for lstm_out in HP_LSTM_OUT.domain.values:
                    hparams = {
                        HP_MODEL_NAME: model_name,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                        HP_LSTM_OUT: lstm_out
                    }

                    """## Train model"""
                    print({h.name: hparams[h] for h in hparams})

                    # Make a folder to save model weights, and

                    run = datetime.now().strftime("%Y%m%d-%H%M%S")
                    log_folder = "logs/" + model_name + '/'
                    logdir = Path(log_folder + run)

                    """Callbacks"""
                    callbacks = []
                    # check if the directory to save the model in exists
                    os.makedirs(os.path.dirname(logdir), exist_ok=True)
                    callbacks.append(TensorBoard(log_dir=logdir))

                    #Create the hyperparameter callback
                    logdir = log_folder + run
                    callbacks.append(hp.KerasCallback(logdir, hparams))
                    model = globals()[hparams[HP_MODEL_NAME]].create_model(vocabulary_size, embedding_vecor_length, max_length, lstm_out, optimizer= optimizer, dropout = dropout_rate)
                    model.fit(train_gen, validation_data=val_gen, epochs=50,
                              callbacks= callbacks)

                    if save_model == True:
                        model.save(Path(logdir + '/model.h5'))

