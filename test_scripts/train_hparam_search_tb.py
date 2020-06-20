from DataLoader import LoadTweets, TokenizeTweets
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import os
from pathlib import Path

#Import the required models


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
    HP_MODEL_NAME = hp.HParam('model_name', hp.Discrete(['CNN_model', 'LSTM_CNN_deep_model']))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.5]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
    HP_LSTM_OUT = hp.HParam('lstm_output_size', hp.Discrete([20]))
    HP_CONV_MODULES = hp.HParam('conv_num_modules', hp.Discrete([1, 2, 3]))
    HP_CONV_FILTERS = hp.HParam('conv_num_filters', hp.Discrete([16, 32, 64]))
    HP_CONV_FILTER_SIZE = hp.HParam('conv_filter_size', hp.Discrete([1, 2, 5]))

    #Loop through the hyperparameters
    for model_name in HP_MODEL_NAME.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                for lstm_out in HP_LSTM_OUT.domain.values:
                    for conv_num_modules in HP_CONV_MODULES.domain.values:
                        for conv_num_filters in HP_CONV_FILTERS.domain.values:
                            for conv_filter_size in HP_CONV_FILTER_SIZE.domain.values:
                                hparams = {
                                    HP_MODEL_NAME: model_name,
                                    HP_DROPOUT: dropout_rate,
                                    HP_OPTIMIZER: optimizer,
                                    HP_LSTM_OUT: lstm_out,
                                    HP_CONV_MODULES: conv_num_modules,
                                    HP_CONV_FILTERS: conv_num_filters,
                                    HP_CONV_FILTER_SIZE: conv_filter_size
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

                                model = globals()[hparams[HP_MODEL_NAME]].create_model(max_words=vocabulary_size, embedding_vecor_length=embedding_vecor_length, max_length=max_length, dropout=dropout_rate,
                                             lstm_out = lstm_out, conv_num_modules=conv_num_modules, conv_num_filters=conv_num_filters, conv_filter_size=conv_filter_size,
                                             optimizer='adam', print_summary=True, )
                                model.fit(train_gen, validation_data=val_gen, epochs=100,
                                          callbacks= callbacks)

                                if save_model == True:
                                    model.save(Path(logdir + '/model.h5'))

