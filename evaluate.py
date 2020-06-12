"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : evaluate.py
                    
==============================================================================
"""
from DataLoader import LoadTweets, tokenizer_tweets

import tensorflow.keras as keras
from pathlib import Path
import pandas as pd
import numpy as np

def import_model(file) :
    model = keras.models.load_model(Path('weights', model_name))

    return model
    

def create_submission(file, predictions, save_to_file=True) :
    '''
    Parameters
    ----------
    file : string
        filename to export csv to
    predictions : numpy array
        1D numpy array with predictions
    save_to_file : Bool, optional
        whether to save to file or not. The default is =True.

    Returns
    -------
    export : panda dataframe
        predictions along with ID in format

    '''
    
    test_IDs = pd.read_csv(Path('dataset', 'test.csv'), usecols=['id']).id
    
    export = pd.DataFrame({'id': test_IDs,
                           'target': predictions}, index=np.arange(test_set_size))
    
    if save_to_file == True :
        export.to_csv(Path('dataset/submissions',file), index=False)
    
        print('Saved csv file to: {}'.format(file))
    
    return export

if __name__ == '__main__' :
    
    model_name = 'lstm_embedding-32_out-100_epochs-5.h5'
    
    model = import_model(file=model_name)
    
    # get data from test set
    test_set_size = 3263 # amount of tweets in testset
    max_length=500
    max_words=10000
    
    tokenizer = tokenizer_tweets(max_words) # I would prefer this would be just called inside the DataLoader
    gen = LoadTweets(tokenizer, split='test', batch_size = test_set_size, shuffle=False, max_length=max_length)
    
    predictions_raw = model.predict(gen)
    
    predictions = np.squeeze((predictions_raw > 0.5).astype(int))
    
    create_submission(file='submission_test1.csv', predictions=predictions)