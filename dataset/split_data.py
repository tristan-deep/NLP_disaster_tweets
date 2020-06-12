import pandas as pd
from pathlib import Path
import numpy as np

def split_dataset(data_path = 'dataset/complete_labelled_dataset.csv'):
    #Load dataset
    complete_dataset = pd.read_csv(Path(data_path))
    #Check how many entries the dataset has.
    no_entries = np.shape(complete_dataset)[0]
    #Randomly permuted index matrix
    indices = np.random.permutation(no_entries)
    num_train_samples  = int(0.8*no_entries)
    train_idx, val_idx = indices[:num_train_samples], indices[num_train_samples:]
    train_split, val_split = complete_dataset.loc[train_idx], complete_dataset.loc[val_idx]
    return train_split, val_split


train_set, val_set = split_dataset(data_path = 'dataset/complete_labelled_dataset.csv')


#Do not save the sets now, as we already use them.

#train_set.to_csv('dataset/train_set.csv')
#val_set.to_csv('dataset/val_set.csv')