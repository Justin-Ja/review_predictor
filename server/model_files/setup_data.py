import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re #Regex lib
import spacy 
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# gonna need a proper way of downloading the data. not good idea just to upload the parquet file directly to github

train_file_path = 'FIXTHISLATEREEREE'
# Assume for now there is data in the data folder

def download_dataset():

    return train_file_path

# if subset = 0 OR 1 then no subset at all
def create_dataLoaders(subset_percentage: float = 0.0):
    torch.manual_seed(42) # move this or REmove this bc idk

    train_df = pd.read_parquet(train_file_path)

