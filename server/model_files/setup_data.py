import torch
import pandas as pd
import numpy as np
import re #Regex lib
import spacy 
import string
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# TODO: gonna need a proper way of downloading the data. not good idea just to upload the parquet file directly to github
# possibly clone it? test in a completely separate folder first. Worst case just upload the files to github, theyre 323 mb
train_file_path = 'data/train-00000-of-00001.parquet'
test_file_path = 'data/test-00000-of-00001.parquet'

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].astype(np.int32)), self.y[idx], self.X[idx][1]

# def download_dataset():

#     return train_file_path

# Creates train/test dataloaders and returns it along with  vocab_size in a Tuple (vocab_size needed to create instance of LSTM guest)
def create_dataLoaders(path: Path,
                    batch_size: int = 5000,
                    test_split_percentage: float = 0.25, 
                    subset_percentage: float = 0.0, 
                    randomize_subset: bool = False):
    
    torch.manual_seed(42)
    pd.options.display.max_columns = 6

    if (subset_percentage <= 0 and subset_percentage > 1) or (test_split_percentage <= 0 and test_split_percentage > 1):
        print("ERROR: Cannot create dataloaders: invalid percentage values (not between 0 and 1)")
        return None
    
    train_df = pd.read_parquet(path)

    if subset_percentage > 0 and subset_percentage < 1:
        if not randomize_subset:
            train_df = train_df.sample(frac=subset_percentage, random_state=1) #This sets the sample seed to 1, to have a consistent subset selection
        else:
            train_df = train_df.sample(frac=subset_percentage)

    tok = spacy.load('en_core_web_sm')
    counts = Counter()

    for _, row in train_df.iterrows():
        counts.update(tokenize(tok, row['text']))

    #deleting infrequent words
    print("num_words before:",len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    print("num_words after:",len(counts.keys()))

    #Initialize a vocab dictonary with padding as index 0 and UNKNOWN as index 1. Added words get their own indecies
    vocab_index_dict = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab_index_dict[word] = len(words)
        words.append(word)

    # Adding column to dataframe for encoded version of text
    train_df['encoded'] = train_df['text'].apply(lambda x: np.array(encode_sentence(x, vocab_index_dict, tok)[0]))

    print("Total count of each star rating (0-4):")
    print(Counter(train_df['label']))

    X = list(train_df['encoded'])
    y = list(train_df['label'])

    # Yes, im aware there is a test source file as well. I'm not at the point with training where the whole set is needed + this is easier for now
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.275)

    train_ds = ReviewsDataset(X_train, y_train)
    test_ds = ReviewsDataset(X_test, y_test)

    vocab_size = len(words)

    NUM_WORKERS = os.cpu_count() - 2 # Don't want to use all CPU cores on training (Trying not to obliterate my computer, ideally)
    if NUM_WORKERS < 1:
        NUM_WORKERS = 1
    
    # print(f"Num workers: {NUM_WORKERS}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=NUM_WORKERS)

    # Not an ideal return call, but it works :/
    return (train_dl, test_dl, vocab_size)


# This function removes unwanted characters and then turns each word into its own token (separating words out from the sentence)
def tokenize (tok, text):
    #Remove non-ascii chars
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    #Sets up regex to remove punctuation and numbers and some whitespace chars
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    
    nopunct = regex.sub(" ", text.lower())

    return [token.text for token in tok.tokenizer(nopunct)]


# Essentially we are turning our english sentences into tokens, then tokens into numbers through a vocab dictionary
# N should be a little higher than the average length of input (review text length in this case)
def encode_sentence(text, vocab_dict, tok, N=140):
    tokenized = tokenize(tok, text)
    encoded = np.zeros(N, dtype=int)

    enc1 = np.array([vocab_dict.get(word, vocab_dict["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    
    return encoded, length

