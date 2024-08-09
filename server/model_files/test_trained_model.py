import torch
import pandas as pd
from setup_data import create_dataLoaders
from utils import load_model_LSTM_regr
import numpy as np
import spacy 
from collections import Counter
from setup_data import encode_sentence

# This file is not used directly in the server/frontend component of the app
# This is for final testing of trained models to test the model more closely to what happens in the app
# See get_review_score_and_prediction for 

#Setup data reading and model loading
torch.manual_seed(42)
MODEL_PATH = 'models'

while True:
    model_name = input("Give name of model to load: ")
    if model_name.endswith(".pth") or model_name.endswith(".pt"):
        break
    else:
        print("Invalid name. Model name should end with '.pt' or '.pth'.")

loaded_model_LSTM_regression = load_model_LSTM_regr(model_name, MODEL_PATH)

test_file_path = 'data/test-00000-of-00001.parquet'

#TODO: Argparse the subset % and randomization. 
# We set batch size to 1 since batch size is needed in training but we only want to view 10 or less reviews to see how the model is doing
dataLoaders_and_vocab = create_dataLoaders(test_file_path, 1, 0.25, 0.00025, True)
test_dl = dataLoaders_and_vocab[1]



# Need to create a test_subset pd to be able to access/print the review and label that was evaluated
# This does require a little bit of repeated processing.
test_df = pd.read_parquet(test_file_path)
test_subset = test_df.sample(frac=0.00025)


test_subset['review_length'] = test_subset['text'].apply(lambda x: len(x.split()))
print(test_subset.head())
counts = Counter()

print("num_words before:",len(counts.keys()))
for word in list(counts):
    if counts[word] < 2:
        del counts[word]
print("num_words after:",len(counts.keys()))

vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)

tok = spacy.load('en_core_web_sm')
test_subset['encoded'] = test_subset['text'].apply(lambda x: np.array(encode_sentence(x, vocab2index, tok)[0]))


# This section sets the model to evaluate, and prints out the review, its star score (label from 1-5) and the estimated score by the model (1-5 stars)
with torch.inference_mode():
    loaded_model_LSTM_regression.eval()
    correct = 0
    i = 0
    sum_loss = 0
    sum_rmse = 0 # Root mean squared error
    for x,y,l in test_dl:
        x = x.long()
        y = y.long()
        y_hat = loaded_model_LSTM_regression(x,l)
        
        original_value = test_subset.iloc[i, 0]
        print(f"Original Y/label/review score: {original_value + 1}")
        print(f"Predicted values: {y_hat[0] + 1}")
        print(f"Review text: {test_subset.iloc[i, 1]}")
        print()

        i = i + 1