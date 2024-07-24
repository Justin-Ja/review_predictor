from pathlib import Path
import os
import torch
from model_class import LSTM_regr
import pandas as pd
from setup_data import create_dataLoaders
from utils import determine_device, load_model_LSTM_regr
import numpy as np
import spacy 
from collections import Counter
from setup_data import encode_sentence



#Setup data reading and model loading
torch.manual_seed(42)
loaded_model_LSTM_regression = load_model_LSTM_regr()

# Load the saved model
# loaded_model_LSTM_regression = LSTM_regr(vocab_size, info_numbers[1], info_numbers[2], info_numbers[3], info_numbers[4])
# loaded_model_LSTM_regression.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

test_file_path = 'data/test-00000-of-00001.parquet'

#TODO: Argparse the subset % and randomization. 
# We set batch size to 1 since batch size is needed in training but we only want to view 10 or less reviews to see how the model is doing
dataLoaders_and_vocab = create_dataLoaders(test_file_path, 1, 0.25, 0.00025, True)
train_dl = dataLoaders_and_vocab[0]
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
        pass
        #del counts[word]
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

        # pred = torch.max(y_hat, 1)[1]
        # print(pred)
        # correct += (pred == y).float().sum()
        # sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
        
        original_value = test_subset.iloc[i, 0]
        print(f"Original Y/label/review score: {original_value + 1}")
        print(f"Predicted values: {y_hat[0] + 1}")
        print(f"Review text: {test_subset.iloc[i, 1]}")
        print()

        i = i + 1



# # Take user input
# userStr = input("Enter a review: ")

# # Preprocess the user input
# encoded_input, input_length = encode_sentence(userStr, vocab2index)

# # Convert to tensor
# input_tensor = torch.from_numpy(encoded_input).unsqueeze(0).long()

# # Evaluate the model
# with torch.inference_mode():
#     loaded_model_LSTM_regression.eval()
#     y_hat = loaded_model_LSTM_regression(input_tensor, torch.tensor([input_length]))
#     # predicted_score = y_hat.item() + 1  # Convert back to 1-5 scale
#     print(f"Predicted review score (1-5 scale): {y_hat + 1}")