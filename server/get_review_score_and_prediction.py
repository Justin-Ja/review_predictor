import torch
import pandas as pd
from model_files import setup_data, utils
from pathlib import Path
import numpy as np
import spacy 
from collections import Counter

MODEL_PATH = Path('model_files/models')
TEST_FILE_PATH = 'model_files/data/test-00000-of-00001.parquet'

# Returns an object containing a review's text, label (score) and the predicted score
# Assumes that model_name ends with .pth or .pt, should be a constant value that is passed in.
def get_review_score_pred(model_name: str):

    loaded_model_LSTM_regression = utils.load_model_LSTM_regr(model_name, MODEL_PATH)

    subset = get_encoded_review(TEST_FILE_PATH)

    print(subset)
    print("lmao")
    review_text = str(subset['text'].iloc[0])
    score = int(subset['label'].iloc[0]) + 1   # Convert back to 1-5 scale from 0-4 so add +1
    input_length = int(subset['review_length'].iloc[0])


    print(subset['encoded'].iloc[0])

    # Convert to tensor
    input_tensor = torch.from_numpy(subset['encoded'].iloc[0]).unsqueeze(0).long()
    
    with torch.inference_mode():
        loaded_model_LSTM_regression.eval()
        pred_score = loaded_model_LSTM_regression(input_tensor, torch.tensor([input_length]))
        print(f"Predicted review score (1-5 scale): {pred_score + 1}") 
        print(pred_score.item())
        print(type(pred_score.item()))
        pred_score = (pred_score.item() + 1) # Convert back to 1-5 scale
    
    print(review_text)
    print(score)
    print(pred_score)
    return {
        'text': review_text,
        'score': score,
        'pred_score': pred_score, 
    }
        

# This function gets a review and gets the encoded version of said review for ML evaluation
# See setup_data.py for more detail on what is happening
# We want only one review, as every time we want a new review/round, we can just hit the related endpoint and run this again
def get_encoded_review(file_path):
    test_df = pd.read_parquet(file_path)

    subset = test_df.sample(frac=(5/float(len(test_df))))
    tok = spacy.load('en_core_web_sm')

    subset['review_length'] = subset['text'].apply(lambda x: len(x.split()))
    counts = Counter()

    for _, row in subset.iterrows():
            counts.update(setup_data.tokenize(tok, row['text']))

    # TODO: If we're only passing in one review, do we want to remove uncommon words? we cut a review with 110 words to 35, might mess with results
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    subset['encoded'] = subset['text'].apply(lambda x: np.array(setup_data.encode_sentence(x, vocab2index, tok)[0]))

    return subset


if __name__ == "__main__":
    print("Testing evaluating one review...")
    get_review_score_pred("dummy.pth")