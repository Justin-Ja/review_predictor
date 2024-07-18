from model_class import LSTM_regr
from setup_data import create_dataLoaders
from engine import train_model
from utils import determine_device, save_model
from timeit import default_timer as timer
import argparse
from typing import Final

#TODO: Import argparse and set that up after testing
parser = argparse.ArgumentParser(prog="\nA ML script to prepare data and train a LSTM language model, with potential to save it\n")

parser.add_argument('-e', '--epochs', help='The number of loops should the model train/test for', 
                    default=25, type=int)
parser.add_argument('-u', '--units', help='Number of neurons to use per ML layer. Higher value helps the model be more accurate, at the cost of training time\n', 
                    default=64, type=int)
parser.add_argument('-d', '--dropout', help='Dropout percentage, a value between 0 and 1. Used in training to drop a certain portion of data in a batch to help prevent overfitting\n', 
                    default=0.25, type=float)
parser.add_argument('-s', '--subset', help='The percentage (value between 0 to 1) of the subset that will be used for training instead of the full set. Good for quick training purposes.\n', 
                    default=0.001, type=float)
parser.add_argument('-r', '--random', help='Boolean to determine if subsets created should be randomized. True is randomized subset elements, false is constant subset elements\n', 
                    default=False, type=bool)

args = parser.parse_args()

EPOCHS: Final[int] = args.epochs
HIDDEN_UNITS: Final[int] = args.units
DROPOUT: Final[float] = args.dropout
SUBSET_PERCENTAGE: Final[float] = args.subset
RANDOMIZE: Final[bool] = args.random

if(DROPOUT < 0 or DROPOUT > 1):
    print("Dropout is not a percentage value (0 < x < 1). \nAborting...")
    exit()

device = determine_device()
train_file_path = 'data/train-00000-of-00001.parquet'

dataLoaders_and_vocab = create_dataLoaders(train_file_path, 5000, DROPOUT, 0.001, RANDOMIZE)

train_dl = dataLoaders_and_vocab[0]
test_dl = dataLoaders_and_vocab[1]
vocab_size = dataLoaders_and_vocab[2]

#print(vocab_size)

model_LSTM = LSTM_regr(vocab_size, HIDDEN_UNITS, HIDDEN_UNITS, dropout = DROPOUT, hidden_layers = 1)

print("Starting training, this will take several minutes...\n")

start = timer()
#TODO: implement some lr scheduler (? i think thats the name) to not have this call twice (slightly confusing output with multiple epoch 1's)
# if implemented, add LR in argparse
train_model(model_LSTM, train_dl, test_dl, epochs=EPOCHS, lr=0.01, device=device)
train_model(model_LSTM, train_dl, test_dl, epochs=EPOCHS, lr=0.001, device=device) 
end = timer()

print(f"Total training time: {end - start}")

while True:
    isSave = input("\nSave this model?\n(YOU CANNOT RECOVER THE CURRENT STATE OF THE MODEL IF YOU DO NOT SAVE)\nY/N: ").strip().lower()
    
    if isSave in ("y", "n"):
        break
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")

if isSave.lower() == "y":
    save_model(model_LSTM)
else:
    print("Exiting without saving model...\n")