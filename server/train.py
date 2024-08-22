# from model_class import LSTM_regr
# from setup_data import create_dataLoaders
# from engine import train_model
# from utils import determine_device, save_model
from model_files import model_class, setup_data, engine, utils
from timeit import default_timer as timer
import argparse
from typing import Final
from CONSTANTS import TRAIN_FILE_PATH

#TODO: Import argparse and set that up after testing
parser = argparse.ArgumentParser(prog="\nA ML script to prepare data and train a LSTM language model, with potential to save it\n")

parser.add_argument('-e', '--epochs', help='The number of loops should the model train and test for', 
                    default=25, type=int)
parser.add_argument('-l', '--learningRate', help='The step size the model will take after each epoch. This will be the initial value, as there is a scheduler that will reduce the LR by 10x halfway through.', 
                    default=0.01, type=float)
parser.add_argument('-u', '--units', help='Number of neurons to use per ML layer. Higher value helps the model be more accurate, at the cost of training time\n', 
                    default=64, type=int)
parser.add_argument('-d', '--dropout', help='Dropout percentage, a value between 0 and 1. Used in training to drop a certain portion of data in a batch to help prevent overfitting\n', 
                    default=0.25, type=float)
parser.add_argument('-s', '--subset', help='The percentage (value between 0 to 1) of the subset that will be used for training instead of the full set. Good for quick training purposes.\n', 
                    default=0.001, type=float)
parser.add_argument('-t', '--testSplit', help='The percentage (value between 0 to 1) of the input data that will be split into test data. A value of 0.2 means 20 percent of the data will be test data\n', 
                    default=0.2, type=float)
parser.add_argument('-r', '--random', help='Boolean to determine if subsets created should be randomized. True is randomized subset elements, false is constant subset elements\n', 
                    default=False, type=bool)

args = parser.parse_args()

EPOCHS: Final[int] = args.epochs
LEARNING_RATE: Final[float] = args.learningRate
HIDDEN_UNITS: Final[int] = args.units
DROPOUT: Final[float] = args.dropout
SUBSET_PERCENTAGE: Final[float] = args.subset
TEST_SPLIT: Final[float] = args.testSplit
RANDOMIZE: Final[bool] = args.random

if(DROPOUT < 0 or DROPOUT > 1):
    print("Dropout is not a percentage value (0 < x < 1). \nAborting...")
    exit()

device = utils.determine_device()

dataLoaders_and_vocab = setup_data.create_dataLoaders(TRAIN_FILE_PATH, 
                                           batch_size=5000, 
                                           test_split_percentage=0.25, 
                                           subset_percentage=SUBSET_PERCENTAGE, 
                                           randomize_subset=RANDOMIZE)

train_dl = dataLoaders_and_vocab[0]
test_dl = dataLoaders_and_vocab[1]
vocab_size = dataLoaders_and_vocab[2]

#print(vocab_size)

model_LSTM = model_class.LSTM_regr(vocab_size, HIDDEN_UNITS, HIDDEN_UNITS, dropout = DROPOUT, hidden_layers = 1)

print("Starting training, this will take several minutes...\n")

start = timer()
engine.train_model(model_LSTM, train_dl, test_dl, epochs=EPOCHS, lr=LEARNING_RATE, device=device)
end = timer()

print(f"Total training time: {end - start}")
print("\n-------------------------")

while True:
    print("\nSave this model?\n(YOU CANNOT RECOVER THE CURRENT STATE OF THE MODEL IF YOU DO NOT SAVE)\n")
    isSave = input("Type \'save\' to save the model or \'exit\' to exit without saving:\n").strip().lower()
    
    if isSave in ("save", "exit"):
        break

if isSave == "save":
    utils.save_model(model_LSTM)
else:
    print("Exiting without saving model...\n")