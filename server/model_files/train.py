from model_class import LSTM_regr
from setup_data import create_dataLoaders
from engine import train_model
from utils import determine_device, save_model
from timeit import default_timer as timer

#TODO: Import argparse and set that up after testing

device = determine_device()
train_file_path = 'data/train-00000-of-00001.parquet'

dataLoaders_and_vocab = create_dataLoaders(train_file_path, 5000, 0.25, 0.001, True)

train_dl = dataLoaders_and_vocab[0]
test_dl = dataLoaders_and_vocab[1]
vocab_size = dataLoaders_and_vocab[2]

print(train_dl)
print(test_dl)
print(vocab_size)

model_LSTM = LSTM_regr(vocab_size, 80, 80, dropout = 0.25, hidden_layers = 1)

print("Starting training, this will take several minutes...\n")

start = timer()
#TODO: implement some lr scheduler (? i think thats the name) to not have this call twice (slightly confusing output with multiple epoch 1's)
train_model(model_LSTM, train_dl, test_dl, epochs=5, lr=0.01, device=device)
train_model(model_LSTM, train_dl, test_dl, epochs=5, lr=0.001, device=device) 
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