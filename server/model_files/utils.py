import torch
import os
from pathlib import Path
from .model_class import LSTM_regr

def save_model(model: LSTM_regr) -> None:    
    # Create models directory 
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Create model save path 
    while True:
        name = input("Name the model: ")
        if name.endswith(".pth") or name.endswith(".pt"):
            break
        else:
            print("Invalid name. Model name should end with '.pt' or '.pth'.")

    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Save the model state dict 
    print(f"\nSaving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

    # Save the parameters that were used to create the model 
    # (loading a model requires an instance initialized with the same values as the saved model)
    init_dict = model.get_init_args()
    file_name = os.path.join(MODEL_PATH, os.path.splitext(name)[0] + ".dat")
    
    with open(file_name, 'w') as f:
        for _, value in init_dict.items():
            f.write(f"{value}\n")

# Loads a LSTM_regression model from a saved file. Function is used at different directory levels, so model_path is passed in
def load_model_LSTM_regr(model_name: str, model_path: str) -> LSTM_regr:
    
    tempName = os.path.splitext(model_name)[0] + ".dat"
    MODEL_INFO_PATH = model_path / tempName
    MODEL_SAVE_PATH = model_path / model_name

    info_numbers = []

    with open(MODEL_INFO_PATH, 'r') as file:
        for line in file:
            number = line.strip()
            if number:
                info_numbers.append(float(number) if '.' in number else int(number))

    vocab_size = info_numbers[0]
    em_dim = info_numbers[1]
    hidden_dim = info_numbers[2]
    dropout =  info_numbers[3]
    hidden_layers = info_numbers[4]

    loaded_model_LSTM_regression = LSTM_regr(vocab_size, em_dim, hidden_dim, dropout, hidden_layers)
    loaded_model_LSTM_regression.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    return loaded_model_LSTM_regression

def determine_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"