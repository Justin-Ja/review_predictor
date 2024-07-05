import torch
import os
from pathlib import Path
from model_class import LSTM_regr

def save_model(model: LSTM_regr) -> None:
    name = input("Name the model: ")
    
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
    file_name = os.path.splitext(name)[0] + ".txt"
    
    with open(file_name, 'w') as f:
        for key, value in init_dict.items():
            if value.isdigit():
                f.write(f"{value}\n")

# Loads a LSTM_regression model from a saved file
def load_model_LSTM_regr(name: str, 
                         vocab_size: int, 
                         embedding_dim: int, 
                         hidden_dim: int, 
                         dropout = 0.2, 
                         hidden_layers = 1) -> LSTM_regr:
    
    MODEL_PATH = Path("models")
    MODEL_SAVE_PATH = MODEL_PATH / name

    #TODO: Determine if these init_args should be read from file or expected from user to pass in
    loaded_model_LSTM_regression = LSTM_regr(vocab_size, embedding_dim, hidden_dim, dropout, hidden_layers)
    loaded_model_LSTM_regression.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    return loaded_model_LSTM_regression