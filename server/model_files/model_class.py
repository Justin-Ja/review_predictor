import torch
from torch import nn

# A LSTM model used to predict review scores through regression
# Regression improves model accuracy by allowing decimal (float) guesses to be made by the model (compared to only int), at cost of training speed
class LSTM_regr(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.2, hidden_layers=1) :
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.hidden_layers = hidden_layers
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Forward pass. length_x could be used to create padding management, but that would be a future worry/update
    def forward(self, x, length_x):
        # x = self.embeddings(x)
        # x = self.dropout(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(self.dropout(self.embeddings(x))) # Avoid re-assigning x for slightly better performance
        return self.linear(hidden_state[-1])
    
    # Mainly for debugging purposes
    def print_init_args(self):
        print(f'vocab_size: {self.vocab_size}')
        print(f'embedding_dim: {self.embedding_dim}')
        print(f'hidden_dim: {self.hidden_dim}')
        print(f'dropout: {self.dropout_rate}')
        print(f'hidden_layers: {self.hidden_layers}')
    
    # Used to save the init_args used for a model to a file to load said model later on
    # Returns a dictonary of all init values used for an instance
    def get_init_args(self):
        return {
            'vocab_size': str(self.vocab_size),
            'embedding_dim': str(self.embedding_dim),
            'hidden_dim': str(self.hidden_dim),
            'dropout_rate': str(self.dropout_rate),
            'hidden_layers': str(self.hidden_layers)
        }