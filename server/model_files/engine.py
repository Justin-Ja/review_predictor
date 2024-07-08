import torch
import numpy as np
from torch import nn
from sklearn.metrics import mean_squared_error

# Runs through the whole loop of training and testing a model
def train_model(model: torch.nn.Module, train_dl, test_dl, epochs=10, lr=0.001, device: torch.device = 'cpu'):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
   
    for epoch in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0

        for x, y, l in train_dl:

            x, y = x.to(device), y.to(device)

            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        
        val_loss, val_acc, val_rmse = __test_model(model, test_dl)
        if epoch % 5 == 1:
            print(f"Epoch: {epoch} |"
                  f"val loss {val_loss:.3f} |"
                  f"val loss {sum_loss/total:.3f} |"
                  f"val accuracy {val_acc:.3f} |"
                  f" and val rmse {val_rmse:.4f}"
                )

# Runs the testing portion of model training, and determines loss/accuracy/RMSE aka how well the model's doing
def __test_model(model, test_dl, device: torch.device = 'cpu'):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0 # Root mean squared error

    for x,y,l in test_dl:
        x, y = x.to(device), y.to(device)

        x = x.long()
        y = y.long()
        y_hat = model(x,l)

        loss = nn.functional.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    
    return sum_loss/total, correct/total, sum_rmse/total       
