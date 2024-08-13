import torch
import numpy as np
from torch import nn
from sklearn.metrics import mean_squared_error

# Runs through the whole loop of training and testing a model
def train_model(model: torch.nn.Module, train_dl, test_dl, epochs=10, lr=0.001, device: torch.device = 'cpu'):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
   
    for epoch in range(epochs):
        #Setup and set model to train
        model.train()
        sum_loss = 0.0
        total = 0

        #Batch training, x is review text, y is label, l is x length
        for x, y, l in train_dl:

            #Send info to GPU if avaliable, and convert to proper datatypes
            x, y = x.to(device), y.to(device)
            x = x.long()
            y = y.float()
        
            #Forward pass
            y_pred = model(x, l)
            
            #Loss calculation
            loss = nn.functional.mse_loss(y_pred, y.unsqueeze(-1))
            
            #Zero gradients, backpropagation on loss, update optimizer
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()

            #Calc total loss for the batch
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]

        test_loss = __test_model(model, test_dl)

        if epoch % 3 == 1:
            print(f"Epoch: {epoch} | Train loss: {sum_loss/total:.4f} | Test loss: {test_loss:.4f} |")

    # Once last print to see final state of model
    print(f"\nFinal results:" f" | Train loss {sum_loss/total:.4f} |" f"Test loss {test_loss:.4f} |")


# Runs the testing portion of model training, and determines MSE Loss - AKA how well the model's doing
def __test_model(model, test_dl, device: torch.device = 'cpu'):
    model.eval()
    total = 0
    sum_loss = 0.0

    for x, y, l in test_dl:
        x, y = x.to(device), y.to(device)

        x = x.long()
        y = y.float()

        y_hat = model(x, l)
        loss = np.sqrt(nn.functional.mse_loss(y_hat, y.unsqueeze(-1)).item())
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]

    return sum_loss/total
