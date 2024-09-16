import torch
import numpy as np

def train(dataloader, my_model, criterion, my_optimizer):
    my_model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_sum = 0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X  = X.unsqueeze(1) # Adding channel dimension
        # Compute prediction and loss
        pred = my_model(X)
        # if batch == 0:
        #     print(y)
        #     print(pred)
        loss = criterion(pred, y)
        loss_sum += loss
        correct += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()

        # Backpropagation
        my_optimizer.zero_grad()
        loss.backward()
        my_optimizer.step()
        
    print(f'Accuracy: {correct / size}')
    print(f'Epoch loss: {loss_sum / num_batches}')


def test(dataloader, my_model, loss_fn):
    y_true = []
    y_pred =[]
    
    my_model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0., 0.

    for idx, (X, y) in enumerate(dataloader):
        X = X.unsqueeze(1)
        
        pred = my_model(X)
        
        test_loss += loss_fn(pred, y).item()
        
        correct += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
        
        y_true.append(y.cpu().detach().numpy())
        y_pred.append(torch.argmax(pred, dim=1).cpu().detach().numpy())

    test_loss /= num_batches
    print(f'Training loss: {test_loss}')
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return np.hstack(y_true), np.hstack(y_pred), correct


def test2(dataloader, my_model, criterion):
    my_model.eval()
    size = len(dataloader.dataset)
    loss_sum = 0.
    correct = 0.
    for batch, (X, y) in enumerate(dataloader):
        X = X.unsqueeze(1)
        # Compute prediction and loss
        pred = my_model(X)
        loss = criterion(pred, y)
        loss_sum += loss.item()
        correct += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
        
    print(f'Test Accuracy: {correct / size}')
    return correct / size
