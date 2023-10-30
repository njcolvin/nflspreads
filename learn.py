from torch import no_grad
from constants import device

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    current, batch = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current += len(X)
        if current % 640 == 0:
            print(f"Batch [{current:>5d}/{size:>5d}] - Loss: {loss.item():>7f}")
    
    average_loss = total_loss / (batch + 1)
    print(f"Average Loss: {average_loss:.6f}")

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            # Compute loss and MAE for score_home
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:.6f} \n")