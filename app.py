import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import preprocess
from neuralnetwork import NeuralNetwork
from dataset import NFLDataset
from learn import train, test
from device import device

Xtrn, Xtst, Ytrn, Ytst = preprocess.load()

print()
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

training_data = NFLDataset(Xtrn, Ytrn)
test_data = NFLDataset(Xtst, Ytst)
train_dataloader = DataLoader(training_data)
test_dataloader = DataLoader(test_data)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)
# initial_weights = {}
# for name, param in model.named_parameters():
#     initial_weights[name] = param.clone().detach().cpu().numpy()

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

with torch.no_grad():
    for i in range(10):
        x, y = test_data[i][0], test_data[i][1]
        x = x.to(device)
        pred = model(x)
        print(f'Predicted: "{pred}", Actual: "{y}"')