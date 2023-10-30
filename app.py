import torch
from torch import nn
from torch.utils.data import DataLoader
import preprocess
from neuralnetwork import NeuralNetwork
from dataset import NFLDataset
from learn import train, test
from constants import device
from postprocess import evaluate_models

Xtrn, Xtst, Ytrn, Ytst = preprocess.load_spreadspoke()

print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

training_data = NFLDataset(Xtrn, Ytrn)
test_data = NFLDataset(Xtst, Ytst)
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0005)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

evaluate_models(model, Xtrn, Xtst, Ytrn, Ytst)