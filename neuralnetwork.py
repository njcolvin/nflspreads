from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(15, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 2), # favorite covers, underdog covers, push
        )
        
    def forward(self, x):
        return self.network(x)