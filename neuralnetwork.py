from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 3), # team0 covers, team1 covers, push
        )
        
    def forward(self, x):
        return self.network(x)