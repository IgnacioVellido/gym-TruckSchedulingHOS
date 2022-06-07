import copy
from torch import nn

class DriverNet(nn.Module):
    """ Mini Linear network
    input -> linear (hidden) -> ReLu -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        rows, cols = input_dim
        

        self.online = nn.Sequential(
            # nn.Flatten(0), # TODO: Remove 0, only using it while input is 1D
            nn.Linear(rows * cols, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)