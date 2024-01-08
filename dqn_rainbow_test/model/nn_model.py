import torch
import torch.nn as nn


class NN(nn.Module):

    def __init__(self, input_shape, output_dim, seed):
        super(NN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.Sequential(
                    nn.Linear(input_shape[0], 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                    nn.ReLU(),
                )

    def forward(self, x):
        return self.fc(x)

