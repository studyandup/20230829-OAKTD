import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, seed, initial=True):
        super(LinearLayer, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.Linear(input_dim, output_dim)
        if initial:
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            torch.nn.init.constant_(self.fc.weight, 0)

    def forward(self, x):
        return self.fc(x)
