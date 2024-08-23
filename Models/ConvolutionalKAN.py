import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientKAN import *

class ConvKAN(nn.Module):
    def __init__(self):
        super(ConvKAN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1, 1)
        self.fc = KAN(layers_hidden=[16*7*7,10], grid_size=10, spline_order=2).to('cuda')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16*7*7)
        x = self.fc(x)
        return x