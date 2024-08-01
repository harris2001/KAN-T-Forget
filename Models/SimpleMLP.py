import torch
from torch import nn
from torch.nn import functional as F

# Create conventional MLP model
class SimpleMLP(nn.Module):

    def __init__(self):
        super(SimpleMLP, self).__init__()
        # 28*28 input image, 128 output units, 10 output units
        self.fc1 = nn.Linear(28*28, 2**7)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Flatten the image
        x = x.view(-1, 28*28)
        # Pass through the first layer
        x = F.relu(self.fc1(x))
        # Pass through the second layer
        x = self.fc2(x)
        # Apply log softmax
        x = F.log_softmax(x, dim=1)
        return x