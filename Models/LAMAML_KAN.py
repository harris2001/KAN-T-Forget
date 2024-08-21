import torch.nn as nn
from avalanche.models.dynamic_modules import MultiTaskModule,\
    MultiHeadClassifier
from Scaled_KAN import FastKAN
 
####################
#     CIFAR-100
####################

class ConvCIFAR(nn.Module):
    def __init__(self, num_classes=10, device="cpu"):
        super(ConvCIFAR, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Linear layers
        # self.relu = nn.ReLU(inplace=True)
        # self.linear1 = FastKAN(nn.Linear(16*160, 320)
        # self.linear2 = nn.Linear(320, 320)
        # # Classifier
        # self.classifier = nn.Linear(320, num_classes)
        self.device = device
        self.num_classes = num_classes
        self.classifier = FastKAN(layers_hidden=[16*160,self.num_classes], grid_min=100, grid_max=101, device=self.device)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 2560)
        x = self.classifier(x)
        return x


class MTConvCIFAR(ConvCIFAR, MultiTaskModule):
    def __init__(self, device="cuda"):
        super(MTConvCIFAR, self).__init__()
        # Classifier
        # self.classifier = MultiHeadClassifier(320)
        self.classifier = FastKAN(layers_hidden=[16*160,100], grid_min=100, grid_max=101, device=self.device)
        self.device = device

    def forward(self, x, task_labels):
        x = self.conv_layers(x)
        x = x.view(-1, 16*160)
        x = self.classifer(x)
        return x