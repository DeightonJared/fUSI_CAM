import torch.nn as nn
from torchvision.models import resnet18


class MyCNN(nn.Module):
    """ To be able to compute the class activation maps, the only requirement on the CNN is that
    it must end with the Average Pooling Layer, followed by a single Linear Layer. Otherwise, everything else about
    the architecture can be changed."""

    def __init__(self):
        super().__init__()

        self.act = nn.ELU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv2d(1, 32, 5, 2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 5, 2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
    
        self.linear = nn.Linear(128, 2)

    def forward(self, x):

        x = self.act(self.conv1(x)) 
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        x = self.pool(x)

        # Reshape the tensor from size (N, C, 1, 1) to just (N, C)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x

import torch.nn as nn

class MyCNN_flexible(nn.Module):
    """Flexible CNN Architecture for testing various configurations."""
    def __init__(self, num_filters=32, num_conv_layers=4, activation='ELU'):
        super().__init__()

        self.act = {
            'ReLU': nn.ReLU(),
            'ELU': nn.ELU(),
            'LeakyReLU': nn.LeakyReLU()
        }[activation]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.convs = nn.ModuleList()
        in_channels = 1

        for i in range(num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=2, padding=1))
            in_channels = num_filters

        self.linear = nn.Linear(num_filters, 2)

    def forward(self, x):
        for conv in self.convs:
            x = self.act(conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear(x)
        return x
