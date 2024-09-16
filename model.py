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

def get_resnet_model():
    """ ResNet models all end in an average pool layer and linear layer, so class activation maps can be computed
    with these models as well. This function returns a resnet18 model with the final layer changed to output a
    2-d vector for binary classification. """

    model = resnet18()
    in_features = model.fc.in_features
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3) ## ADDED THIS
    model.fc = nn.Linear(in_features, 2)
    

    return model
