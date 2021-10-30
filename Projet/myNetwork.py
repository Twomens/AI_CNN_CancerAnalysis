import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

#__________________Global Network__________________

#__________________Thomas Network__________________
class thomasNet(nn.Module):
    def __init__(self):
        super(thomasNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 1, 256)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(25088, 1152)  #
        self.fc2 = nn.Linear(1152, 10)  # (10 classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#__________________Hadrian Network__________________


#__________________Benjamin Network__________________


#__________________Marieme Network__________________