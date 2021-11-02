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

#__________________Hadrien Network__________________

def conv_layer(chan_in, chan_out, conv_ker, conv_pad):
    layer = nn.Sequential(
        nn.Conv2d(chan_in, chan_out, conv_ker, padding=conv_pad),
        nn.BatchNorm2d(chan_out),
        nn.ReLU()
    )
    return layer

class H_SegNet(nn.Module):
    def __init__(self):
        super(H_SegNet, self).__init__()#?
        
        ### essaye de Segnet

        ## VGG architecture without fully connected

        self.conv_layer1 = conv_layer(1,64,3,1)
        self.conv_layer2 = conv_layer(64,128,3,1)
        self.conv_layer3 = conv_layer(128,256,3,1)
        self.conv_layer4 = conv_layer(256,512,3,1)

        ## Deconv architecture

        self.Dconv_layer1 = conv_layer(512,256,3,1)
        self.Dconv_layer2 = conv_layer(256,128,3,1)
        self.Dconv_layer3 = conv_layer(128,64,3,1)
        self.Dconv_layer4 = conv_layer(64,4,3,1)



    def forward(self,x):

        x, id1 = F.max_pool2d(self.conv_layer1(x), kernel_size=2, stride=2, return_indices=True)

        x, id2 = F.max_pool2d(self.conv_layer2(x), kernel_size=2, stride=2, return_indices=True)

        x, id3 = F.max_pool2d(self.conv_layer3(x), kernel_size=2, stride=2, return_indices=True)

        x, id4 = F.max_pool2d(self.conv_layer4(x), kernel_size=2, stride=2, return_indices=True)

        x = self.Dconv_layer1(F.max_unpool2d(x, id4, kernel_size=2, stride=2))

        x = self.Dconv_layer2(F.max_unpool2d(x, id3, kernel_size=2, stride=2))

        x = self.Dconv_layer3(F.max_unpool2d(x, id2, kernel_size=2, stride=2))

        x = self.Dconv_layer4(F.max_unpool2d(x, id1, kernel_size=2, stride=2))

        x = F.softmax(x, dim=1)

        return x


#__________________Benjamin Network__________________


#__________________Marieme Network__________________