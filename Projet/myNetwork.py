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


class H_small(nn.Module):
    def __init__(self):
        super(H_small, self).__init__()
        self.conv_layer1 = conv_layer(1,10,3,1)
        self.Dconv_layer1 = conv_layer(10,4,3,1)

    def forward(self,x):
        x, id1 = F.max_pool2d(self.conv_layer1(x), kernel_size=2, stride=2, return_indices=True)
        x = self.Dconv_layer1(F.max_unpool2d(x, id1, kernel_size=2, stride=2))
        x = F.softmax(x, dim=1)
        return x




class H_Unet(nn.Module):
    def __init__(self):
        super(H_Unet,self).__init__()
        
        features = 32

        self.encoder11 = conv_layer(1,features,3,1)
        self.encoder12 = conv_layer(features, features, 3, 1)

        self.encoder21 = conv_layer(features,features * 2,3,1)
        self.encoder22 = conv_layer(features * 2, features * 2, 3, 1)

        self.encoder31 = conv_layer(features * 2,features * 4,3,1)
        self.encoder32 = conv_layer(features * 4, features * 4, 3, 1)

        self.encoder41 = conv_layer(features * 4,features * 8,3,1)
        self.encoder42 = conv_layer(features * 8, features * 8, 3, 1)

        self.bottleneck1 = conv_layer(features * 8, features * 16, 3, 1)
        self.bottleneck2 = conv_layer(features * 16, features * 16, 3, 1)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)#upsampling
        self.decoder41 = conv_layer(features * 16, features * 8, 3, 1)# features 16 parce que concat, 8 * 2
        self.decoder42 = conv_layer(features * 8, features * 8, 3, 1)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder31 = conv_layer(features * 8, features * 4, 3, 1)
        self.decoder32 = conv_layer(features * 4, features * 4, 3, 1)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder21 = conv_layer(features * 4, features * 2, 3, 1)
        self.decoder22 = conv_layer(features * 2, features * 2, 3, 1)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder11 = conv_layer(features * 2, features, 3, 1)
        self.decoder12 = conv_layer(features, features, 3, 1)

        self.conv = nn.Conv2d(features, 4, 1)


    def forward(self, x):

        enc1 = self.encoder12(self.encoder11(x))
        enc2 = self.encoder22(self.encoder21(F.max_pool2d(enc1,kernel_size=2, stride=2)))
        enc3 = self.encoder32(self.encoder31(F.max_pool2d(enc2,kernel_size=2, stride=2)))
        enc4 = self.encoder42(self.encoder41(F.max_pool2d(enc3,kernel_size=2, stride=2)))

        bottleneck = self.bottleneck2(self.bottleneck1(F.max_pool2d(enc4,kernel_size=2, stride=2)))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder42(self.decoder41(dec4))

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder32(self.decoder31(dec3))

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder22(self.decoder21(dec2))

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder12(self.decoder11(dec1))

        logits = self.conv(dec1)

        return F.softmax(logits, dim=1)



#__________________Benjamin Network__________________


#__________________Marieme Network__________________