import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

#__________________Global Network__________________

#__________________Thomas Network__________________

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class thomasUNet(nn.Module):
    def __init__(self):
        super(thomasUNet, self).__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 4, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs

class thomasNet(nn.Module):

    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
    #                 padding_mode='zeros', device=None, dtype=None)
    # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None,
    #                      dtype=None)
    # torch.nn.ReLU(inplace=False)


    def __init__(self):
        super(thomasNet, self).__init__()

        # INPUT : 256x256

        # CONV 1
        self.conv1 = nn.Conv2d(1,10,3, padding = 1)
        self.batch1 = nn.BatchNorm2d(10)
        self.act1 = nn.ReLU()

        # # CONV 2
        # self.conv2 = nn.Conv2d(10,20,3,padding =1)
        # self.batch2 = nn.BatchNorm2d(20)
        # self.act2 = nn.ReLU()
        #
        # # CONV 3
        # self.conv3 = nn.Conv2d(20,30,3,padding =1)
        # self.batch3 = nn.BatchNorm2d(30)
        # self.act3 = nn.ReLU()
        #
        # # DECONV 4
        # self.conv4 = nn.Conv2d(30,20,3,padding =1)
        # self.batch4 = nn.BatchNorm2d(20)
        # self.act4 = nn.ReLU()
        #
        # # DECONV 5
        # self.conv5 = nn.Conv2d(20,10,3,padding =1)
        # self.batch5 = nn.BatchNorm2d(10)
        # self.act5 = nn.ReLU()

        # DECONV 6
        self.conv6 = nn.Conv2d(10,4,3,padding =1)
        self.batch6 = nn.BatchNorm2d(4)
        self.act6 = nn.ReLU()

        # NORMALISATION SOFTMAX
        self.norm = nn.Softmax()

        # OUTPUT : 256x256

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act1(x)

        # x = self.conv2(x)
        # x = self.batch2(x)
        # x = self.act2(x)
        #
        # x = self.conv3(x)
        # x = self.batch3(x)
        # x = self.act3(x)
        #
        # x = self.conv4(x)
        # x = self.batch4(x)
        # x = self.act4(x)
        #
        # x = self.conv5(x)
        # x = self.batch5(x)
        # x = self.act5(x)

        x = self.conv6(x)
        x = self.batch6(x)
        x = self.act6(x)

        return self.norm(x)

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

        #return F.softmax(logits, dim=1)
        return logits



#__________________Benjamin Network__________________


#__________________Marieme Network__________________