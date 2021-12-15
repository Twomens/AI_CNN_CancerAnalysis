import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

########################################################################################################
#__________________________________LE MODELE FINAL EST TOUT EN BAS_____________________________________#
########################################################################################################


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

    def __init__(self):
        super(thomasNet, self).__init__()

        # INPUT : 256x256

        # CONV 1
        self.conv1 = nn.Conv2d(1,10,3, padding = 1)
        self.batch1 = nn.BatchNorm2d(10)
        self.act1 = nn.ReLU()

        # # CONV 2
        self.conv2 = nn.Conv2d(10,20,3,padding =1)
        self.batch2 = nn.BatchNorm2d(20)
        self.act2 = nn.ReLU()
        #
        # # CONV 3
        self.conv3 = nn.Conv2d(20,30,3,padding =1)
        self.batch3 = nn.BatchNorm2d(30)
        self.act3 = nn.ReLU()
        #
        # # DECONV 4
        self.conv4 = nn.Conv2d(30,20,3,padding =1)
        self.batch4 = nn.BatchNorm2d(20)
        self.act4 = nn.ReLU()
        #
        # # DECONV 5
        self.conv5 = nn.Conv2d(20,10,3,padding =1)
        self.batch5 = nn.BatchNorm2d(10)
        self.act5 = nn.ReLU()

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

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.act4(x)
        
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.act5(x)

        x = self.conv6(x)
        x = self.batch6(x)
        x = self.act6(x)

        return self.norm(x)

#__________________Hadrien Network__________________

def conv_layer(chan_in, chan_out, conv_ker, conv_pad, dropout=False, dpr=0.4):
    layer = None
    if(dropout):
        layer = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, conv_ker, padding=conv_pad),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(),
            nn.Dropout(p=dpr)
        )
    else:  
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


def Dconv_layer(chan_in, chan_out, conv_ker, conv_pad,dropout=False, dpr=0.3):
    layer = None
    if(dropout):
        layer = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, conv_ker, padding=conv_pad),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(),
            nn.Conv2d(chan_out, chan_out, conv_ker, padding=conv_pad),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(),
            nn.Dropout(p=dpr)
        )
    else:
        layer = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, conv_ker, padding=conv_pad),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(),
            nn.Conv2d(chan_out, chan_out, conv_ker, padding=conv_pad),
            nn.BatchNorm2d(chan_out),
            nn.ReLU()
        )
    return layer

class H_Dunet(nn.Module):
    def __init__(self):
        super(H_Dunet,self).__init__()
        features = 32

        self.pool = nn.MaxPool2d((2, 2))

        self.encoder11 = conv_layer(1,features,3,1,True)
        self.encoder12 = conv_layer(features,features * 2,3,1,True)
        self.encoder13 = conv_layer(features * 2, features * 4,3,1,True)
        self.encoder14 = conv_layer(features * 4, features * 8,3,1,True)

        self.bottleneck1 = conv_layer(features *8, features * 16,3,1)
        
        self.upconv14 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder14 = conv_layer(features * 16, features * 8, 3, 1)
        
        self.upconv13 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder13 = conv_layer(features * 8, features * 4, 3, 1)
        
        self.upconv12 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder12 = conv_layer(features * 4, features * 2, 3, 1)

        self.upconv11 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder11 = conv_layer(features * 2, features, 3, 1)

        self.output1 = nn.Conv2d(features, 4, 1)

        self.encoder21 = conv_layer(4,features,3,1,True)
        self.encoder22 = conv_layer(features,features * 2,3,1,True)
        self.encoder23 = conv_layer(features * 2, features * 4,3,1,True)
        self.encoder24 = conv_layer(features * 4, features * 8,3,1,True)

        self.bottleneck2 = conv_layer(features *8, features * 16,3,1)
        
        self.upconv24 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder24 = conv_layer(features * 3 * 8, features * 8, 3, 1)
        
        self.upconv23 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder23 = conv_layer(features * 3 * 4, features * 4, 3, 1)
        
        self.upconv22 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder22 = conv_layer(features * 3 * 2, features * 2, 3, 1)

        self.upconv21 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder21 = conv_layer(features * 3, features, 3, 1)

        self.output2 = nn.Conv2d(features, 4, 1)

    def forward(self, x):

        ## UNET 1 ##
        enc11 = self.encoder11(x)
        enc12 = self.encoder12(self.pool(enc11))
        enc13 = self.encoder13(self.pool(enc12))
        enc14 = self.encoder14(self.pool(enc13))

        bot1 = self.bottleneck1(self.pool(enc14))

        dec14 = self.upconv14(bot1)
        dec14 = torch.cat((dec14, enc14), dim=1)
        dec14 = self.decoder14(dec14)

        dec13 = self.upconv13(dec14)
        dec13 = torch.cat((dec13, enc13), dim=1)
        dec13 = self.decoder13(dec13)

        dec12 = self.upconv12(dec13)
        dec12 = torch.cat((dec12, enc12), dim=1)
        dec12 = self.decoder12(dec12)

        dec11 = self.upconv11(dec12)
        dec11 = torch.cat((dec11, enc11), dim=1)
        dec11 = self.decoder11(dec11)

        out1 = self.output1(dec11)

        ## Unet 2
        x2 = torch.mul(x, out1)

        enc21 = self.encoder21(x2)
        enc22 = self.encoder22(self.pool(enc21))
        enc23 = self.encoder23(self.pool(enc22))
        enc24 = self.encoder24(self.pool(enc23))

        bot2 = self.bottleneck2(self.pool(enc24))

        dec24 = self.upconv24(bot2)
        dec24 = torch.cat((dec24, enc14, enc24), dim=1)
        dec24 = self.decoder24(dec24)

        dec23 = self.upconv23(dec24)
        dec23 = torch.cat((dec23, enc13, enc23), dim=1)
        dec23 = self.decoder23(dec23)

        dec22 = self.upconv22(dec23)
        dec22 = torch.cat((dec22, enc12, enc22), dim=1)
        dec22 = self.decoder22(dec22)

        dec21 = self.upconv21(dec22)
        dec21 = torch.cat((dec21, enc11, enc21), dim=1)
        dec21 = self.decoder21(dec21)

        out2 = self.output2(dec21)
        return out2     

class H_Dunet2(nn.Module):
    def __init__(self):
        super(H_Dunet,self).__init__()
        features = 32

        self.pool = nn.MaxPool2d((2, 2))

        self.encoder11 = Dconv_layer(1,features,3,1, True)
        self.encoder12 = Dconv_layer(features,features * 2,3,1, True)
        self.encoder13 = Dconv_layer(features * 2, features * 4,3,1, True)
        self.encoder14 = Dconv_layer(features * 4, features * 8,3,1, True)

        self.bottleneck1 = Dconv_layer(features *8, features * 16,3,1)
        
        self.upconv14 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder14 = Dconv_layer(features * 16, features * 8, 3, 1)
        
        self.upconv13 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder13 = Dconv_layer(features * 8, features * 4, 3, 1)
        
        self.upconv12 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder12 = Dconv_layer(features * 4, features * 2, 3, 1)

        self.upconv11 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder11 = Dconv_layer(features * 2, features, 3, 1)

        self.output1 = nn.Conv2d(features, 4, 1)

        self.encoder21 = Dconv_layer(4,features,3,1, True)
        self.encoder22 = Dconv_layer(features,features * 2,3,1, True)
        self.encoder23 = Dconv_layer(features * 2, features * 4,3,1, True)
        self.encoder24 = Dconv_layer(features * 4, features * 8,3,1, True)

        self.bottleneck2 = Dconv_layer(features *8, features * 16,3,1)
        
        self.upconv24 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder24 = Dconv_layer(features * 3 * 8, features * 8, 3, 1)
        
        self.upconv23 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder23 = Dconv_layer(features * 3 * 4, features * 4, 3, 1)
        
        self.upconv22 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder22 = Dconv_layer(features * 3 * 2, features * 2, 3, 1)

        self.upconv21 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder21 = Dconv_layer(features * 3, features, 3, 1)

        self.output2 = nn.Conv2d(features, 4, 1)

    def forward(self, x):

        ## UNET 1 ##
        enc11 = self.encoder11(x)
        enc12 = self.encoder12(self.pool(enc11))
        enc13 = self.encoder13(self.pool(enc12))
        enc14 = self.encoder14(self.pool(enc13))

        bot1 = self.bottleneck1(self.pool(enc14))

        dec14 = self.upconv14(bot1)
        dec14 = torch.cat((dec14, enc14), dim=1)
        dec14 = self.decoder14(dec14)

        dec13 = self.upconv13(dec14)
        dec13 = torch.cat((dec13, enc13), dim=1)
        dec13 = self.decoder13(dec13)

        dec12 = self.upconv12(dec13)
        dec12 = torch.cat((dec12, enc12), dim=1)
        dec12 = self.decoder12(dec12)

        dec11 = self.upconv11(dec12)
        dec11 = torch.cat((dec11, enc11), dim=1)
        dec11 = self.decoder11(dec11)

        out1 = self.output1(dec11)

        ## Unet 2
        x2 = torch.mul(x, out1)

        enc21 = self.encoder21(x2)
        enc22 = self.encoder22(self.pool(enc21))
        enc23 = self.encoder23(self.pool(enc22))
        enc24 = self.encoder24(self.pool(enc23))

        bot2 = self.bottleneck2(self.pool(enc24))

        dec24 = self.upconv24(bot2)
        dec24 = torch.cat((dec24, enc14, enc24), dim=1)
        dec24 = self.decoder24(dec24)

        dec23 = self.upconv23(dec24)
        dec23 = torch.cat((dec23, enc13, enc23), dim=1)
        dec23 = self.decoder23(dec23)

        dec22 = self.upconv22(dec23)
        dec22 = torch.cat((dec22, enc12, enc22), dim=1)
        dec22 = self.decoder22(dec22)

        dec21 = self.upconv21(dec22)
        dec21 = torch.cat((dec21, enc11, enc21), dim=1)
        dec21 = self.decoder21(dec21)

        out2 = self.output2(dec21)
        return out2     


#__________________Benjamin Network__________________

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=4):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


########################################################################################################
#____________________________FINAL MODEL : DOUBLE ATTENTION UNET ______________________________________#
########################################################################################################


class D_AttU(nn.Module):
    def __init__(self, img_ch=1, output_ch=4):
        super(D_AttU, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]#64 128, 256, 512, 1024
        #32 64 128 256 512 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ################################# ATTU 1 #############################################

        self.Conv11 = conv_block(img_ch, filters[0])
        self.Conv12 = conv_block(filters[0], filters[1])
        self.Conv13 = conv_block(filters[1], filters[2])
        self.Conv14 = conv_block(filters[2], filters[3])
        self.Conv15 = conv_block(filters[3], filters[4])

        self.Up15 = up_conv(filters[4], filters[3])
        self.Att15 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv15 = conv_block(filters[4], filters[3])

        self.Up14 = up_conv(filters[3], filters[2])
        self.Att14 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv14 = conv_block(filters[3], filters[2])

        self.Up13 = up_conv(filters[2], filters[1])
        self.Att13 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv13 = conv_block(filters[2], filters[1])

        self.Up12 = up_conv(filters[1], filters[0])
        self.Att12 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=16)
        self.Up_conv12 = conv_block(filters[1], filters[0])

        self.Conv1 = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        ################################# ATTU 2 ################################################

        self.Conv21 = conv_block(output_ch, filters[0])
        self.Conv22 = conv_block(filters[0], filters[1])
        self.Conv23 = conv_block(filters[1], filters[2])
        self.Conv24 = conv_block(filters[2], filters[3])
        self.Conv25 = conv_block(filters[3], filters[4])

        self.Up25 = up_conv(filters[4], filters[3])
        self.Att25 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv25 = conv_block(filters[3] * 3, filters[3])

        self.Up24 = up_conv(filters[3], filters[2])
        self.Att24 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv24 = conv_block(filters[2] * 3, filters[2])

        self.Up23 = up_conv(filters[2], filters[1])
        self.Att23 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv23 = conv_block(filters[1] * 3, filters[1])

        self.Up22 = up_conv(filters[1], filters[0])
        self.Att22 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=16)
        self.Up_conv22 = conv_block(filters[0] * 3, filters[0])

        self.Conv2 = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        ############## ATTU 1 ############################
        e1 = self.Conv11(x)

        e2 = self.pool(e1)
        e2 = self.Conv12(e2)

        e3 = self.pool(e2)
        e3 = self.Conv13(e3)

        e4 = self.pool(e3)
        e4 = self.Conv14(e4)

        e5 = self.pool(e4)
        e5 = self.Conv15(e5)

        d5 = self.Up15(e5)
        x4 = self.Att15(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv15(d5)

        d4 = self.Up14(d5)
        x3 = self.Att14(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv14(d4)

        d3 = self.Up13(d4)
        x2 = self.Att13(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv13(d3)

        d2 = self.Up12(d3)
        x1 = self.Att12(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv12(d2)

        out1 = self.Conv1(d2)
        x2 = torch.mul(x, out1)

        ####################### ATTU 2 ###############################

        e21 = self.Conv21(x2)

        e22 = self.pool(e21)
        e22 = self.Conv22(e22)

        e23 = self.pool(e22)
        e23 = self.Conv23(e23)

        e24 = self.pool(e23)
        e24 = self.Conv24(e24)

        e25 = self.pool(e24)
        e25 = self.Conv25(e25)

        d25 = self.Up25(e25)
        x24 = self.Att25(g=d25, x=e24)
        d25 = torch.cat((x24, d25, e4), dim=1)
        d25 = self.Up_conv25(d25)

        d24 = self.Up24(e24)
        x23 = self.Att24(g=d24, x=e23)
        d24 = torch.cat((x23, d24, e3), dim=1)
        d24 = self.Up_conv24(d24)

        d23 = self.Up23(e23)
        x22 = self.Att23(g=d23, x=e22)
        d23 = torch.cat((x22, d23, e2), dim=1)
        d23 = self.Up_conv23(d23)

        d22 = self.Up22(e22)
        x21 = self.Att22(g=d22, x=e21)
        d22 = torch.cat((x21, d22, e1), dim=1)
        d22 = self.Up_conv22(d22)

        out2 = self.Conv2(d22)

        return out2






