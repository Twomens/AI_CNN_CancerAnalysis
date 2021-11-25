import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os

from imageDataGenerator import savePNG, saveBatchPNG
from progressBar import printProgressBar
from os.path import isfile, join


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#les classes du mask sont [0.0000, 0.3333, 0.6667, 1.0000] la fonction la rend ces classes sous la forme 0,1,2,3
def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    
    denom = 0.33333334 # use for ACDC this value
    return (batch / denom).round().long().squeeze()

#fait une inference, utilisé pour la val loss pk pas de backprop
def inference(net, img_batch, modelName, epoch,savePNG):
    total = len(img_batch)
    net.eval()

    CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        segmentation_classes = getTargetSegmentation(labels)
        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())

        if savePNG:
            saveBatchPNG(F.softmax(net_predictions,dim=1), "Data/val/Pred/epoch_" + str(epoch) + "/", img_names)


    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()

def getOneHotSegmentation(batch):
    backgroundVal = 0

    l1 = 0.4
    l2 = 0.7

    b1 = (batch == backgroundVal)
    b2 = (batch < l1) * ~b1
    b3 = (batch < l2) * ~b2 * ~b1
    b4 = (batch > l2)

    oneHotLabels = torch.cat((b1, b2, b3, b4),
                             dim=1)

    return oneHotLabels.float()

#class DiceLoss(nn.Module):
#    def