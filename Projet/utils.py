import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
from progressBar import printProgressBar
from os.path import isfile, join


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    
    denom = 0.33333334 # use for ACDC this value
    return (batch / denom).round().long().squeeze()

def inference(net, img_batch, modelName, epoch):
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

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()
