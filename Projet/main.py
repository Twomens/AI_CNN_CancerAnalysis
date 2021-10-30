from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
from myNetwork import Net

import medicalDataLoader
import argparse
from utils import *

import random
import torch

from PIL import Image, ImageOps


import warnings
warnings.filterwarnings("ignore")

def runTraining(args):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    ## Get statistics
    batch_size = args.batch_size
    batch_size_val = args.batch_size_val

    lr = args.lr
    epoch = args.epochs
    root_dir = 'Data/'  # path to the dataset

    print(' Dataset: {} '.format(root_dir))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set_full = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=False,
                                                      equalize=False)

    train_loader_full = DataLoader(train_set_full,
                              batch_size=batch_size,
                              worker_init_fn=np.random.seed(0),
                              num_workers=0,
                              shuffle=True)


    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=False)

    # Initialize
    num_classes = args.num_classes

    print("~~~~~~~~~~~ Creating the CNN model ~~~~~~~~~~")
    #### Create your own model #####

    net = Net()


    print(" Model Name: {}".format(args.modelName))

    print("Total params: {0:,}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    #### Loss Initialization ####
    CE_loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        net.cuda()
        CE_loss.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99))

    ### To save statistics ####
    lossTotalTraining = []
    Best_loss_val = 1000
    BestEpoch = 0
    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")

    directory = 'Results/Statistics/' + args.modelName

    if os.path.exists(directory)==False:
        os.makedirs(directory)

    for i in range(epoch):
        net.train()
        lossEpoch = []
        num_batches = len(train_loader_full)
        for j, data in enumerate(train_loader_full):
            ### Set to zero all the gradients
            net.zero_grad()
            optimizer.zero_grad()

            images, labels, img_names = data

            ### From numpy to torch variables
            labels = to_var(labels)
            images = to_var(images)

            ################### Train ###################
            #-- The CNN makes its predictions (forward pass)
            net_predictions = net(images)

            #-- Compute the loss --#
            segmentation_classes = getTargetSegmentation(labels)
            CE_loss_value = CE_loss(net_predictions, segmentation_classes)
            lossTotal = CE_loss_value

            lossTotal.backward()
            optimizer.step()

            lossEpoch.append(lossTotal.cpu().data.numpy())
            printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f}, ".format(lossTotal))

            #### Jose-TIP: Is it the best option to display only the loss value??? ####

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossTotalTraining.append(lossEpoch)

        printProgressBar(num_batches, num_batches,
                             done="[Training] Epoch: {}, LossG: {:.4f}".format(i,lossEpoch))

        loss_val = inference(net, val_loader, args.modelName, i)

        np.save(os.path.join(directory, 'Losses.npy'), lossTotalTraining)

        ### Save latest model ####

        #### Jose-TIP: Is it the best one??? ####

        if not os.path.exists('./models/' + args.modelName):
            os.makedirs('./models/' + args.modelName)

        torch.save(net.state_dict(), './models/' + args.modelName + '/' + str(i) + '_Epoch')


        print("###                                                       ###")
        print("###  [VAL]  Best Loss : {:.4f} at epoch {}  ###".format(Best_loss_val, BestEpoch))
        print("###                                                       ###")

        if i % (BestEpoch + 100) == 0 and i>0:
            for param_group in optimizer.param_groups:
                lr = lr*0.5
                param_group['lr'] = lr
                print(' ----------  New learning Rate: {}'.format(lr))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--modelName",default="Test_Model",type=str)
    parser.add_argument('--batch_size',default=8,type=int)
    parser.add_argument('--batch_size_val',default=4,type=int)
    parser.add_argument('--num_classes',default=4,type=int)
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--lr',default=0.0001,type=float)
    args=parser.parse_args()
    runTraining(args)