import myNetwork

class params():
    user = None # to specify the network we'll use
    net = None
    netName = None
    # hyperparameters
    batchSize = None
    batchSizeVal = None
    learningRate = None
    nbEpochs = None
    augmentDataSet = False

    def __init__(self, whoseConf):
        user = whoseConf.lower()
        if(user == 'thomas'):
            print("Setting Thomas configuration")
            setParamThomas(self)
        elif(user == 'hadrian'):
            print("Setting Hadrian configuration")
            setParamHadrian(self)
        elif (user == 'benjamin'):
            print("Setting Benjamin configuration")
            setParamBenjamin(self)
        elif (user == 'marieme'):
            print("Setting Marieme configuration")
            setParamMarieme(self)
        else:
            print("Setting Global configuration")
            setParamGlobal(self)


#__________________Global Config__________________
def setParamGlobal(self):
    self.user = 'global'
    self.net = myNetwork.thomasNet() #à changer
    self.netName = "Main Model, based on ???"
    self.batchSize = 8
    self.batchSizeVal = 4
    self.learningRate = 0.0001
    self.nbEpochs = 50
    self.augmentDataSet = False # not implemented yet
    return

#__________________Thomas Config__________________
def setParamThomas(self):
    self.user = 'thomas'
    self.net = myNetwork.thomasNet()
    self.netName = "Thomas Model, based on U-Net"
    self.batchSize = 16 # ou 8 ? à tester
    self.batchSizeVal = 4
    self.learningRate = 0.001 # essayer d'augmenter la taille du batch et du lr
    self.nbEpochs = 2 # 30
    self.augmentDataSet = True
    return

#__________________Hadrian Config__________________
def setParamHadrian(self):
    return

#__________________Benjamin Config__________________
def setParamBenjamin(self):
    return

#__________________Marième Config__________________
def setParamMarieme(self):
    return