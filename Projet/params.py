from sympy.strategies.core import switch

#__________________List Parameters__________________
import myNetwork


class params():
    user = None # to specify the network we'll use
    net = None

    def __init__(whoseConf,self):
        user = whoseConf.lower()
        if(whoseConf.lower() == 'thomas'):
            print("Setting Thomas configuration")
            setParamThomas()
        elif(whoseConf.lower() == 'hadrian'):
            print("Setting Hadrian configuration")
            setParamHadrian()
        elif (whoseConf.lower() == 'benjamin'):
            print("Setting Benjamin configuration")
            setParamBenjamin()
        elif (whoseConf.lower() == 'marieme'):
            print("Setting Marieme configuration")
            setParamMarieme()
        else:
            print("Setting Global configuration")
            setParamGlobal()


#__________________Global Config__________________
def setParamGlobal(self):
    self.user = 'global'
    return

#__________________Thomas Config__________________
def setParamThomas(self):
    self.user = 'thomas'
    self.net = myNetwork.thomasNet()
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