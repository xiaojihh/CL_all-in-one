from models import *
import warnings
warnings.filterwarnings('ignore')
from option import *
from torchvision.models import vgg16
from data_utils import*
import time
import torch.nn as nn
from torch import optim


from models.FFA import FFA

models_={
	'ffa':FFA(gps=opt.gps,blocks=opt.blocks),
}


if __name__=='__main__':
    # net=models_[opt.net]
    # net.recover_head(2)
    # net.add_head()
    # print(net.heads[1])
    # for i,m in enumerate(net.heads[1].parameters()):
    #     print(m.size())
    label=torch.zeros(1)
    print(label)
    print(label==0)