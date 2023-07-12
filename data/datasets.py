import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import functional as FF



def normalize(data):
    return data / 255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])





class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=240,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        self.haze_imgs_dir=self.haze_imgs_dir
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,20000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('.')[0].split('_')[0]
        clear_name=id+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)

class Rain100H_Dataset(data.Dataset):
    def __init__(self, data_path='', train=True):
        super(Rain100H_Dataset, self).__init__()
        self.data_path=data_path
        self.rain_path=os.path.join(self.data_path, 'rain')
        self.norain_path=os.path.join(self.data_path, 'norain')
        self.train = train
    
    def __getitem__(self, index):
        if self.train:
            rain_file = "%d.png" % (index + 1)
            norain_file='%d.png' % (index + 1)
        else:
            rain_file = "norain-%dx2.png" % (index + 1)
            norain_file='norain-%d.png' % (index + 1)
        
        rain_img=cv2.imread(os.path.join(self.rain_path, rain_file))
        b, g, r = cv2.split(rain_img)
        rain_img = cv2.merge([r, g, b])
        rain_img=np.float32(normalize(rain_img))
        rain_img=rain_img.transpose(2,0,1)
        
        
        norain_img=cv2.imread(os.path.join(self.norain_path, norain_file))
        b, g, r = cv2.split(norain_img)
        norain_img = cv2.merge([r, g, b])
        norain_img=np.float32(normalize(norain_img))
        norain_img=norain_img.transpose(2,0,1)

        norain_img=torch.Tensor(norain_img)
        rain_img=torch.Tensor(rain_img)
        rain_img=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(rain_img)
    
        return rain_img, norain_img
    
    def __len__(self):
        imgs=os.listdir(self.rain_path)
        return len(imgs)

class CSD_Dataset(data.Dataset):
    def __init__(self, data_path, train):
        super(CSD_Dataset,self).__init__()
        self.data_path=data_path
        self.snow_path=os.path.join(self.data_path, 'Snow')
        self.nosnow_path=os.path.join(self.data_path, 'Gt')
        self.train = train
    
    def __getitem__(self, index):

        snow_file = "%d.tif" % (index + 1)
        nosnow_file='%d.tif' % (index + 1)

        
        snow_img=cv2.imread(os.path.join(self.snow_path, snow_file))
        b, g, r = cv2.split(snow_img)
        snow_img = cv2.merge([r, g, b])
        snow_img=np.float32(normalize(snow_img))
        snow_img=snow_img.transpose(2,0,1)
        
        
        nosnow_img=cv2.imread(os.path.join(self.nosnow_path, nosnow_file))
        b, g, r = cv2.split(nosnow_img)
        nosnow_img = cv2.merge([r, g, b])
        nosnow_img=np.float32(normalize(nosnow_img))
        nosnow_img=nosnow_img.transpose(2,0,1)

        nosnow_img=torch.Tensor(nosnow_img)
        snow_img=torch.Tensor(snow_img)
        snow_img=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(snow_img)
    
        return snow_img, nosnow_img
    
    def __len__(self):
        imgs=os.listdir(self.snow_path)
        return len(imgs)

class Snow100K_Dataset(data.Dataset):
    def __init__(self,path,train,size=240,format='.jpg'):
        super(Snow100K_Dataset,self).__init__()
        self.size=size
        self.train=train
        self.format=format
        self.snow_imgs_dir=os.listdir(os.path.join(path,'synthetic'))
        self.snow_imgs=[os.path.join(path,'synthetic',img) for img in self.snow_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
    def __getitem__(self, index):
        snow=Image.open(self.snow_imgs[index])
        if isinstance(self.size,int):
            while snow.size[0]<self.size or snow.size[1]<self.size :
                index=random.randint(0,20000)
                snow=Image.open(self.snow_imgs[index])
        img=self.snow_imgs[index]
        img = img.split('/')[-1]
        clear=Image.open(os.path.join(self.clear_dir,img))
        clear=tfs.CenterCrop(snow.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(snow,output_size=(self.size,self.size))
            snow=FF.crop(snow,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        snow,clear=self.augData(snow.convert("RGB") ,clear.convert("RGB") )
        return snow,clear
    def augData(self,data,target):
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.snow_imgs)




def get_trainloader(args):

    path = args.data_path

    ots_traindata=RESIDE_Dataset(path+'/RESIDE/OTS_beta',train=True, size=args.crop_size, format='.jpg')
    rain100h_traindata=Rain100H_Dataset(path+'/Rain100H/train', train=True)
    snow100k_traindata = Snow100K_Dataset(path+'/Snow100K/train', train=True, size=args.crop_size)

    ots_loadertrain=DataLoader(dataset=ots_traindata, batch_size=args.bs, shuffle=True)
    rain100h_loadertrain=DataLoader(dataset=rain100h_traindata, batch_size=args.bs, shuffle=True)
    snow100k_loadertrain = DataLoader(dataset=snow100k_traindata, batch_size=args.bs, shuffle=True)

    TASK = {'haze': ots_loadertrain, 'rain': rain100h_loadertrain, 'snow': snow100k_loadertrain}

    trainloader = []

    for task in args.task_order:
        trainloader.append(TASK[task])
    return trainloader

def get_testloader(args):
    path = args.data_path

    ots_testdata=RESIDE_Dataset(path+'/RESIDE/SOTS/outdoor',train=False,size='whole img',format='.png')
    rain100h_testdata=Rain100H_Dataset(data_path=path+'/Rain100H/test', train=False)
    snow100k_testdata = Snow100K_Dataset(path+'/Snow100K/test/Snow100K-M', train = False, size='whole img')

    ots_loadertest=DataLoader(dataset=ots_testdata, batch_size=1, shuffle=True)
    rain100h_loadertest=DataLoader(dataset=rain100h_testdata, batch_size=1, shuffle=True)
    snow100k_loadertest = DataLoader(dataset=snow100k_testdata, batch_size=1)

    TASK = {'haze': ots_loadertest, 'rain': rain100h_loadertest, 'snow': snow100k_loadertest}

    testloader = []

    for task in args.task_order:
        testloader.append(TASK[task])
    return testloader