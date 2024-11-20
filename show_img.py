import torch
import torchvision.transforms as tfs

import matplotlib.pyplot as plt
import cv2, os
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision.utils import make_grid, save_image
from models.Autoencoder import *
from models.FFA import FFA
# from models.FFA_importance import FFA
# from models.modified_FFA import FFA
from models.ContrastLoss import Vgg19

def tensorshow(tensors):
    fig = plt.figure(dpi=1000)
    plt.axis('off')
    for tensor, i in zip(tensors, range(len(tensors))):
        tensor.clamp(0,1)
        img = make_grid(tensor)
        npimg = img.cpu().numpy()
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('./Feature/adjustor/h.png')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    device = torch.device('cuda:2')
    model = FFA(gps=3, blocks=20).to(device)
    # pre_dict = torch.load('./ffa_best.pk', map_location=device)['model']        # pred_dict权重的字典
    # net_state_dict = model.state_dict()                                       # 模型有哪些参数
    # temp = OrderedDict()              
    # for n, p in pre_dict.items():
    #     if n in net_state_dict:
    #         temp[n] = p
    # net_state_dict.update(temp)
    # model.load_state_dict(net_state_dict)


    # ffa_best['model'] = temp
    # torch.save(ffa_best, 'ffa_best.pk')

    model.load_state_dict(torch.load('./ffa_best_ft.pk', map_location=device)['model'])
    # model.load_state_dict(torch.load('./ffa_best_pod.pk', map_location=device)['model'])
    model.freeze_all()

    # encoder = Adjustor(in_channels=64, bias=True).to(device)
    # encoder.load_state_dict(torch.load(os.path.join('/data/jyl/FFA3/checkpoints/', 'encoder_haze.pth'), map_location=device))
    # encoder.freeze_all()
    # decoder = Decoder(h_channels=16, inputs_channels=192, bias=True).to(device)
    # decoder.load_state_dict(torch.load(os.path.join('/data/jyl/FFA2/checkpoints/', 'decoder_rain.pth'), map_location=device))
    # decoder.freeze_all() 

    # input = Image.open('./data/0329_0.8_0.12.jpg')
    # input = Image.open('./data/0003_0.8_0.2.jpg')
    # input = Image.open('./data/norain-51x2.png')
    # input = Image.open('./data/norain-133x2.png')
    # input = Image.open('./data/beautiful_smile_01122.jpg')
    # input = Image.open('./data/city_read_13800.jpg')
    # input = Image.open('./data2/task1/0070_1_0.2.jpg')
    # input = Image.open('./data/1815_1_0.2.jpg')
    # input = Image.open('./data/norain-2x2.png')

    # types = os.listdir('./data2/adverse')
    size = 800
    save_path = './data3/ft'
    for ty in ['snow_haze']:
        ty_path = os.path.join('./data3/adverse', ty)
        print(ty_path)
        files = os.listdir(ty_path)
        create_dir(os.path.join(save_path, ty))
        for file in files:
            print(file)
            input = Image.open(os.path.join(ty_path, file))
            print(input.size)
            transform = tfs.Compose([
                                    tfs.ToTensor(),
                                    # tfs.Resize((size, int(size*3/2))),
                                    tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14,0.15, 0.152])])
            input = transform(input).to(device)
            input = input.unsqueeze(0)
            if 'afc' in save_path:
                pred,_ = model(input)
            else:
                pred = model(input)
            model.zero_grad()
            pred = pred.clone().detach().to(torch.device('cpu'))
            print(os.path.join(save_path, ty, file))
            save_image(pred, os.path.join(save_path, ty, file))
            # h = encoder(input, features)
            # tensorshow(h)

    # ty_path = './datasets/CSD/Test/Snow'
    # # for i in range(1, 2000+1):
    # i = 976
    # input = Image.open(os.path.join(ty_path, f'{i}.tif'))
    # input = tfs.ToTensor()(input)
    # # create_dir(os.path.join('./data3/adverse', 'snow_haze'))
    # create_dir(os.path.join('./data3/pod', 'snow_haze'))
    # # save_image(input, os.path.join('./data3/adverse', 'snow_haze', f'{i}.png'))
    # transform = tfs.Compose([
    #                         tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14,0.15, 0.152])])
    # input = transform(input).to(device)
    # input = input.unsqueeze(0)
    # pred = model(input)
    # pred = pred.clone().detach().to(torch.device('cpu'))
    # print(os.path.join('./data3/pod', 'snow_haze', f'{i}.png'))
    # save_image(pred, os.path.join('./data3/pod', 'snow_haze', f'{i}.png'))
