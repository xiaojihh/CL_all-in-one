import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse, time, os, math

from models.FFA import FFA
from datasets.Exemplar import Exemplar_Dataset
from models.Autoencoder import Encoder, Decoder
from data_utils import ots_traindata

parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--h_channels', type=int, default=8)
parser.add_argument('--path', type=str, default='./checkpoints')
opt = parser.parse_args()


def add_noise(img, factor = 0.1):
    noise_img = img + torch.rand_like(img)*factor
    noise_img = torch.clip(noise_img, 0., 1.)
    return noise_img

def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5*(1 + math.cos(t * math.pi / T))* init_lr
    return lr

if __name__ == '__main__':
    start_time = time.time()

    # train_loader = DataLoader(dataset=ots_traindata, batch_size=opt.bs, shuffle=True)
    exemplar = Exemplar_Dataset(max_num_exemplar=500)
    exemplar.load_exemplar('/data/jyl/datasets/RESIDE/OTS_beta', task_num=1)
    # exemplar.load_exemplar('/data/jyl/task0/FFA-Net/datasets/Rain100H/train', task_num=2)
    # exemplar.collect_exemplar(rain100h_traindata, task_num=2)
    train_loader = DataLoader(dataset=exemplar, batch_size=opt.bs, shuffle=True)

    device = torch.device(opt.device)
    model = FFA(3,20).to(device)
    model.load_state_dict(torch.load('./checkpoints/haze/ffa_best.pk', map_location=device)['model'])
    model.eval()
    model.freeze_all()

    pjt = Encoder(input_channels=192, h_channels=opt.h_channels, bias=True).to(device)
    r_pjt = Decoder(h_channels=opt.h_channels, inputs_channels=192, bias=True).to(device)

    criterion = nn.L1Loss().to(device)
    optim = torch.optim.Adam([{'params': filter(lambda x: x.requires_grad, pjt.parameters())},
                              {'params': filter(lambda x: x.requires_grad, r_pjt.parameters())}], lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
    optim.zero_grad()

    epochs = opt.epochs
    for epoch in range(epochs):
        
        lr = lr_schedule_cosdecay(epoch,epochs)
        for param_group in optim.param_groups:
            param_group['lr'] = lr
        for x in train_loader:
            x = x.to(device)
            _, f = model(x, return_f=True)

            pjt.zero_grad()
            r_pjt.zero_grad()
            optim.zero_grad()

            h = pjt(f)
            r_f = r_pjt(h)
            loss = criterion(f, r_f)
            loss.backward()
            optim.step()
            print(f'\rLoss:{loss.item():.5f} |epoch:{epoch+1}/{epochs} |time_used{(time.time()-start_time)/60:.1f}min', end='', flush=True)
        torch.save(pjt.state_dict(), os.path.join(opt.path, 'encoder_haze.pth'))
        torch.save(r_pjt.state_dict(), os.path.join(opt.path, 'decoder_haze.pth'))
        print('')