import torch
from torch import optim
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import time
import os

from models.FFA import FFA
from models.Autoencoder import Auencoder
from data.exemplar import Exemplar_Dataset
from utils.metrics import psnr, ssim
from utils.utils import create_dir
from utils.lr_schedule import lr_schedule_cosdecay



class CLAIO():
    def __init__(self, net: FFA, criterion, trainloader, testloader, device, trainLogger, args) -> None:
        self.net = net
        self.criterion = criterion
        self.trainloader = trainloader
        self.testloader = testloader
        self.trainLogger = trainLogger
        
        self.autoencoder = Auencoder(args)
        self.device = device
        self.args = args
        self.old_net = None

    def train(self, task_id):
        self.net.train_all()
        optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, self.net.parameters()), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08)

        T = self.args.steps
        lr = self.args.lr

        max_ssim = 0
        max_psnr = 0
        start_time = time.time()
        old_x, old_y = None, None

        create_dir(os.path.join(self.args.save_model_dir, self.args.task_order[task_id]))
        for step in range(1, self.args.steps+1):
            self.net.train()
            if not self.args.no_lr_sche:
                lr = lr_schedule_cosdecay(step, T, self.args.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            x, y = next(iter(self.trainloader[task_id]))
            x = x.to(self.device)
            y = y.to(self.device)
            if task_id > 0:
                old_x = next(iter(self.exemplar_loader))
                old_x = old_x.to(self.device)
            
            self.net.zero_grad()
            optimizer.zero_grad()

            loss, swloss, kdloss, p_loss = self.compute_loss(x, y, old_x)
            loss.backward()
            optimizer.step()

            print(f'\rswloss:{swloss.item():.5f} |kdloss:{kdloss.item():.5f} |p_loss:{p_loss.item():.5f} |step:{step}/{self.args.steps} |lr: {lr:.7f} |time_used: {(time.time()-start_time)/60:.1f}min', end='', flush=True)

            if step > 0 and step % self.args.eval_step == 0:
                torch.save(self.net.state_dict(), os.path.join(self.args.save_model_dir, self.args.task_order[task_id], 'net_step%d.pth' % (step)))
                ssim_eval, psnr_eval = self.test(task_id)
                print(f'step:{step} |ssim:{ssim_eval:.4f} |psnr:{psnr_eval:.2f}')
                self.trainLogger.write(f'step:{step} |ssim:{ssim_eval:.4f} |psnr:{psnr_eval:.2f}\n')
                if ssim_eval >= max_ssim and psnr_eval >= max_psnr:
                    max_ssim = ssim_eval
                    max_psnr = psnr_eval
                    torch.save({
                        'step': step,
                        'max_ssim': max_ssim,
                        'max_psnr': max_psnr,
                        'model': self.net.state_dict()
                    }, os.path.join(self.args.save_model_dir, self.args.task_order[task_id], 'ffa_best.pk'))
                    print(f'model saved at setp:{step} |max_ssim:{max_ssim:.4f} |max_psnr:{max_psnr:.2f}')
                    self.trainLogger.write(f'model saved at setp:{step} |max_ssim:{max_ssim:.4f} |max_psnr:{max_psnr:.2f}\n')

    
    def compute_loss(self, x, y, old_x):
        kdloss = torch.zeros(1)
        p_loss = torch.zeros(1)

        pred = self.net(x)
        swloss = self.criterion[0](pred, y)
        if self.args.contrastloss:
            swloss += self.args.beta1 * self.criterion[1](pred, y, x)

        loss = swloss

        if old_x is not None:
            pred_old, f_old = self.old_net(old_x, return_f=True)
            pred_new, f_new = self.net(old_x, return_f=True)
            kdloss = self.criterion[0](pred_old, pred_new)
            if self.args.contrastloss:
                kdloss +=  self.args.beta2 * self.criterion[1](pred_new, pred_old, old_x)

            loss += self.args.alpha * kdloss

        if self.args.projector and old_x is not None:
            h_old = self.autoencoder.pjt(f_old)
            h_new = self.autoencoder.pjt(f_new)
            p_loss = self.criterion[0](h_old, h_new)
            loss += self.args.lamb * p_loss
        
        return loss, swloss, kdloss, p_loss


    @torch.no_grad()
    def test(self, task_id):
        self.net.eval()
        torch.cuda.empty_cache()

        ssim_eval = []
        psnr_eval = []
        print('')
        for loader in self.testloader[:task_id+1]:
            ssims = []
            psnrs = []
            for _, (inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pred = self.net(inputs)

                ssim1 = ssim(pred, targets).item()
                psnr1 = psnr(pred, targets)
                ssims.append(ssim1)
                psnrs.append(psnr1)
            print(f'psnr:{np.mean(psnrs):.4f} |ssim:{np.mean(ssims):.4f}')
            ssim_eval.append(np.mean(ssims))
            psnr_eval.append(np.mean(psnrs))
        
        return np.mean(ssim_eval), np.mean(psnr_eval)
    
    def after_train(self, task_id):
        if task_id == 0:
            exemplar = Exemplar_Dataset(max_num_exemplar=500)
            exemplar.collect_exemplar(self.trainloader[task_id].dataset, task_num=task_id+1)
            self.exemplar_loader = DataLoader(dataset=exemplar, batch_size=self.args.bs, shuffle=True)
        else:
            self.exemplar_loader.dataset.collect_exemplar(self.trainloader[task_id], task_num=task_id+1)

        self.net.load_state_dict(torch.load(os.path.join(self.args.save_model_dir, self.args.task_order[task_id], 'ffa_best.pk'), map_location=self.device)['model'])
        self.net.freeze_all()

        self.autoencoder.train(train_loader=self.exemplar_loader, net=self.net, epochs=50, task_id=task_id)

        self.old_net = deepcopy(self.net)
        self.old_net.freeze_all()
        self.autoencoder.pjt.freeze_all()





