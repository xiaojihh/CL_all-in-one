import torch 
import torch.nn as nn
from torch import optim
from torch.nn import Module

import time
from collections import OrderedDict
from copy import deepcopy
import math
import os

from  datasets.Exemplar import Exemplar_Dataset
from models.Autoencoder import Encoder
from models.FFA import FFA
from models.ContrastLoss import LossNetwork as ContrastLoss
from option import *
from data_utils import *




def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5*(1 + math.cos(t * math.pi / T))* init_lr
    return lr

def train(model:FFA, criterion, optim, loader_train, loader_test, exemplar_loader):
	losses = []
	star_step = 0
	max_ssim = 0
	max_psnr = 0
	ssims = []
	psnrs = []

	T = opt.steps
	lr = opt.lr

	model_old=None
	if opt.old_model_path:
		model_old = deepcopy(model)
		model_old.eval()
		model_old.freeze_all()
	if opt.projector:
		encoder = Encoder(input_channels=192, h_channels=opt.h_channels).to(device)
		encoder.load_state_dict(torch.load(os.path.join(opt.old_model_path, 'encoder_rain.pth'), map_location=device))


    
	model.load_state_dict(torch.load('./checkpoints/snow/net_step75000.pth', map_location=device))
	max_psnr = torch.load('./checkpoints/snow/ffa_best.pk', map_location=device)['max_psnr']
	max_ssim = torch.load('./checkpoints/snow/ffa_best.pk', map_location=device)['max_ssim']
	star_step = 75000
	p_loss = torch.zeros(1)
	kdloss = torch.zeros(1)
	conloss = torch.zeros(1)
	for step in range(star_step+1, opt.steps+1):
		model.train()
		if not opt.no_lr_sche:
			lr = lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group['lr'] = lr
		
		x, y = next(iter(loader_train))
		x = x.to(device)
		y = y.to(device)
		old_x, old_y = next(iter(exemplar_loader))
		old_x = old_x.to(device)
		old_y = old_y.to(device)
		

		model.zero_grad()
		optim.zero_grad()
		pred = model(x)
		l1loss = criterion[0](pred, y)


		pred_old, f_old = model_old(old_x, return_f=True)
		pred_new, f_new = model(old_x, return_f=True)
		kdloss = criterion[0](old_y, pred_new)
		loss = l1loss + kdloss

		if opt.contrastloss:
			conloss=0.2*criterion[1](pred, y, x) + 0.8*criterion[1](pred_new, old_y, old_x)
			loss += opt.lamb1 * conloss
		if opt.projector:
			h_old = encoder(f_old)
			h_new = encoder(f_new)
			p_loss = criterion[0](h_old, h_new)
			loss += opt.lamb2 * p_loss

		loss.backward()
		optim.step()

		losses.append(loss.item())
		print(f'\rL1loss:{l1loss.item():.5f} |kdloss:{kdloss.item():.5f} |conloss:{conloss.item():.5f} |p_loss:{p_loss.item():.5f} |step:{step}/{opt.steps} |lr: {lr:.7f} |time_used: {(time.time()-start_time)/60:.1f}min', end='', flush=True)

		if step > 0 and step % opt.eval_step == 0:
			torch.save(model.state_dict(), os.path.join(opt.save_model_dir, 'net_step%d.pth' % (step)))
			with torch.no_grad():
				ssim_eval, psnr_eval = test(model, loader_test)
			print(f'step:{step} |ssim:{ssim_eval:.4f} |psnr:{psnr_eval:.2f}')
			trainLogger.write(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)

			if ssim_eval >= max_ssim and psnr_eval >= max_psnr:
				max_ssim = ssim_eval
				max_psnr = psnr_eval
				torch.save({
					'step': step,
					'max_ssim': max_ssim,
					'max_psnr': max_psnr,
					'model': model.state_dict()
				}, opt.save_model_path)
				print(f'model saved at setp:{step} |max_ssim:{max_ssim:.4f} |max_psnr:{max_psnr:.2f}')
				trainLogger.write(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')


def test(model: Module, loader_test):
	model.eval()
	torch.cuda.empty_cache()

	ssim_eval = []
	psnr_eval = []
	print('')
	for loader in loader_test:
		ssims = []
		psnrs = []
		for _, (inputs, targets) in enumerate(loader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			pred = model(inputs)

			ssim1 = ssim(pred, targets).item()
			psnr1 = psnr(pred, targets)
			ssims.append(ssim1)
			psnrs.append(psnr1)
		trainLogger.write(f'psnr:{np.mean(psnrs):.4f} |ssim:{np.mean(ssims):.4f}')
		print(f'psnr:{np.mean(psnrs):.4f} |ssim:{np.mean(ssims):.4f}')
		ssim_eval.append(np.mean(ssims))
		psnr_eval.append(np.mean(psnrs))
    
	return np.mean(ssim_eval), np.mean(psnr_eval)


start_time = time.time()
create_dir('./Log/')
trainLogger = open('./Log/train.log', 'a+')
trainLogger.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
trainLogger.write('\n')

if __name__ == '__main__':

	device = torch.device(opt.device)
	model = FFA(gps=3, blocks=20).to(device)
	model.load_state_dict(torch.load(os.path.join(opt.old_model_path, 'ffa_best.pk'), map_location=device)['model'])
	print('load model complete')

	loader_train = snow100k_loadertrain
	loader_test = [ots_loadertest,rain100h_loadertest, snow100k_loadertest]

	exemplar = Exemplar_Dataset(max_num_exemplar=500)
	exemplar.load_exemplar('/data/jyl/datasets/RESIDE/OTS_beta', task_num=1)
	exemplar.load_exemplar('/data/jyl/datasets/Rain100H/train', task_num=2)
	# exemplar.collect_exemplar(rain100h_traindata, task_num=2)
	exemplarloader = DataLoader(dataset=exemplar, batch_size=opt.bs, shuffle=True)
	print('exemplar complete')

	criterion = []
	criterion.append(nn.L1Loss().to(device))
	if opt.contrastloss:
		criterion.append(ContrastLoss(device).to(device))

	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()

	train(model, criterion, optimizer, loader_train, loader_test, exemplarloader)
	trainLogger.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
	

