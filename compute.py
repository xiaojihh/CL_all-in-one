from torch.autograd.grad_mode import no_grad
from models.FFA import FFA
from models.PerceptualLoss import LossNetwork as PerLoss
from models.ContrastLoss import LossNetwork as ContrastLoss
from option import *
from data_utils import *
from torchvision.models import vgg16
import time
import torch.nn as nn
from torch import optim

models_={
	'ffa':FFA(gps=opt.gps,blocks=opt.blocks),
}
loaders_={
	# 'ITS':ITS_train_loader,
	# 'SOTS-Indoor':ITS_test_loader
	# 'OTS_train':OTS_train_loader,
	# 'SOTS-Outdoor':OTS_test_loader
	# 'TrainA':TrainA_loader,
	# 'TestA':TestA_loader,
	# 'Dense_train':Dense_train_loader,
	# 'Dense_test':Dense_test_loader
}


T=opt.steps
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr



def train(net,loader_train,criterion,optim):
	lr=lr_schedule_cosdecay(1000000,T)
	for param_group in optim.param_groups:
		param_group['lr'] = lr

	All_Importance=[]
	Importance = []
	Star_vals=[]
	for w in net.parameters():
		All_Importance.append(torch.zeros_like(w))
		Star_vals.append(torch.zeros_like(w))
		Importance.append(torch.zeros_like(w))

	def compute_M(loader_train):
		batch_num=0
		for i, (x,y) in enumerate(loader_train, 0):
			net.train()
			x=x.to(device)
			y=y.to(device)
			output = net(x)
			loss = criterion[0](output, y)
			if opt.perloss:
				loss2=criterion[0](output,y)
				loss=loss+0.04*loss2
			if opt.contrastloss:
				loss=loss+loss2
				net.zero_grad()
				optim.zero_grad()
			loss.backward()
			with torch.no_grad():
				for i, w in enumerate(net.parameters()):
					Importance[i].mul_(batch_num / (batch_num + 1))
					Importance[i].add_(torch.abs(w.grad.data)/(batch_num+1))
			net.zero_grad()
			optim.zero_grad()
			batch_num+=1

	def compute_allM(task_id):
		l = len(Importance)
		with torch.no_grad():
			for i in range(l):
				All_Importance[i]=(All_Importance[i]*task_id + Importance[i]) / (task_id+1)


	compute_M(loader_train)
	torch.save(Importance,os.path.join(opt.save_model_dir, 'Importance.pth'))
	with torch.no_grad():
		compute_allM(opt.task_id)
		for i,w in enumerate(net.parameters()):
					Star_vals[i].copy_(w)

	torch.save(All_Importance,os.path.join(opt.save_model_dir, 'All_Importance.pth'))
	torch.save(Star_vals,os.path.join(opt.save_model_dir, 'Star.pth'))

if __name__ == "__main__":
	loader_train=loaders_[opt.trainset]
	net=models_[opt.net]
	device = torch.device(opt.device)
	# if torch.cuda.device_count() > 1:
  	# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
  	# 	# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  	# 	net = nn.DataParallel(net, device_ids=[0, 1])
	net=net.to(device)
	net.load_state_dict(torch.load(os.path.join(opt.save_model_path))['model'])

	criterion = []
	criterion.append(nn.L1Loss().to(device))
	if opt.perloss:
		vgg_model = vgg16(pretrained=True).features[:16]
		vgg_model = vgg_model.to(device)
		for param in vgg_model.parameters():
			param.requires_grad = False
		criterion.append(PerLoss(vgg_model).to(device))     # clean image 和derain image 的特征的MSEloss 
	if opt.contrastloss:
		criterion.append(ContrastLoss().to(device))
	# betas: 用于计算梯度的平均和平方的系数   eps:为了提高数值稳定性而添加到分母的一个项
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net,loader_train,criterion,optimizer)
