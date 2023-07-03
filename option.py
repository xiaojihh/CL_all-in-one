import os,argparse
import warnings

warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=500000)
parser.add_argument('--device',type=str,default='cuda:0')
parser.add_argument('--eval_step',type=int,default=5000)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--lamb1', default=0.1, type=float, help='learning rate')
parser.add_argument('--lamb2', default=3, type=float, help='learning rate')
parser.add_argument('--old_model_path',type=str,default='./checkpoints/rain', help='the pre_train model .pk')
parser.add_argument('--save_model_dir',type=str,default='./checkpoints/snow', help='save train model dir.pk')
parser.add_argument('--save_model_path',type=str,default='', help='save train model path .pk')
parser.add_argument('--net',type=str,default='ffa')
parser.add_argument('--gps',type=int,default=3,help='residual_groups')
parser.add_argument('--blocks',type=int,default=20,help='residual_blocks')
parser.add_argument('--bs',type=int,default=1,help='batch size')
parser.add_argument('--crop', type=str, default=True)
parser.add_argument('--crop_size',type=int,default=240,help='Takes effect when using --crop ')
parser.add_argument('--contrastloss', type=str, default=True, help='Contrastive Loss')
parser.add_argument('--projector', type=str, default=False, help='Projector Loss')
parser.add_argument('--no_lr_sche', type=str, default=False,help='no lr cos schedule')
parser.add_argument('--h_channels', type=int, default=16)
parser.add_argument("--task_id", type=int, default=2)


opt=parser.parse_args()
#opt.device='cuda' if torch.cuda.is_available() else 'cpu'
save_model_name=opt.net +'_'+'best.pk'
opt.save_model_dir=os.path.join(opt.save_model_dir)
opt.save_model_path=os.path.join(opt.save_model_dir, save_model_name)

print(opt)

def create_dir(path):
	if not os.path.exists(path):
		os.mkdir(path)

# create_dir(opt.save_model_dir+'task%d'%opt.task_id)