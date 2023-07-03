import torchvision.transforms as tfs
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os, random

class Exemplar_Dataset(Dataset):
    def __init__(self, max_num_exemplar):
        super().__init__()
        self.max_num_exemplar = max_num_exemplar
        self.images = []
        self.transforms = tfs.Compose([
            tfs.CenterCrop(240),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
        ])
    
    def load_exemplar(self, data_path, task_num):
        self.exemplars_per_task = int(np.ceil(self.max_num_exemplar / task_num))
        self.task_num = task_num
        img_list_dir = os.listdir(data_path+'/exemplar')
        img_list_dir.sort()
        img_list = []
        for img_dir in img_list_dir:
            img = Image.open(os.path.join(data_path, 'exemplar', img_dir))
            img = self.transforms(img.convert('RGB'))
            if 'RESIDE' in data_path:
                id = img_dir.split('/')[-1].split('.')[0].split('_')[0]
                id = id + '.jpg'
                label=Image.open(os.path.join(os.path.join(data_path, 'clear', id)))
            elif 'Rain100H' in data_path:
                label=Image.open(os.path.join(os.path.join(data_path, 'norain', img_dir)))
                
            label = tfs.CenterCrop(240)(label.convert("RGB"))
            label = tfs.ToTensor()(label)
            
            img_list.append((img, label))
            # img_list.append(img)
        self.images.append(img_list)
        self._clear_more(self.exemplars_per_task)
    
    def collect_exemplar(self, train_dataset, task_num):
        self.exemplars_per_task = int(np.ceil(self.max_num_exemplar / task_num))
        self.task_num = task_num
        data_num = len(train_dataset)
        assert (data_num > self.exemplars_per_task), 'Not enough samples to store'
        select_index = random.sample(range(data_num), self.exemplars_per_task)
        self.images.append([train_dataset[idx][0] for idx in select_index])
        self._clear_more(self.exemplars_per_task)
    
    def _clear_more(self, exemplars_per_task):
        for label in range(self.task_num):
            self.images[label] = self.images[label][0:exemplars_per_task]
    
    def __getitem__(self, idx):
        label = idx // self.exemplars_per_task
        idx = idx % self.exemplars_per_task
        return self.images[label][idx]
    
    def __len__(self):
        return sum([len(self.images[label]) for label in range(self.task_num)])