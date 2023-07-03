import os, glob, random
import re
from shutil import copyfile

def create_dir(path):
	if not os.path.exists(path):
		os.mkdir(path)

def findLastCheckpoint(save_dir):
    file_list= glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        step_exist = []
        for file_ in file_list:
            result = re.findall('.*epoch(.*).pth', file_)
            step_exist.append(int(result[0]))
        initial_step = max(step_exist)
    else:
        initial_step = 0
    return initial_step

def normalize(data):
    return data / 255.


if __name__ =='__main__':
    s_path = '/data/chengde/task4/FFA/datasets/RESIDE/OTS_beta/hazy/'
    d_path = '/data/chengde/task4/FFA/datasets/RESIDE/OTS_beta/exemplar/'
    img_list_dir = os.listdir(s_path)
    index_list = random.sample(img_list_dir, 500)
    for img in index_list:
        copyfile(s_path+img, d_path+img)