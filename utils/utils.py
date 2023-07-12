import os, glob
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