'''

import os
import tarfile

folder_path = "/home/***/code/src/code and baselines/dataset/imagenet2012/train/"


for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file.endswith('.tar'):
        #
        extract_folder = os.path.join(folder_path, file.split('.tar')[0])
        os.makedirs(extract_folder, exist_ok=True)
        
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(path=extract_folder)
'''

'''
import os

folder_path = "/home/***/code/src/code and baselines/dataset/imagenet2012/train"
file_type = '.tar'

for file in os.listdir(folder_path):
    if file.endswith(file_type):
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)'''

'''    
import tarfile
import os

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


# 
source_dir = "/home/***/code/src/code and baselines/dataset/imagenet2012/fed_data"
output_filename = "/home/***/code/src/code and baselines/dataset/imagenet2012/fed_data.tar"
make_tarfile(output_filename, source_dir)'''

'''#
import os
import numpy as np
train_data_path = "/home/***/code/src/code and baselines/dataset/imagenet2012/train"
lablename_list = os.listdir(train_data_path)
label=0
cla_dict={}
for name in lablename_list:
    cla_dict[name]=label
    label += 1
print(cla_dict)
np.save("/home/***/code/src/code and baselines/dataset/imagenet2012/label_dict.npy", cla_dict)'''


import os
import shutil
import sys
tiny_imagenet_path = "/home/***/code/src/code and baselines/dataset/tiny-imagenet_1.5/train"
all_train_path = "/home/***/EXP/data/ImageNet/train"
all_val_path = "/home/***/EXP/data/ImageNet/val"

tiny_train_path = "/home/***/code/src/code and baselines/dataset/imagenet2012/imagenet200/train"
tiny_val_path = "/home/***/code/src/code and baselines/dataset/imagenet2012/imagenet200/val"

tiny_lablename_list = os.listdir(tiny_imagenet_path)
for labelname in tiny_lablename_list:
    destination_path = os.path.join(tiny_train_path, labelname)
    shutil.copytree(os.path.join(all_train_path, labelname), destination_path)

for labelname in tiny_lablename_list:
    destination_path = os.path.join(tiny_val_path, labelname)
    shutil.copytree(os.path.join(all_val_path, labelname), destination_path)
