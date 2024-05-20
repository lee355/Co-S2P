'''
#将目录下的所有tar文件解压到对应名字的文件夹下
import os
import tarfile

folder_path = "/home/liyan/code/src/code and baselines/dataset/imagenet2012/train/"

# 遍历文件夹中的所有文件
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file.endswith('.tar'):
        # 创建对应的文件夹用于存放解压后的文件
        extract_folder = os.path.join(folder_path, file.split('.tar')[0])
        os.makedirs(extract_folder, exist_ok=True)
        
        # 解压tar包
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(path=extract_folder)
'''

'''#将目录下的相应后缀的文件删除
import os

folder_path = "/home/liyan/code/src/code and baselines/dataset/imagenet2012/train"
file_type = '.tar'

for file in os.listdir(folder_path):
    if file.endswith(file_type):
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)'''

'''#文件夹及其子文件夹打包成tar包        
import tarfile
import os

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


# 示例用法
source_dir = "/home/liyan/code/src/code and baselines/dataset/imagenet2012/fed_data"
output_filename = "/home/liyan/code/src/code and baselines/dataset/imagenet2012/fed_data.tar"
make_tarfile(output_filename, source_dir)'''

'''#设置标签映射关系
import os
import numpy as np
train_data_path = "/home/liyan/code/src/code and baselines/dataset/imagenet2012/train"
lablename_list = os.listdir(train_data_path)
label=0
cla_dict={}
for name in lablename_list:
    cla_dict[name]=label
    label += 1
print(cla_dict)
np.save("/home/liyan/code/src/code and baselines/dataset/imagenet2012/label_dict.npy", cla_dict)'''

#从imagenet数据集中选出小样本数据集
import os
import shutil
import sys
tiny_imagenet_path = "/home/ly/code/src/code and baselines/dataset/tiny-imagenet_1.5/train"
all_train_path = "/home/wyy/EXP/data/ImageNet/train"
all_val_path = "/home/wyy/EXP/data/ImageNet/val"

tiny_train_path = "/home/ly/code/src/code and baselines/dataset/imagenet2012/imagenet200/train"
tiny_val_path = "/home/ly/code/src/code and baselines/dataset/imagenet2012/imagenet200/val"

tiny_lablename_list = os.listdir(tiny_imagenet_path)
for labelname in tiny_lablename_list:
    destination_path = os.path.join(tiny_train_path, labelname)
    shutil.copytree(os.path.join(all_train_path, labelname), destination_path)

for labelname in tiny_lablename_list:
    destination_path = os.path.join(tiny_val_path, labelname)
    shutil.copytree(os.path.join(all_val_path, labelname), destination_path)
