import numpy as np
import os
import sys
import random
import shutil
from pathlib import Path

alpha = 0.5
client_num = 8
#random.seed(10)
def split_list_by_ratio(input_list, test_ratio):
    #random.shuffle(input_list)  # 随机打乱列表中的元素顺序
    split_index = int(len(input_list) * test_ratio)  # 根据比例计算划分的索引位置
    test_list, train_list = input_list[:split_index], input_list[split_index:]  # 划分成两个列表
    return train_list, test_list

meta_data = np.load("/home/ly/code/src/code and baselines/dataset/imagenet2012/imagenet200/index_"+str(client_num)+"_alpha"+str(alpha)+".npy", 
                    allow_pickle=True).tolist()
all_image_path = Path("/home/ly/code/src/code and baselines/dataset/imagenet2012/imagenet200/train")

print(type(meta_data))
print(len(meta_data))
print(meta_data.keys())
print(meta_data[0][0])

folder_path = "/home/ly/code/src/code and baselines/dataset/imagenet2012/imagenet200/fed_data/client_"+str(client_num)+"non-iid_"+str(alpha)

for client_id, image_path_list in meta_data.items():
    print("Client "+str(client_id)+" processing!!!")
    clietn_folder = os.path.join(folder_path, 'client_'+str(client_id))
    client_train_folder = os.path.join(clietn_folder, 'train')
    client_test_folder = os.path.join(clietn_folder, 'test')
    os.makedirs(clietn_folder, exist_ok=True)
    os.makedirs(client_train_folder, exist_ok=True)
    os.makedirs(client_test_folder, exist_ok=True)
    label_imagesPath_dict = {}
    for i in range(len(image_path_list)):
        images_label = image_path_list[i].split('/')[1]
        if images_label not in label_imagesPath_dict.keys():
            label_imagesPath_dict[images_label] = [image_path_list[i]]
        else:
            label_imagesPath_dict[images_label].append(image_path_list[i])
    #为每一个训练集和测试集建立多个图像文件夹，每一个子文件夹下文件标签相同
    for label in label_imagesPath_dict.keys():
        client_train_label_folder = os.path.join(client_train_folder, label)
        client_test_label_folder = os.path.join(client_test_folder, label)
        os.makedirs(client_train_label_folder, exist_ok=True)
        os.makedirs(client_test_label_folder, exist_ok=True)
    #开始转移图片
    for label, image_path_list in label_imagesPath_dict.items(): 
        train_list, test_list = split_list_by_ratio(label_imagesPath_dict[label], test_ratio=0.2)
        train_destination_folder = os.path.join(client_train_folder, label)
        test_destination_folder = os.path.join(client_test_folder, label)
        for image_path in train_list:
            image_path = all_image_path.joinpath(image_path)
            shutil.copy(image_path, train_destination_folder)
        for image_path in test_list:
            image_path = all_image_path.joinpath(image_path)
            shutil.copy(image_path, test_destination_folder)
        