import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from torchvision import datasets, transforms
import torch

from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from PIL import Image
import cv2

img_size = 224
transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize images to fit the input shape of the model
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])
local_bs = 128
bs = 256
num_users = 12


class CustomImageNet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')

        '''if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))'''
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def get_ImageNetdataloaders(train_data_path, test_data_path):
    #load each client data
    train_image_data = []
    train_label_data = []
    lablename_list = os.listdir(train_data_path)
    cla_dict = np.load("./dataset/imagenet2012/label_dict.npy",allow_pickle=True).item()
    #print(lablename_list[0])
    for i in range(len(lablename_list)):
        file_names = [os.path.join(os.path.join(train_data_path, lablename_list[i]),file) 
                      for file in os.listdir(os.path.join(train_data_path, lablename_list[i]))]
        train_image_data += file_names
        for j in range(len(file_names)):
            train_label_data.append(cla_dict[lablename_list[i]])
    
    client_train_loaders = DataLoader(CustomImageNet(train_image_data, train_label_data, transform), batch_size=local_bs, shuffle=True)
    
    test_image_data = []
    test_label_data = []
    lablename_list = os.listdir(test_data_path)
    #print(lablename_list[0])
    for i in range(len(lablename_list)):
        file_names = [os.path.join(os.path.join(test_data_path, lablename_list[i]),file) 
                      for file in os.listdir(os.path.join(test_data_path, lablename_list[i]))]
        test_image_data += file_names
        for j in range(len(file_names)):
            test_label_data.append(cla_dict[lablename_list[i]])
    client_test_loader = DataLoader(CustomImageNet(test_image_data, test_label_data, transform), batch_size=bs, shuffle=True)
    return client_train_loaders, client_test_loader, img_size

def get_serverdataloader(server_test_path):
    test_image_data = []
    test_label_data = []
    lablename_list = os.listdir(server_test_path)
    cla_dict = np.load("./dataset/imagenet2012/label_dict.npy").item()
    #print(lablename_list[0])
    for i in range(len(lablename_list)):
        file_names = [os.path.join(os.path.join(server_test_path, lablename_list[i]),file) 
                      for file in os.listdir(os.path.join(server_test_path, lablename_list[i]))]
        test_image_data += file_names
        for j in range(len(file_names)):
            test_label_data.append(cla_dict[lablename_list[i]])
    server_test_loader = DataLoader(CustomImageNet(test_image_data, test_label_data, transform), batch_size=bs, shuffle=True)
    return server_test_loader, img_size


train_data_path = "./dataset/imagenet2012/fed_data/client_12non-iid_alpha1.5/client_0/train"
test_data_path = "./dataset/imagenet2012/fed_data/client_12non-iid_alpha1.5/client_0/test"  
client_train_loaders, client_test_loader, img_size = get_ImageNetdataloaders(train_data_path, test_data_path)
print("Load Client DataLoader Done!!!")
for batch_idx, (image, label) in enumerate(client_train_loaders):
    print(image.shape)
    print(image)
    
server_test_path = "./dataset/imagenet2012/val"
server_test_loader, img_size = get_ImageNetdataloaders(server_test_path)
print("Load Server DataLoader Done!!!")
