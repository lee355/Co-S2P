import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from torchvision import datasets, transforms
import torch

from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from PIL import Image

img_size = 224
transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize images to fit the input shape of the model
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

try:
    import jtop
    
    from utils import get_data_path,get_local_batch_size,get_local_model_rate
    local_bs = get_local_batch_size()
    bs = local_bs
    model_rate = get_local_model_rate()

except ImportError:
    local_bs = 256
    bs = 256
    model_rate = 0.75
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
        
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def get_ImageNetdataloaders(train_data_path, test_data_path):
    #load each client data
    train_image_data = []
    train_label_data = []
    lablename_list = os.listdir(train_data_path)
    if os.path.exists('/home/nvidia'):
        cla_dict = np.load("/home/xugw/code/src/code and baselines/dataset/imagenet2012/imagenet300/label_dict.npy",allow_pickle=True).item()
    else:
        cla_dict = np.load("/home/xugw/code/src/code and baselines/dataset/imagenet2012/imagenet300/label_dict.npy",allow_pickle=True).item()
    #print(lablename_list[0])
    for i in range(len(lablename_list)):
        file_names = [os.path.join(os.path.join(train_data_path, lablename_list[i]),file) 
                      for file in os.listdir(os.path.join(train_data_path, lablename_list[i]))]
        train_image_data += file_names
        for j in range(len(file_names)):
            train_label_data.append(cla_dict[lablename_list[i]])
    
    client_train_loaders = DataLoader(CustomImageNet(train_image_data, train_label_data, transform), batch_size=local_bs, shuffle=True,num_workers=8)
    
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
    client_test_loader = DataLoader(CustomImageNet(test_image_data, test_label_data, transform), batch_size=bs, shuffle=True,num_workers=8)
    return client_train_loaders, client_test_loader, img_size

def get_serverdataloader(server_test_path):
    test_image_data = []
    test_label_data = []
    lablename_list = os.listdir(server_test_path)
    if os.path.exists('/home/nvidia'):
        cla_dict = np.load("/home/liyan/code/src/code and baselines/dataset/imagenet2012/imagenet300/label_dict.npy",allow_pickle=True).item()
    else:
        cla_dict = np.load("/home/liyan/code/src/code and baselines/dataset/imagenet2012/imagenet300/label_dict.npy",allow_pickle=True).item()
    #print(lablename_list[0])
    for i in range(len(lablename_list)):
        file_names = [os.path.join(os.path.join(server_test_path, lablename_list[i]),file) 
                      for file in os.listdir(os.path.join(server_test_path, lablename_list[i]))]
        test_image_data += file_names
        for j in range(len(file_names)):
            test_label_data.append(cla_dict[lablename_list[i]])
    server_test_loader = DataLoader(CustomImageNet(test_image_data, test_label_data, transform), batch_size=bs, shuffle=True,num_workers=8)
    return server_test_loader, img_size

def get_local_model_rate():
    return model_rate