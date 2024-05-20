# -*- coding: utf-8 -*-
import sys
import os
sys.path.append("..")
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import bernoulli
from model import FedModel
#from .models.pruned_layers import *

import copy
from collections import OrderedDict
from pathlib import Path
import math
import time
#from torchstat import stat

class Client:
    """
        Class implementing local clients, with local datasets.
    """

    def __init__(self, drop, img_size, num_classes, embed_dim, mlp_dim, transformer_depth, transformer_head, 
                 lambda1, temperature, 
                 weight_decay, no_prox, lr, 
                 train_dataset, test_dataset, self_distillation, ratio, mask_round_ratio, device):
        
        #self.client_id = id   
        '''current_folder = Path(__file__).parent.resolve()
        parent_folder = Path(__file__).parent.parent.resolve()'''
        #self.client_mask_path = "/home/liyan/fed_checkpoint/tmp1/client_mask/"+str(self.client_id)+"_local_mask_and_classifier.pth"
        self.local_mask_and_classifier = None
        self.drop = drop
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.transformer_depth = transformer_depth
        self.transformer_head = transformer_head
        self.lambda1 = lambda1
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.no_prox = no_prox
        self.lr = lr
        self.num_classes = num_classes
        self.self_distillation = self_distillation
        self.local_train_dataloader = train_dataset
        self.local_test_dataloader = test_dataset
        self.n = len(self.local_train_dataloader)
        # Local model at the client
        self.device = device
        self.ratio = ratio
        self.mask_indices = {}
        #self.initial_weights = copy.deepcopy(initial_weights)
        self.training_round = 0
        self.training_mode = "mask"  #mask, weight, both
        self.local_model = FedModel(drop=self.drop, img_size=self.img_size, num_classes=self.num_classes,
                                    embed_dim=self.embed_dim, transformer_depth=self.transformer_depth, transformer_head=self.transformer_head, mlp_dim=self.mlp_dim, 
                                    lambda1=self.lambda1, temperature=self.temperature,
                                    training_mode="weight", self_distillation=self.self_distillation, 
                                    weight_decay=self.weight_decay, no_prox=self.no_prox, lr=self.lr, 
                                    model_rate=self.ratio, device=self.device)
        
        #self.set_local_weights(initial_weights)
        self.model_size = self.local_model.model_size  # bit
        self.mask_round_ratio = mask_round_ratio
        self.weight_mask = None

    def set_local_weights(self, w):
        #print(f'Model: {type(self.local_model.model)}.')
        '''self.local_model = FedModel(args=self.args, ratio=self.ratio, img_size=self.img_size, 
                                    training_mode=self.training_mode, device=self.device)'''
        self.initial_weights = copy.deepcopy(w)
        #self.local_model.set_weights(w)
    
    def get_local_weights(self):
        return self.local_model.get_weights()

    def set_local_round(self, n_round):
        self.training_round = n_round
    

    def upload_weight(self):
        with torch.no_grad():
            local_model_dict = self.local_model.get_weights()
        return local_model_dict
    
    def train_local(self, n_round):
        loss = self.local_model.train_weight(data_loader=self.local_data_loader)

    
    def test_local_model(self, n_samples=1):
        """
            Test the global model on test dataset, n_samples to be used only with mask training.
        """
        total = 0
        correct = 0
        loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (test_x, test_y) in enumerate(self.local_test_loader):  
                temp_l = []
                for n in range(n_samples):
                    test_x = test_x.to(self.device)
                    test_y = test_y.to(self.device)

                    outputs = self.model(test_x)
                    _, pred_y = torch.max(outputs[-1].data, 1)
                    temp_l.append(pred_y.cpu().numpy())
                axis = 0
                temp_l = np.asarray(temp_l)
                u, indices = np.unique(temp_l, return_inverse=True)
                maj_pred = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(temp_l.shape),
                                                           None, np.max(indices) + 1), axis=axis)]
                total += test_y.size(0)
                maj_pred = np.asarray(maj_pred)
                correct += (maj_pred == test_y.cpu().numpy()).sum().item()
                test_y = test_y.to(torch.int64) 
                loss += criterion(outputs[-1], test_y)

        return (loss / total, correct / total)
    
