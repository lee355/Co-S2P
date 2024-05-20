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
    
    def set_local_training_mode(self, training_mode):
        self.training_mode = training_mode


    def train_local(self, n_round, initial_weights, local_ep):
        self.set_local_round(n_round)
        self.initial_weights = initial_weights
        if n_round <= 100:
            #set mask rounds and weight rounds
            mask_rounds = math.floor(self.mask_round_ratio * local_ep)
            #read latest round mask and classifier
            self.local_model = FedModel(drop=self.drop, img_size=self.img_size, num_classes=self.num_classes,
                                    embed_dim=self.embed_dim, transformer_depth=self.transformer_depth, transformer_head=self.transformer_head, mlp_dim=self.mlp_dim, 
                                    lambda1=self.lambda1, temperature=self.temperature,
                                    training_mode=self.training_mode, self_distillation=False, 
                                    weight_decay=self.weight_decay, no_prox=self.no_prox, lr=self.lr, 
                                    model_rate=self.ratio, device=self.device)
            self.local_model.model.load_state_dict(initial_weights, strict=False)
            if n_round != 0:
                self.load_local_mask_and_classifier()
 
            loss_mask = self.local_model.train_mask(data_loader=self.local_train_dataloader, local_sub_epoch=mask_rounds)
            self.save_local_mask_and_classifier()
            
            mask_model_weights = self.get_local_weights()
            self.mask_indices = {}
            self.single_mask_indices = []
            last_mask_indices = None
            transformer_heads_ls = []
            for k, v in mask_model_weights.items():
                if "mask" in k:
                    if "qkv" in k:
                        while True:
                            theta = torch.sigmoid(v)
                            updates_s = torch.bernoulli(theta)
                            if torch.sum(updates_s) != 0:
                                break
                        assert k[:-5] in mask_model_weights.keys()
                        
                        updates_s = updates_s.unsqueeze(1).repeat(1, int(self.embed_dim/self.transformer_head)).view(-1).repeat(3)
                        self.mask_indices[k[:-5]] = updates_s
                        self.single_mask_indices.append(updates_s)
                        mask_model_weights[k[:-5]] = mask_model_weights[k[:-5]][updates_s.bool()]
                        last_mask_indices = copy.deepcopy(updates_s)
                        transformer_heads_ls.append(int(torch.sum(updates_s).item()))
                        continue
                    while True:
                        theta = torch.sigmoid(v)
                        updates_s = torch.bernoulli(theta)
                        if torch.sum(updates_s) != 0:
                            break
                    assert k[:-5] in mask_model_weights.keys()
                    assert k[:-11]+"bias" in mask_model_weights.keys()
                    
                    self.mask_indices[k[:-5]] = updates_s
                    self.mask_indices[k[:-11]+"bias"] = updates_s
                    self.single_mask_indices.append(updates_s)
                    
                    mask_model_weights[k[:-5]] = mask_model_weights[k[:-5]][updates_s.bool()]
                    mask_model_weights[k[:-11]+"bias"] = mask_model_weights[k[:-11]+"bias"][updates_s.bool()]

                    
                    if last_mask_indices is not None: 
                        if "linear2" in k:
                            mask_model_weights[k[:-5]] = mask_model_weights[k[:-5]][:, last_mask_indices.nonzero().squeeze()]
                        if "to_out" in k:
                            mask_model_weights[k[:-5]] = mask_model_weights[k[:-5]][:, last_mask_indices[:int(last_mask_indices.size(0)/3)].nonzero().squeeze()]
                    last_mask_indices = copy.deepcopy(updates_s)

            self.local_model = FedModel(drop=self.drop, img_size=self.img_size, num_classes=self.num_classes, segment_mask=self.single_mask_indices,
                                    embed_dim=self.embed_dim, transformer_depth=self.transformer_depth, transformer_head=transformer_heads_ls, mlp_dim=self.mlp_dim, 
                                    lambda1=self.lambda1, temperature=self.temperature,
                                    training_mode="weight", self_distillation=self.self_distillation, 
                                    weight_decay=self.weight_decay, no_prox=self.no_prox, lr=self.lr, 
                                    model_rate=self.ratio, device=self.device)

            #start train weight model
            self.local_model.model.load_state_dict(mask_model_weights, strict=False)
            loss_weight = self.local_model.train_weights(data_loader=self.local_train_dataloader, 
                                                        local_sub_epoch=local_ep-mask_rounds)
        else:
            print(type(self.local_model.model))
            self.load_local_mask_and_classifier()  
            weight_model_weights = copy.deepcopy(self.initial_weights)
            last_mask_indices = None
            for k, v in self.mask_indices.items():
                #weight_model_weights[k] = weight_model_weights[k] * v
                if "weight" in k:
                    weight_model_weights[k] = weight_model_weights[k][v.bool()]
                    
                    if last_mask_indices is not None:  
                        if "linear2" in k:
                            weight_model_weights[k] = weight_model_weights[k][:, last_mask_indices.nonzero().squeeze()]
                        elif "to_out" in k:
                            weight_model_weights[k] = weight_model_weights[k][:, last_mask_indices[:int(last_mask_indices.size(0)/3)].nonzero().squeeze()]
                    last_mask_indices = copy.deepcopy(v)
                else:
                    weight_model_weights[k] = weight_model_weights[k][v.bool()]
            #start train weight model
            self.local_model.model.load_state_dict(weight_model_weights, strict=False)
            loss_weight = self.local_model.train_weights(data_loader=self.local_train_dataloader, 
                                                        local_sub_epoch=self.args.local_ep)


    def save_local_mask_and_classifier(self):
        local_model_state_dict = self.get_local_weights()
        selected_mask_layers = OrderedDict()
        for k, v in local_model_state_dict.items():
            if 'mask' in k or 'exist_classifiers' in k:
                '''if 'mask' in k:
                    self.weight_mask[k[:-5]] = torch.bernoulli(torch.sigmoid(v))'''
                selected_mask_layers[k] = v #bernoulli.rvs(torch.sigmoid(v).cpu().numpy())
        self.local_mask_and_classifier = selected_mask_layers

    def load_local_mask_and_classifier(self):
        self.local_model.set_weights(self.local_mask_and_classifier)

    
    
    def upload_gradients(self):
        param_dict = dict()
        #print(f'Current lcoal Model: {type(self.local_model.model)}.')
        with torch.no_grad():
            local_model_dict = self.local_model.get_weights()
            param_dict = {k: copy.deepcopy(v) for k, v in local_model_dict.items() 
                            if 'exist_classifiers' not in k or ('exist_classifiers' in k and 'exist_classifiers'+str(self.ratio*4) in k) }
            last_mask_indices = None
            for k, v in param_dict.items():
                if k in self.mask_indices.keys():
                    if 'weight' in k:
                        '''print(k)
                        print(self.mask_indices[k].size(0))
                        print(v.size())'''
                        extended_tensor = torch.zeros((self.mask_indices[k].size(0), v.size(1)), dtype=v.dtype).to(v.device)
                        '''print(extended_tensor.size())
                        print(self.initial_weights[k].size())'''
                        extended_tensor[self.mask_indices[k].nonzero().squeeze()] = v
                        if last_mask_indices is not None:   
                            if "linear2" in k:
                                tmp = extended_tensor
                                extended_tensor = torch.zeros((extended_tensor.size(0), last_mask_indices.size(0)), dtype=v.dtype).to(v.device)
                                extended_tensor[:, last_mask_indices.nonzero().squeeze()] = tmp
                            elif "to_out" in k:
                                tmp = extended_tensor
                                extended_tensor = torch.zeros((extended_tensor.size(0), last_mask_indices[:int(last_mask_indices.size(0)/3)].size(0)), dtype=v.dtype).to(v.device)
                                extended_tensor[:, last_mask_indices[:int(last_mask_indices.size(0)/3)].nonzero().squeeze()] = tmp
                        param_dict[k] = ((extended_tensor - self.initial_weights[k]) * self.mask_indices[k].reshape((-1,1))).to(v.device)
                        last_mask_indices = self.mask_indices[k]
                    elif 'bias' in k:
                        extended_tensor = torch.zeros((self.mask_indices[k].size(0)), dtype=v.dtype).to(v.device)
                        extended_tensor[self.mask_indices[k].nonzero().squeeze()] = v
                        param_dict[k] = ((extended_tensor - self.initial_weights[k]) * self.mask_indices[k]).to(v.device)
                else:
                    param_dict[k] = (param_dict[k] - self.initial_weights[k]).to(v.device)
        return param_dict, self.training_round

    
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
    
