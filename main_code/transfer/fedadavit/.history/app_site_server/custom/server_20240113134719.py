import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import math
from model import FedModel
import numpy as np
from typing import List
from scipy.stats import norm
import copy
import collections
from pathlib import Path
import timm
import collections


class Server:
    """
        Class for the central authority, i.e., the server, that coordinates the federated process.
    """

    def __init__(self, test_data, drop, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim, img_size, lambda1, temperature, model_rate=1, device=None,model_dict=None):
        
        self.device = device
        self.img_size = img_size
        self.training_mode = "mask" 
        self.global_model = FedModel(drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, 
                                     lambda1, temperature, model_rate=1, device=device)
        self.global_model.set_weights(model_dict)
        self.round_global_weights = []
        self.round_global_weights.append(self.global_model.get_weights())
        self.current_sample_clients = []   
        self.current_training_clients = [] 
        self.current_trained_clients = []  
        self.current_aggregate_clients = []  
        self.current_idle_clients = []
        self.stale_steps = 0
        self.update_number = 0
        self.stale_ths = 10
        self.n_round = 0
    
    def set_current_round(self, n_round):
        self.n_round = n_round
    
    
    def save_global_weight(self):
        param_dict = {k: v for k, v in self.global_model.get_weights().items()}
        self.round_global_weights.append(param_dict)
        #torch.save({'weight_model_state_dict': param_dict}, self.server_model_path)   


    def aggregate_weights(self, client_weights, n_roud):
        return self.sample_weights_aggregation(client_weights)
    
    def sample_weights_aggregation(self, client_weights):
        aggregated_weights = copy.deepcopy(self.global_model.model.state_dict())
        
        for k,v in aggregated_weights.items():
            aggregated_weights[k] = torch.zeros_like(v,device=v.device)

        for key in aggregated_weights.keys():
            cnt = 0
            for client in client_weights.keys():
                if key in client_weights[client].keys():
                    aggregated_weights[key] += client_weights[client][key]
                    cnt += 1
            aggregated_weights[key] = aggregated_weights[key]/cnt
                
        '''#aggregated_weights = client_gradients_ls[0]
        for client in client_weights.keys():
            for key,value in client_weights[client].items():
                if key not in aggregated_weights.keys():
                    aggregated_weights[key] = (value/len(client_weights))
                else:
                    aggregated_weights[key] += (value/len(client_weights))'''
        
        self.global_model.set_weights(aggregated_weights)
    
    