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
from lsq_layer import QuantAct, QuantConv2d, QuantLinear, QuantMultiHeadAct, QuantMuitiHeadLinear, QuantMuitiHeadLinear_in
import collections

@torch.no_grad()
def initialize_quantization(data_loader, model, device, sample_iters=5):
    for name, m in model.named_modules():
        if (isinstance(m, QuantLinear) or isinstance(m, QuantConv2d) or isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in)) and m.alpha is not None:
            #print(f"initialize the weight scale for module {name}")
            m.initialize_scale(device)

    # switch to evaluation mode
    model.eval()
    n = 0
    for batch_idx, (images, target) in enumerate(data_loader):
        n += 1
        if n > sample_iters:
            break
        images = images.to(device)
        output = model(images)
    for name, m in model.named_modules():
        if (isinstance(m, QuantAct) or isinstance(m, QuantMultiHeadAct)) and m.alpha is not None:
            print(f"initialize the activation scale for module {name}")
            m.initialize_scale_offset(device)
    return

class Server:
    """
        Class for the central authority, i.e., the server, that coordinates the federated process.
    """

    def __init__(self, test_data, drop, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim, img_size, lambda1, temperature, model_rate=1, device=None):
        
        self.device = device
        self.img_size = img_size
        self.training_mode = "mask" 
        self.global_model = FedModel(drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, 
                                     lambda1, temperature, model_rate=1, device=device) 
        initialize_quantization(test_data, self.global_model.model, self.device)
        print('---------------------------------------------------')
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
        aggregated_weights = collections.OrderedDict()
        
        #aggregated_weights = client_gradients_ls[0]
        for client in client_weights.keys():
            for key,value in client_weights[client].items():
                if key not in aggregated_weights.keys():
                    aggregated_weights[key] = (value/len(client_weights))
                else:
                    aggregated_weights[key] += (value/len(client_weights))
        
        self.global_model.set_weights(aggregated_weights)
    
    