import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import math
from model import FedModel
#from .drive import drive_compress, drive_decompress, drive_plus_compress, drive_plus_decompress
#from .eden.eden import eden_builder
import numpy as np
from typing import List
from scipy.stats import norm
import copy
import collections
from pathlib import Path
import timm
from quantization.lsq_layer import QuantAct, QuantConv2d, QuantLinear, QuantMultiHeadAct, QuantMuitiHeadLinear, QuantMuitiHeadLinear_in
import collections

@torch.no_grad()
def initialize_quantization(data_loader, model, device, sample_iters=5):
    for name, m in model.named_modules():
        if (isinstance(m, QuantLinear) or isinstance(m, QuantConv2d) or isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in)) and m.alpha is not None:
            print(f"initialize the weight scale for module {name}")
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
        self.round_global_weights = []
        self.round_global_weights.append(self.global_model.get_weights())
        self.current_sample_clients = []   #当前轮客户端的列�?
        self.current_training_clients = []  #目前还在训练的客户端列表
        self.current_trained_clients = []   #目前已经训练完的客户�?
        self.current_aggregate_clients = []  #当前轮要聚合的客户端列表
        self.current_idle_clients = []
        self.stale_steps = 0
        self.update_number = 0
        self.stale_ths = 10
        self.n_round = 0
    
    def set_current_round(self, n_round):
        self.n_round = n_round
    

    def sample_clients(self):
        """
            Sample clients at each round (now uniformly at random, can be changed).
        """
        #从上一轮中被聚合的客户端以及上一轮闲散的客户端中选择
        n_samples = math.floor(self.frac * self.num_users)
        tmp = self.current_idle_clients
        self.current_sample_clients = list(np.random.choice(tmp,size=min(len(tmp),n_samples), replace=False))
        self.current_idle_clients = list(set(tmp) - set(self.current_sample_clients))
        return self.current_sample_clients
    
    
    
    #在每一次聚合之后进行模型权重（没有掩码）的保存
    def save_global_weight(self):
        param_dict = {k: v for k, v in self.global_model.get_weights().items()}
        self.round_global_weights.append(param_dict)
        #torch.save({'weight_model_state_dict': param_dict}, self.server_model_path)   #存储最新的模型参数

    def find_bitrate(self, probs, num_params):
        local_bitrate = 0
        for p in probs:
            local_bitrate += p * math.log2(1 / p)
        return local_bitrate * num_params

    
    def set_init_training_mode(self, current_epoch, mask_convergence):
        self.training_mode = "mask" if current_epoch > self.args.local_ep/2 or mask_convergence == True else "weight"
    
 

    def aggregate_gradients(self, client_gradients, recevie_round, n_round, model_rate):
        aggregated_weights = copy.deepcopy(self.global_model.get_weights())


        #print(client_gradients_ls[0])
        client_staleness_ls = self.compute_staleness(self.current_aggregate_clients, client_gradients, recevie_round, model_rate, n_round)
        for i, client_name in enumerate(client_gradients.keys()):
            for key, value in client_gradients[client_name].items():
                if "to_qkv.bias" not in key:   #weight模型中设置了to_qkv的bias为false，但是mask中无法进行设�?
                    aggregated_weights[key] += client_staleness_ls[i] * value
        
        self.global_model.set_weights(aggregated_weights)
        self.save_global_weight()


    def broadcast_model(self, sampled_clients, n_round):
        """
            Send the global updated model to the clients. 
        """
        for client in sampled_clients:
            self.clients_list[client].set_local_training_mode(self.training_mode)
            #self.clients_list[client].set_local_weights(self.all_client_model_list[client].get_weights())
            self.clients_list[client].set_local_weights(self.global_model.get_weights())
            self.clients_list[client].set_local_round(n_round)
    def sample_clients(self, n_samples, t=None):
        return np.random.choice(np.arange(self.n_clients),
                                size=n_samples,
                                replace=False)


    def set_client_list(self, clients_list: List[Client]):
        self.clients_list = clients_list

    def aggregate_weights(self, client_idxs, n_roud):
        return self.sample_weights_aggregation(client_idxs)
    
    def sample_weights_aggregation(self, client_idxs):
        aggregated_weights = collections.OrderedDict()
        
        client_gradients_ls = []
        for client_id in client_idxs:
            client_gradients = self.clients_list[client_id].upload_weight()
            client_gradients_ls.append(client_gradients)
        
        #aggregated_weights = client_gradients_ls[0]
        for i in range(len(client_gradients_ls)):
            for key,value in client_gradients_ls[i].items():
                if key not in aggregated_weights.keys():
                    aggregated_weights[key] = (value/len(client_gradients_ls))
                else:
                    aggregated_weights[key] += (value/len(client_gradients_ls))
        
        self.global_model.set_weights(aggregated_weights)
    
    def broadcast_model(self, sampled_clients):
        for client in sampled_clients:
            self.clients_list[client].set_local_weights(self.global_model.get_weights())

