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
    

    def sample_clients(self):
        """
            Sample clients at each round (now uniformly at random, can be changed).
        """
        n_samples = math.floor(self.frac * self.num_users)
        tmp = self.current_idle_clients
        self.current_sample_clients = list(np.random.choice(tmp,size=min(len(tmp),n_samples), replace=False))
        self.current_idle_clients = list(set(tmp) - set(self.current_sample_clients))
        return self.current_sample_clients
    
    def set_current_training_clients(self):
        self.current_training_clients += self.current_sample_clients

    def set_asynchronous_aggregation_clients(self):
        total_client_num = len(self.current_training_clients)
        low_client_num = math.floor(total_client_num * 0.5)
        aggregate_client_num = np.random.randint(low_client_num, total_client_num)
        '''self.current_aggregate_clients = list(np.random.choice(self.current_training_clients, 
                                                               size=aggregate_client_num, replace=False))'''
        self.current_aggregate_clients = self.current_sample_clients  
        return self.current_aggregate_clients
    
    def set_continue_training_clients(self):
        self.current_training_clients = list(set(self.current_training_clients) - set(self.current_aggregate_clients))

    def set_idle_clients(self):
        self.current_idle_clients += self.current_aggregate_clients  
    
    def save_global_weight(self):
        param_dict = {k: v for k, v in self.global_model.get_weights().items()}
        self.round_global_weights.append(param_dict)
        #torch.save({'weight_model_state_dict': param_dict}, self.server_model_path)   

    def find_bitrate(self, probs, num_params):
        local_bitrate = 0
        for p in probs:
            local_bitrate += p * math.log2(1 / p)
        return local_bitrate * num_params

    
    def set_init_training_mode(self, current_epoch, mask_convergence):
        self.training_mode = "mask" if current_epoch > self.args.local_ep/2 or mask_convergence == True else "weight"
    
    
    def compute_staleness(self, client_idxs, client_gradients, recevie_round, client_model_rate, n_round):
        '''
        
        '''
        client_gamma_ls = []
        assert len(client_gradients) == len(recevie_round)

        for i, (client_name, client_receive_round) in enumerate(recevie_round.items()):
            latest_weights = self.round_global_weights[n_round]
            client_weights = self.round_global_weights[client_receive_round]
            
            gradients_importance = sum(torch.sum(torch.abs(v)) 
                                       for v in client_gradients[client_name].values()) / client_model_rate[client_name]
            weights_differnece = sum(torch.sum(torch.abs(latest_weights[k] - client_weights[k])) 
                                     for k in client_weights.keys())
            tmp_gamma = gradients_importance / (weights_differnece + 1)
            client_gamma_ls.append(tmp_gamma)
        
        tensor = torch.tensor(client_gamma_ls)
        norm_result = tensor / torch.norm(tensor, p=1)
        client_staleness_ls = norm_result.to(self.device)
        print(client_staleness_ls)
        return client_staleness_ls

    def compute_n_mask_samples(self, n_round): 
        """
            Return how many per-client samples, i.e., bits, the server wants to receive
            (can be a function of the round number). Now set to 1.
        """
        return 1

    def find_bitrate(self, probs, num_params):
        local_bitrate = 0
        for p in probs:
            local_bitrate += p * math.log2(1 / p)
        return local_bitrate * num_params
    
    def aggregate_gradients(self, client_gradients, recevie_round, n_round, model_rate):
        aggregated_weights = copy.deepcopy(self.global_model.get_weights())


        #print(client_gradients_ls[0])
        client_staleness_ls = self.compute_staleness(self.current_aggregate_clients, client_gradients, recevie_round, model_rate, n_round)
        for i, client_name in enumerate(client_gradients.keys()):
            for key, value in client_gradients[client_name].items():
                if "to_qkv.bias" not in key:   #weight模型中设置了to_qkv的bias为false，但是mask中无法进行设置
                    aggregated_weights[key] += client_staleness_ls[i] * value
        
        self.global_model.set_weights(aggregated_weights)
        self.save_global_weight()

    def aggregate_gradients(self, sampled_mask, local_freq, num_params, n_roud):
        return self.sample_mask_aggregation(sampled_mask, local_freq, num_params, n_roud)

    def sample_mask_aggregation(self, sampled_mask, local_freq, num_params, n_roud):
        """
            Aggregation method for the federated sampling subnetworks scheme ("Bayesian Aggregation").
        """

        aggregated_weights = copy.deepcopy(self.global_model.get_weights())
        aggregated_p = dict()

        for k, v in self.global_model.model.named_parameters():
            if 'mask' in k:
                aggregated_p[k] = torch.zeros_like(v)

        with torch.no_grad():
            # Reset aggregation priors.
            # self.update_prior(n_roud)
            self.reset_prior()
            n_samples = self.compute_n_mask_samples(n_roud)
            p_update = []
            avg_bitrate = 0
            avg_freq = 0
            for client in sampled_mask.keys():
                sampled_mask, local_freq, num_params = sampled_mask, local_freq, num_params  #TODO
                avg_freq += local_freq
                local_bitrate = self.find_bitrate([local_freq + 1e-50, 1 - local_freq + 1e-50], num_params) + math.log2(num_params)
                avg_bitrate += local_bitrate / num_params
                for k, v in sampled_mask.items():
                    if 'mask' in k:
                        self.alphas[k] += v
                        self.betas[k] += (n_samples-v)
                        # Add layerwise estimated ps for each client
                        p_update.extend(v.cpu().numpy().flatten()/n_samples)
            avg_bitrate = avg_bitrate / len(sampled_mask)
            avg_freq = avg_freq / len(sampled_mask)
            # Update the posterior, and compute the mode of the beta distribution, as suggested in
            # https://neurips2021workshopfl.github.io/NFFL-2021/papers/2021/Ferreira2021.pdf
            for k, val in aggregated_weights.items():
                if 'mask' in k:
                    avg_p = (self.alphas[k] - 1) / (self.alphas[k] + self.betas[k] - 2)
                    if self.params.get('model').get('optimizer').get('noisy'):
                        avg_p = self.correct_bias(avg_p)
                    aggregated_weights[k] = torch.tensor(
                        torch.log(avg_p / (1 - avg_p)),
                        requires_grad=True,
                        device=self.device)
        self.global_model.set_weights(aggregated_weights)
        return np.mean(p_update), avg_bitrate, avg_freq

    def update_prior(self, n_round):
        """
            Compute when resetting the prior depending on the round number.
        """
        if n_round < 15:
            self.reset_prior()
        elif n_round % 5 == 0:
            self.reset_prior()

    def reset_prior(self):
        """
            Reset to uniform prior, depending on lambda_init.
        """
        self.alphas = dict()
        self.betas = dict()
        for k, val in self.global_model.model.named_parameters():
            self.alphas[k] = torch.ones_like(val) * self.lambda_init
            self.betas[k] = torch.ones_like(val) * self.lambda_init