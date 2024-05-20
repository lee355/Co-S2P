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
from client import Client
from scipy.stats import norm
import copy
import collections

import timm

class Server:
    """
        Class for the central authority, i.e., the server, that coordinates the federated process.
    """

    def __init__(self, test_data ,drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, model_rate=1, device=None):

        self.device = device
        self.img_size = img_size
        self.global_model = FedModel(drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, model_rate=1, device=device) 
        self.stale_steps = 0
        self.update_number = 0
        self.stale_ths = 10
        self.current_sample_clients = None
        self.param_idx = None
        self.rounds = 0
        self.transformer_head=transformer_head

    def set_rounds(self, rounds):
        self.rounds = rounds - 1
    
    def sample_clients(self, n_samples, t=None):
        return np.random.choice(np.arange(self.n_clients),
                                size=n_samples,
                                replace=False)


    def set_client_list(self, clients_list: List[Client]):
        self.clients_list = clients_list

    def aggregate_weights(self, local_parameters,param_idx , n_roud):
        self.param_idx=param_idx
        return self.combine(local_parameters, self.param_idx)
    
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
    

    '''def broadcast(self):
        self.global_model.model.train(True)
        local_parameters, self.param_idx = self.distribute(self.current_sample_clients)
        for i in range(len(self.current_sample_clients)):
            self.clients_list[self.current_sample_clients[i]].set_local_weights(self.model_rate[i], local_parameters[i])'''

    """ def make_model_rate(self):
        self.model_rate = np.array(self.rate)
        for i in range(len(self.current_sample_clients)):
            self.clients_list[self.current_sample_clients[i]].set_local_rate(self.model_rate[i]) """

    def split_model(self, client_modelrate):
        idx_i = { k: None for k in client_modelrate.keys()}
        idx = {k:collections.OrderedDict() for k in client_modelrate.keys()}
        for k, v in self.global_model.model.state_dict().items():
            parameter_type = k.split('.')[-1]
            for m in client_modelrate.keys():
                scaler_rate = client_modelrate[m]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'embedding' in k.split('.')[-2]:
                                '''print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')'''
                                input_idx_i_m = torch.arange(input_size, device=v.device)
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                '''print(local_output_size)'''
                                roll = int((self.rounds* local_output_size / 16)) % output_size
                                ridx = torch.arange(output_size, device=v.device)
                                ridx = torch.roll(ridx, roll, -1)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'decoder' in k and 'linear2' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                                input_idx_i_m = idx_i[m]
                                scaler_rate = client_modelrate[m]
                                local_output_size = int(np.ceil(output_size // self.transformer_head
                                                                * scaler_rate))
                                roll = int((self.rounds* local_output_size / 16)) % output_size
                                ridx = torch.arange(output_size, device=v.device)
                                ridx = torch.roll(ridx, roll, -1)
                                output_idx_i_m = (ridx.reshape(
                                    self.transformer_head, -1))[:, :local_output_size].reshape(-1)
                                idx_i[m] = output_idx_i_m
                            elif 'exist_linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size,device=v.device)
                                idx_i[m] = output_idx_i_m
                            else:
                                
                                input_idx_i_m = idx_i[m]
                                scaler_rate = client_modelrate[m]
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                roll = int((self.rounds* local_output_size / 16)) % output_size
                                ridx = torch.arange(output_size, device=v.device)
                                ridx = torch.roll(ridx, roll, -1)
                                output_idx_i_m = ridx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'decoder' in k and 'linear2' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        elif 'linear_q' in k or 'linear_k' in k or 'linear_v' in k:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                            '''if 'linear_v' not in k:
                                idx_i[m] = idx[m][k.replace('bias', 'weight')][1]'''
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    input_size = v.size(2)
                    output_size = v.size(1)
                    local_input_size = int(np.ceil(input_size * scaler_rate))
                    roll = int((self.rounds* local_input_size / 16)) % output_size
                    ridx = torch.arange(input_size, device=v.device)
                    ridx = torch.roll(ridx, roll, -1)
                    output_idx_i_m = torch.arange(output_size, device=v.device)
                    input_idx_i_m = ridx[:local_input_size]
                    the_first_d=torch.arange(1,device=v.device)
                    idx[m][k] = (the_first_d,output_idx_i_m, input_idx_i_m)
        return idx
    
    def distribute(self, client_modelrate):
        param_idx = self.split_model(client_modelrate)
        local_parameters = {k:collections.OrderedDict() for k in client_modelrate.keys()}
        for k, v in self.global_model.model.state_dict().items():
            parameter_type = k.split('.')[-1]
            for m in client_modelrate.keys():
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            '''print("client ", user_idx[m])
                            print(k)
                            print(v)
                            print(param_idx[m][k])'''
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])]).cpu().numpy()
                        else:
                            '''print("client ", user_idx[m])
                            print(k)
                            print(v)
                            print(param_idx[m][k])'''
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]]).cpu().numpy()
                    else:
                        '''print("client ", user_idx[m])
                        print(k)
                        print(v)
                        print(param_idx[m][k])'''
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]]).cpu().numpy()
                else:
                    '''print("client ", user_idx[m])
                    print(k)
                    print(v)'''
                    local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])]).cpu().numpy()
        return local_parameters, param_idx

    def combine(self, local_parameters, param_idx):
        count = collections.OrderedDict()
        self.global_parameters = self.global_model.model.state_dict()
        local_device = local_parameters[list(local_parameters.keys())[0]][list(self.global_parameters.keys())[0]].device
        updated_parameters = copy.deepcopy(self.global_parameters)
        for k,v in updated_parameters.items():
            updated_parameters[k] = v.to(local_device)
        for k, v in updated_parameters.items():
            #tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            parameter_type = k.split('.')[-1]
            count[k] = v.new_zeros(v.size(), dtype=torch.float32,device=local_parameters[list(local_parameters.keys())[0]][k].device)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32,device=local_parameters[list(local_parameters.keys())[0]][k].device)
            for m in local_parameters.keys():
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:                                                     
                            if k.split('.')[-2] == 'embedding':
                                '''label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = list(param_idx[m][k])
                                param_idx[m][k][0] = param_idx[m][k][0][label_split]'''
                                tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                                """ elif 'decoder' in k and 'linear2' in k:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1 """
                            else:
                                tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                count[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[param_idx[m][k]] += local_parameters[m][k]
                            count[k][param_idx[m][k]] += 1
                    else:
                        """ if 'decoder' in k and 'linear2' in k:
                            label_split = self.label_split[user_idx[m]]
                            param_idx[m][k] = param_idx[m][k][label_split]
                            tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                            count[k][param_idx[m][k]] += 1 """
                        """ else: """
                        tmp_v[param_idx[m][k]] += local_parameters[m][k]
                        count[k][param_idx[m][k]] += 1
                else:
                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                    count[k][torch.meshgrid(param_idx[m][k])] += 1
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            updated_parameters[k][count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        # delta_t = {k: v - self.global_parameters[k] for k, v in updated_parameters.items()}
        # if self.rounds in self.cfg['milestones']:
        #     self.eta *= 0.5
        # if not self.m_t or self.rounds in self.cfg['milestones']:
        #     self.m_t = {k: torch.zeros_like(x) for k, x in delta_t.items()}
        # self.m_t = {
        #     k: self.beta_1 * self.m_t[k] + (1 - self.beta_1) * delta_t[k] for k in delta_t.keys()
        # }
        # if not self.v_t or self.rounds in self.cfg['milestones']:
        #     self.v_t = {k: torch.zeros_like(x) for k, x in delta_t.items()}
        # self.v_t = {
        #     k: self.beta_2 * self.v_t[k] + (1 - self.beta_2) * torch.multiply(delta_t[k], delta_t[k])
        #     for k in delta_t.keys()
        # }
        # self.global_parameters = {
        #     k: self.global_parameters[k] + self.eta * self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau)
        #     for k in self.global_parameters.keys()
        # }
        #
        # # self.global_parameters = updated_parameters
        self.global_model.model.load_state_dict(updated_parameters)
        return

class TransformerServerRollSO(Server):
    def broadcast(self, local, lr):
        self.global_model.model.train(True)
        local_parameters, self.param_idx = self.distribute(self.current_sample_clients)
        for i in range(len(self.current_sample_clients)):
            self.clients_list[self.current_sample_clients[i]].update(local_parameters[i])