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
                                roll = self.rounds % output_size
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
                                roll = self.rounds % output_size
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
                                roll = self.rounds % output_size
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
                    roll = self.rounds % input_size
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

class Server:
    """
        Class for the central authority, i.e., the server, that coordinates the federated process.
    """

    def __init__(self, test_data ,drop, num_classes , embed_dim , transformer_depth , transformer_head, img_size, ratio_ls, device):
        
        self.device = device
        self.img_size = img_size
        self.training_mode = "mask" 
        self.global_model = FedModel(self.args, ratio=1.0, img_size=self.img_size, 
                                     training_mode='weight', device=self.device) 
        current_folder = Path(__file__).parent.resolve()
        parent_folder = Path(__file__).parent.parent.resolve()
        self.server_model_path = parent_folder.joinpath("fed_checkpoint/tmp13/server_model/local_mask_and_classifier.pth")
        self.round_global_weights.append(self.global_model.get_weights())
        self.all_client_model_list = self.generate_client_models()  #所有客户端的列表
        self.current_sample_clients = []   #当前轮客户端的列表
        self.current_training_clients = []  #目前还在训练的客户端列表
        self.current_trained_clients = []   #目前已经训练完的客户端
        self.current_aggregate_clients = []  #当前轮要聚合的客户端列表
        self.current_idle_clients = []
        self.stale_steps = 0
        self.update_number = 0
        self.stale_ths = 10
        self.n_round = 0

        if self.args.mode == "mask":
            self.alphas = dict()
            self.betas = dict()
            self.lambda_init = 1
            for k, val in self.global_model.model.named_parameters():
                self.alphas[k] = torch.ones_like(val) * self.lambda_init
                self.betas[k] = torch.ones_like(val) * self.lambda_init

    def generate_client_models(self):
        client_model_ls = []
        for ratio in self.ratio_ls:
            client_model_ls.append(FedModel(self.args, ratio=ratio, img_size=self.img_size, training_mode=self.training_mode, device=self.device))
        return client_model_ls
    
    def set_current_round(self, n_round):
        self.n_round = n_round
    

    def sample_clients(self):
        """
            Sample clients at each round (now uniformly at random, can be changed).
        """
        #从上一轮中被聚合的客户端以及上一轮闲散的客户端中选择
        n_samples = math.floor(self.args.frac * self.args.num_users)
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
        self.current_aggregate_clients = self.current_sample_clients  #TODO
        return self.current_aggregate_clients
    
    #这一轮没有被聚合的
    def set_continue_training_clients(self):
        self.current_training_clients = list(set(self.current_training_clients) - set(self.current_aggregate_clients))

    def set_idle_clients(self):
        self.current_idle_clients += self.current_aggregate_clients   #现在闲置的加上上一轮训练结束的
    
    #在每一次聚合之后进行模型权重（没有掩码）的保存
    def save_global_weight(self):
        param_dict = {k: v for k, v in self.global_model.get_weights().items()}
        self.round_global_weights.append(param_dict)
        torch.save({'weight_model_state_dict': param_dict}, self.server_model_path)   #存储最新的模型参数

    def find_bitrate(self, probs, num_params):
        local_bitrate = 0
        for p in probs:
            local_bitrate += p * math.log2(1 / p)
        return local_bitrate * num_params

    
    def set_init_training_mode(self, current_epoch, mask_convergence):
        self.training_mode = "mask" if current_epoch > self.args.local_ep/2 or mask_convergence == True else "weight"
    
    
    def compute_staleness(self, client_idxs, clinet_round_ls, client_gradients_ls):
        '''
        client_gradients_ls: 待聚合的客户端对应的梯度
        clinet_round_ls: 客户端收到服务器的全局参数的轮次
        client_idxs: 当前轮待聚合的客户端的id
        '''
        client_gamma_ls = []
        assert len(client_idxs) == len(clinet_round_ls)
        assert len(client_idxs) == len(client_gradients_ls)
        for i in range(len(clinet_round_ls)):
            '''print(self.n_round)
            print(clinet_round_ls[i])'''
            latest_weights = self.round_global_weights[self.n_round]
            client_weights = self.round_global_weights[clinet_round_ls[i]]
            
            gradients_importance = sum(torch.sum(torch.abs(v)) 
                                       for v in client_gradients_ls[i].values()) / self.ratio_ls[client_idxs[i]]
            weights_differnece = sum(torch.sum(torch.abs(latest_weights[k] - client_weights[k])) 
                                     for k in client_weights.keys())
            '''print(gradients_importance)
            print(weights_differnece)'''
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

    def aggregate_gradients(self, n_round, parallel_result):
        self.sample_gradients_aggregation(n_round, parallel_result)
    

    def sample_gradients_aggregation(self, n_round, parallel_result):
        aggregated_weights = copy.deepcopy(self.global_model.get_weights())
        aggregated_p = dict()
        
        for k, v in self.global_model.model.named_parameters():
            if 'mask' in k:
                aggregated_p[k] = torch.zeros_like(v)

        client_gradients_ls = []
        clinet_round_ls = []
        for client_id in self.current_aggregate_clients:
            tmp = parallel_result[client_id]
            client_gradients_ls.append(tmp[0])
            clinet_round_ls.append(tmp[1])
        #print(client_gradients_ls[0])
        client_staleness_ls = self.compute_staleness(self.current_aggregate_clients, clinet_round_ls, client_gradients_ls)
        for i in range(client_staleness_ls.shape[0]):
            for key,value in client_gradients_ls[i].items():
                if "to_qkv.bias" not in key:   #weight模型中设置了to_qkv的bias为false，但是mask中无法进行设置
                    aggregated_weights[key] += client_staleness_ls[i] * client_gradients_ls[i][key]
        
        self.global_model.set_weights(aggregated_weights)
        self.save_global_weight()


    def sample_mask_aggregation(self, client_idxs, n_round):
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
            # self.update_prior(n_round)
            self.reset_prior()
            n_samples = self.compute_n_mask_samples(n_round)
            p_update = []
            avg_bitrate = 0
            avg_freq = 0
            for client in client_idxs:
                sampled_mask, local_freq, num_params = self.clients_list[client].upload_mask(
                    n_samples=n_samples)
                avg_freq += local_freq
                local_bitrate = self.find_bitrate([local_freq + 1e-50, 1 - 
                                                   local_freq + 1e-50], num_params) + math.log2(num_params)
                avg_bitrate += local_bitrate / num_params
                for k, v in sampled_mask.items():
                    if 'mask' in k:
                        self.alphas[k] += v
                        self.betas[k] += (n_samples-v)
                        # Add layerwise estimated ps for each client
                        p_update.extend(v.cpu().numpy().flatten()/n_samples)
            avg_bitrate = avg_bitrate / len(client_idxs)
            avg_freq = avg_freq / len(client_idxs)
            # Update the posterior, and compute the mode of the beta distribution, as suggested in
            # https://neurips2021workshopfl.github.io/NFFL-2021/papers/2021/Ferreira2021.pdf
            for k, val in aggregated_weights.items():
                if 'mask' in k:
                    avg_p = (self.alphas[k] - 1) / (self.alphas[k] + self.betas[k] - 2)
                    if self.args.optimizer_noisy:
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

    def broadcast_model(self, sampled_clients, n_round):
        """
            Send the global updated model to the clients. 
        """
        for client in sampled_clients:
            self.clients_list[client].set_local_training_mode(self.training_mode)
            #self.clients_list[client].set_local_weights(self.all_client_model_list[client].get_weights())
            self.clients_list[client].set_local_weights(self.global_model.get_weights())
            self.clients_list[client].set_local_round(n_round)
