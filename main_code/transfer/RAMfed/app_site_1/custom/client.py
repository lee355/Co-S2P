import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import bernoulli
from model import FedModel
#from .models.pruned_layers import *

import copy
from collections import OrderedDict
import time

class Client:
    def __init__(self, args, client_id, img_size, dataset, device):
        self.args = args
        self.client_id = client_id
        self.local_data_loader = dataset
        self.n = len(self.local_data_loader)
        # Local model at the client
        self.local_model = None  #FedModel
        self.device = device
        self.img_size = img_size
        #self.set_local_weights(initial_weights)
        self.accumulated_gradients = None
        self.delta = None
        self.model_size = None  # bit
        self.epsilon = 0.01
        self.model_rate=None
        #self.embed_size = self.args.embed_size

    def train_local(self, n_round):
        print("Client "+str(self.client_id)+" with drop rate "+str(self.model_rate)+" start local training!!")
        since = time.time()
        loss = self.local_model.train_weights(data_loader=self.local_data_loader,local_sub_epoch=self.args.local_ep)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def set_local_weights(self,w):
        #self.set_local_rate(model_rate)
        print('************************************************************************************')
        print(self.client_id)
        print(self.model_rate)
        #self.embed_size = int(np.ceil(self.embed_size * self.model_rate))
        self.local_model = FedModel(self.args, img_size=self.img_size, model_rate=self.model_rate, device=self.device)
        self.local_model.set_weights(w)
        self.model_size = self.local_model.model_size
    
    def set_local_rate(self, model_rate):
        self.model_rate = model_rate

    def get_local_weights(self):
        return self.local_model.get_weights()

    def upload_weight(self):
        with torch.no_grad():
            local_model_dict = self.local_model.get_weights()
        return local_model_dict

