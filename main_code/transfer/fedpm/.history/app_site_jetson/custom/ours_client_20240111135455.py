# -*- coding: gbk -*-
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
""" from torch_geometric.nn import GCNConv """

import torch
import copy
from numpy import transpose
import data_utils
from timm.optim.adan import Adan
import numpy as np
import math
from client import Client
from model import FedModel
import time
#from sklearn.metrics import f1_score, accuracy_score

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

def calculate_f1_score(y_true, y_pred, average='weighted'):
    unique_labels = torch.unique(torch.cat((y_true, y_pred)))
    num_labels = len(unique_labels)

    tmp_dict={}
    list_unique_labels = list(unique_labels.clone().detach().cpu().numpy())
    for i in range(len(list_unique_labels)):
        tmp_dict[list_unique_labels[i]] = torch.as_tensor(i)
    
    for i in range(len(y_true)):
        y_true[i]=tmp_dict[y_true[i].item()]
        y_pred[i]=tmp_dict[y_pred[i].item()]

    unique_labels = torch.unique(torch.cat((y_true, y_pred)))
    num_labels = len(unique_labels)

    true_positives = torch.zeros(num_labels).to(y_pred.device)
    false_positives = torch.zeros(num_labels).to(y_pred.device)
    false_negatives = torch.zeros(num_labels).to(y_pred.device)

    for label in unique_labels:
        true_positives[label] = torch.sum((y_true == label) & (y_pred == label))
        false_positives[label] = torch.sum((y_true != label) & (y_pred == label))
        false_negatives[label] = torch.sum((y_true == label) & (y_pred != label))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores[f1_scores.isnan()] = 0  # Handle cases where precision + recall is 0

    if average == 'weighted':
        weights = torch.bincount(y_true)
        f1_score = torch.sum(weights * f1_scores) / torch.sum(weights)
    elif average == 'macro':
        f1_score = torch.mean(f1_scores)
    else:
        raise ValueError("Invalid average parameter. Use 'weighted' or 'macro'.")

    return f1_score.item()


def calculate_accuracy(y_true, y_pred):
    correct_predictions = torch.sum(y_true == y_pred).item()
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy


class Ours_c(Executor):
    #system_info={}
    def __init__(
        self,
        lr=0.05,
        epochs=10,
        train_task_name="train",
        task_get_model_rate ='get_local_model_rate',
        weight_decay=0.02,
        no_prox=False,
        drop = True, 
        num_classes = 1000,
        embed_dim = 64,
        transformer_depth = 12,
        transformer_head =8,
        mlp_dim = 256,
        mask_round_ratio = 0.5,
        lambda1 = 1.0, 
        temperature = 3,
        self_distillation = False
    ):

        super().__init__()
        self._current_round = None
        self.model_rate=data_utils.get_local_model_rate()
        self._epochs = epochs
        self._train_task_name = train_task_name
        self.task_get_model_rate = task_get_model_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        root_path=data_utils.get_data_path()
        self.train_loader, self.test_loader, img_size = data_utils.get_ImageNetdataloaders(root_path+'/train',root_path+'/test')
        self.mask_round_ratio = mask_round_ratio
        self.client = Client(drop, img_size, num_classes, embed_dim, mlp_dim, transformer_depth, transformer_head,
                             lambda1, temperature,
                             weight_decay, no_prox, lr,
                             self.train_loader, self.test_loader, self_distillation, model_rate=self.model_rate, mask_round_ratio=mask_round_ratio, device=self.device)
        #self.model,self.optimizer = get_model(drop=drop,img_size=img_size,num_classes=num_classes,embed_dim=embed_dim,transformer_depth=transformer_depth,transformer_head=transformer_head,mlp_dim=mlp_dim,device=self.device,lr=lr,weight_decay=weight_decay,no_prox=no_prox,model_rate=self.model_rate)

        self.loss_result = []
        self.top_1_result = []
        self.top_5_result = []
        self.f1_score_result = []

        self.loss = torch.nn.CrossEntropyLoss()


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                self.since = time.time()

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v,device=self.device) for k, v in dxo.data['weight'].items()}
                self._current_round = dxo.data['current_round']
                self._local_test_then_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                return self._get_model_weights()

            elif task_name == self.task_get_model_rate :

                outgoing_dxo = DXO(data_kind='model_rate', data={'model_rate' : self.model_rate})
                return outgoing_dxo.to_shareable()
            
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
            
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        
    def _local_test_then_train(self, fl_ctx, weights, abort_signal):
        if self._current_round != 0:
            
            weight_model_weights = copy.deepcopy(weights)
            
            self.client.local_model.model.load_state_dict(state_dict=weight_model_weights, strict=False)
            
            f1_score , val_loss , val_acc , val_acc_5= self.evaluate(self.client.local_model.model,self.test_loader)
            self.log_info(fl_ctx,"[train round {}] loss: {:.3f}, acc: {:.3f} , f1_score :{:.3f} , acc_5: {:.3f}".format(self._current_round,val_loss,val_acc,f1_score,val_acc_5))
            
            self.loss_result.append(val_loss)
            self.top_1_result.append(val_acc)
            self.f1_score_result.append(f1_score)
            self.top_5_result.append(val_acc_5)

            np.save(f'/home/nvidia/result/loss.npy', self.loss_result)
            np.save(f'/home/nvidia/result/top_1.npy', self.top_1_result)
            np.save(f'/home/nvidia/result/f1_score.npy', self.f1_score_result)
            np.save(f'/home/nvidia/result/top_5.npy', self.top_5_result)    
            
  
        self.client.train_local(self.train_loader, self._epochs)

        if self._current_round % 5 == 0 :
                torch.save(self.client.local_model.model.state_dict(),f'/home/nvidia/model/model_params_{self._current_round}.pth')


    def _get_model_weights(self) -> Shareable: 
        # Get the new state dict and send as weights
        client_gradients, local_freq, num_params = self.client.upload_mask(n_samples=1)  #TODO
        gradients = {k: v.cpu().numpy() for k, v in client_gradients.items()}
        train_time = time.time()-self.since
        gradients_and_round={"gradients": gradients, "local_freq": local_freq,'train_time' : train_time, "num_params":num_params}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=gradients_and_round, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._current_round}
        )
        return outgoing_dxo.to_shareable()  

    @torch.no_grad()
    def evaluate(self , model , data_loader):
        loss_function = torch.nn.CrossEntropyLoss()

        model.eval()

        accu_loss = torch.zeros(1).to(self.device)  

        sample_num = 0
        correct_1 = 0
        correct_5 = 0
        predictions=torch.as_tensor([],device=self.device)
        targets=torch.as_tensor([],device=self.device)
        for step, data in enumerate(data_loader):
            images, labels = data[0].to(self.device),data[1].to(self.device)
            targets=torch.cat([targets,labels],dim=0)
            sample_num += images.shape[0]

            pred = model(images.to(self.device))
            pred_classes = torch.max(pred[-1], dim=1)[1]

            labels = labels.to(torch.int64)

            correct_1 += (pred_classes == labels).sum().item()

            _, maxk = torch.topk(pred[-1], 5, dim=-1)
            correct_5 += (labels.view(-1,1) == maxk).sum().item()

            predictions=torch.cat([predictions,pred_classes] , dim=0)
            #accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred[-1], labels.to(self.device))
            accu_loss += loss
        
        targets = targets.to(torch.int32)
        predictions = predictions.to(torch.int32)

        f1 = calculate_f1_score(targets, predictions, average='weighted')
        return f1 , accu_loss.item() / sample_num , 1.0*correct_1/sample_num , 1.0*correct_5 / sample_num



    
