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
from timm.optim.adan import Adan
import Dense_ViT, Dense_dynamic_ViT_v2
import numpy as np
import math
from client import Client
from model import FedModel
#from sklearn.metrics import f1_score, accuracy_score

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

def get_data_path():
    root_path="/home/***/code/src/code and baselines/dataset/imagenet2012/imagenet200/fed_data/client_7non-iid_1.5/"

    dir=os.listdir(root_path)
    client=None
    for ddir in dir:
        if 'client' in ddir:
            client=ddir
            break
    root_path=root_path+'/'+client
    return root_path    

def calculate_f1_score(y_true, y_pred, average='weighted'):
    unique_labels = torch.unique(torch.cat((y_true, y_pred)))
    num_labels = len(unique_labels)

    tmp_dict={}
    list_unique_labels = list(unique_labels.clone().detach().cpu().numpy())
    for i in range(len(list_unique_labels)):
        tmp_dict[list_unique_labels[i]] = i
    
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


class fedrolex_c(Executor):
    #system_info={}
    def __init__(
        self,
        lr=0.05,
        epochs=10,
        task_get_model_rate ='get_local_model_rate',
        train_task_name="train",
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
        segment_mask = None,
        self_distillation = True,
    ):
        """Cifar10 Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on CIFAR10 dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
            pre_train_task_name: Task name for pre train task, i.e., sending initial model weights.
        """
        super().__init__()
        self._current_round = None
        self.model_rate=data_utils.get_local_model_rate()
        self._epochs = epochs
        self._train_task_name = train_task_name
        self.task_get_model_rate = task_get_model_rate
        self.device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        root_path=data_utils.get_data_path()
        self.train_loader, self.test_loader, img_size = data_utils.get_ImageNetdataloaders(root_path+'/train',root_path+'/test')
        self.mask_round_ratio = mask_round_ratio
        self.client = Client(drop, img_size, num_classes, embed_dim, mlp_dim, transformer_depth, transformer_depth,
                             lambda1, temperature,
                             weight_decay, no_prox, lr,
                             self.train_loader, self.test_loader, segment_mask, self_distillation, ratio=self.model_rate, mask_round_ratio=mask_round_ratio, device=self.device)
        #self.model,self.optimizer = get_model(drop=drop,img_size=img_size,num_classes=num_classes,embed_dim=embed_dim,transformer_depth=transformer_depth,transformer_head=transformer_head,mlp_dim=mlp_dim,device=self.device,lr=lr,weight_decay=weight_decay,no_prox=no_prox,model_rate=self.model_rate)

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

                weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)
                return outgoing_dxo.to_shareable()
            
            elif task_name == self.task_get_model_rate :

                outgoing_dxo = DXO(data_kind='model_rate', data={'model_rate' : self.model_rate})
                return outgoing_dxo.to_shareable()

            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _get_model_weights(self) -> Shareable:   #TODO这里需要改成传梯度，可以直接调用client的upload_gradients函数
        # Get the new state dict and send as weights
        client_gradients, receive_round = self.client.upload_gradients()
        gradients = {k: v.cpu().numpy() for k, v in client_gradients.items()}
        gradients_and_round={"gradients": gradients, "receive_round": receive_round}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=gradients_and_round, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()

    def _local_test_then_train(self, fl_ctx, weights, abort_signal):
        self.client.load_local_mask_and_classifier()   #最开始是DenseViT，为其设置分类头
        weight_model_weights  =copy.deepcopy(weights)
        last_mask_indices = None
        for k, v in self.client.mask_indices.items():
            #weight_model_weights[k] = weight_model_weights[k] * v
            if "weight" in k:
                weight_model_weights[k] = weight_model_weights[k] * v.reshape((-1,1))
                non_zero_rows = torch.nonzero(torch.sum(weight_model_weights[k], dim=1)).squeeze()
                weight_model_weights[k] = torch.index_select(weight_model_weights[k], 0, non_zero_rows)
                #对于输入进行改变了的也得进行变换
                if last_mask_indices is not None:  
                    if "linear2" in k:
                        weight_model_weights[k] = weight_model_weights[k][:, last_mask_indices.nonzero().squeeze()]
                    elif "to_out" in k:
                        weight_model_weights[k] = weight_model_weights[k][:, last_mask_indices[:int(last_mask_indices.size(0)/3)].nonzero().squeeze()]
                last_mask_indices = copy.deepcopy(v)
            else:
                weight_model_weights[k] = weight_model_weights[k] * v
                non_zero_indices = torch.nonzero(weight_model_weights[k]).squeeze()
                weight_model_weights[k] = weight_model_weights[k][non_zero_indices]
            
        # Set the model weights
        self.client.model.load_state_dict(state_dict=weight_model_weights, strict=False)
        
        f1_score , val_loss , val_acc = self.evaluate(self.model,self.test_loader)
        self.log_info(fl_ctx,"[train round {}] loss: {:.3f}, acc: {:.3f} , f1_score :{:.3f}".format(self._current_round,val_loss,val_acc,f1_score))

        if self._current_round <= 19: 
            self.client.train_local(self._current_round, copy.deepcopy(weights), self._epochs)
        else:
            self.client.set_local_round(self._current_round)
            loss_weight = self.client.local_model.train_weights(data_loader=self.train_loader, 
                                                        local_sub_epoch=self._epochs)

    @torch.no_grad()
    def evaluate(self,model, data_loader):
        loss_function = torch.nn.CrossEntropyLoss()

        model.eval()

        accu_loss = torch.zeros(1).to(self.device)  # 累计损失

        sample_num = 0
        predictions=torch.as_tensor([],device=self.device)
        targets=torch.as_tensor([],device=self.device)
        for step, data in enumerate(data_loader):
            images, labels = data[0].to(self.device),data[1].to(self.device)
            targets=torch.cat([targets,labels],dim=0)
            sample_num += images.shape[0]

            pred = model(images.to(self.device)).softmax(dim=1)
            pred_classes = torch.max(pred, dim=1)[1]

            predictions=torch.cat([predictions,pred_classes],dim=0)
            #accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(self.device))
            accu_loss += loss
        targets = targets.to(torch.int32)
        predictions = predictions.to(torch.int32)
        f1 = calculate_f1_score(targets, predictions, average='weighted')
        accuracy = calculate_accuracy(targets, predictions)
        return f1 , accu_loss.item() / sample_num , accuracy



    
