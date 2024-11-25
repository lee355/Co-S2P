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
import Dense_ViT
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


def get_model(drop, img_size,num_classes,embed_dim,transformer_depth,transformer_head, mlp_dim,device,lr,weight_decay,no_prox,model_rate=1):
    model, optimizer = None, None
    model = Dense_ViT.ViT(drop=drop, model_rate=model_rate, image_size=img_size, patch_size=16, num_classes=num_classes, dim=int(np.ceil(embed_dim*model_rate)),
                      depth=transformer_depth, heads=transformer_head, 
                      mlp_dim=int(np.ceil(mlp_dim*model_rate))).to(device)
    #model.apply(init_param)
    optimizer = Adan(model.parameters(), lr=lr, weight_decay=weight_decay, no_prox=no_prox)
    return model, optimizer
   

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
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        root_path='/home/***/fed_data/client_16non-iid_1.5/client_0'
        self.train_loader,self.test_loader,img_size = data_utils.get_ImageNetdataloaders(root_path+'/train',root_path+'/test')
        self.model,self.optimizer = get_model(drop=drop,img_size=img_size,num_classes=num_classes,embed_dim=embed_dim,transformer_depth=transformer_depth,transformer_head=transformer_head,mlp_dim=mlp_dim,device=self.device,lr=lr,weight_decay=weight_decay,no_prox=no_prox,model_rate=self.model_rate)

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

    def _get_model_weights(self) -> Shareable:
        # Get the new state dict and send as weights
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()

    def _local_test_then_train(self, fl_ctx, weights, abort_signal):
        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        f1_score , val_loss , val_acc = self.evaluate(self.model,self.test_loader)

        self.log_info(fl_ctx,"[train round {}] loss: {:.3f}, acc: {:.3f} , f1_score :{:.3f}".format(self._current_round,val_loss,val_acc,f1_score))


        # Basic training
        self.model.train()
        for epoch in range(self._epochs):
            running_loss = torch.zeros(1).to(self.device)
            sample_num = torch.zeros(1).to(self.device)
            for i, batch in enumerate(self.train_loader):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                sample_num += labels.size()[0]
                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                """ for name, p in self.model.named_parameters():
                    #if ("head" in name or "pre_logits" in name) and 'weight' in name:
                    tensor = p.data
                    grad_tensor = p.grad.data
                    abs_tensor=tensor.flatten().__abs__()
                    abs_tensor.sort()
                    EPS=abs_tensor[int(0.25*len(abs_tensor))]
                    grad_tensor = torch.where(tensor < EPS, 0, grad_tensor)
                    p.grad.data = grad_tensor """
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += cost 
            self.log_info(fl_ctx, f"Epoch: {epoch}/{self._epochs}, " f"Loss: {running_loss / sample_num}")

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

            since = time.time()
            pred = model(images.to(self.device)).softmax(dim=1)
            pred_classes = torch.max(pred, dim=1)[1]

            predictions=torch.cat([predictions,pred_classes],dim=0)
            #accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(self.device))
            accu_loss += loss

        f1 = f1_score(targets.cpu(), predictions.cpu(), average='weighted')
        accuracy = accuracy_score(targets.cpu(), predictions.cpu())
        return f1 , accu_loss.item() / sample_num , accuracy



    
