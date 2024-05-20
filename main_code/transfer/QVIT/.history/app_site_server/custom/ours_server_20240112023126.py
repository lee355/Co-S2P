# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any
import copy
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score
#from torch_geometric.nn import GCNConv

from torch import nn
import torch
from einops import rearrange
from numpy import transpose
import data_utils
import time
from server import Server
import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.client import Client
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable



def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = w_avg[k]/len(w)
    return w_avg


def _check_non_neg_int(data: Any, name: str):
    if not isinstance(data, int):
        raise ValueError(f"{name} must be int but got {type(data)}")

    if data < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")



class ScatterAndGather_Ours(Controller):
    def __init__(
        self,
        min_clients: int = 1,
        num_rounds: int = 5,
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        task_get_model_rate ='get_local_model_rate' ,
        train_task_name=AppConstants.TASK_TRAIN,
        train_timeout: int = 0,
        ignore_result_error: bool = False,
        allow_empty_global_weights: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
        snapshot_every_n_rounds: int = 1,
        drop = True, 
        num_classes = 1000,
        embed_dim = 256,
        transformer_depth = 12,
        transformer_head =8,
        mlp_dim = 1024,
        lambda1 = 1.0,
        temperature = 3
    ):
        """The controller for ScatterAndGather Workflow.

        The ScatterAndGather workflow defines FederatedAveraging on all clients.
        The model persistor (persistor_id) is used to load the initial global model which is sent to all clients.
        Each client sends it's updated weights after local training which is aggregated (aggregator_id). The
        shareable generator is used to convert the aggregated weights to shareable and shareable back to weight.
        The model_persistor also saves the model after training.

        Args:
            min_clients (int, optional): Min number of clients in training. Defaults to 1.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            start_round (int, optional): Start round for training. Defaults to 0.
            wait_time_after_min_received (int, optional): Time to wait before beginning aggregation after
                contributions received. Defaults to 10.
            aggregator_id (str, optional): ID of the aggregator component. Defaults to "aggregator".
            persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
            shareable_generator_id (str, optional): ID of the shareable generator. Defaults to "shareable_generator".
            train_task_name (str, optional): Name of the train task. Defaults to "train".
            train_timeout (int, optional): Time to wait for clients to do local training.
            ignore_result_error (bool, optional): whether this controller can proceed if client result has errors.
                Defaults to False.
            allow_empty_global_weights (bool, optional): whether to allow empty global weights. Some pipelines can have
                empty global weights at first round, such that clients start training from scratch without any global info.
                Defaults to False.
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
            persist_every_n_rounds (int, optional): persist the global model every n rounds. Defaults to 1.
                If n is 0 then no persist.
            snapshot_every_n_rounds (int, optional): persist the server state every n rounds. Defaults to 1.
                If n is 0 then no persist.

        Raises:
            TypeError: when any of input arguments does not have correct type
            ValueError: when any of input arguments is out of range
        """
        super().__init__(task_check_period=task_check_period)

        # Check arguments
        if not isinstance(min_clients, int):
            raise TypeError("min_clients must be int but got {}".format(type(min_clients)))
        elif min_clients <= 0:
            raise ValueError("min_clients must be greater than 0.")

        _check_non_neg_int(num_rounds, "num_rounds")
        _check_non_neg_int(start_round, "start_round")
        _check_non_neg_int(wait_time_after_min_received, "wait_time_after_min_received")
        _check_non_neg_int(train_timeout, "train_timeout")
        _check_non_neg_int(persist_every_n_rounds, "persist_every_n_rounds")
        _check_non_neg_int(snapshot_every_n_rounds, "snapshot_every_n_rounds")

        if not isinstance(train_task_name, str):
            raise TypeError("train_task_name must be a string but got {}".format(type(train_task_name)))

        if not isinstance(task_check_period, (int, float)):
            raise TypeError(f"task_check_period must be an int or float but got {type(task_check_period)}")
        elif task_check_period <= 0:
            raise ValueError("task_check_period must be greater than 0.")

        self.task_get_model_rate=task_get_model_rate
        self.train_task_name = train_task_name
        self.client_w={}
        self.client_modelrate={}
        self.client_received_round={}
        self.client_time = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_loader,img_size = data_utils.get_serverdataloader('/home/ly/code/src/code and baselines/dataset/imagenet2012/imagenet200/val')
        self.server = Server(self.test_loader, drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, lambda1, temperature, model_rate=1, device=self.device)
        self.param_idx = None
        self.loss_result = []
        self.top_1_result = []
        self.top_5_result = []
        self.f1_score_result = []
        self.resources_usage_result = []

        # config data
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._wait_time_after_min_received = wait_time_after_min_received
        self._start_round = start_round
        self._train_timeout = train_timeout
        self._persist_every_n_rounds = persist_every_n_rounds
        self._snapshot_every_n_rounds = snapshot_every_n_rounds
        self.ignore_result_error = ignore_result_error
        self.allow_empty_global_weights = allow_empty_global_weights

        # workflow phases: init, train, validate
        self._phase = AppConstants.PHASE_INIT
        self._global_weights = None
        self._current_round = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing ScatterAndGather workflow.")
        self._phase = AppConstants.PHASE_INIT

        self._global_weights=self.server.global_model.model.state_dict()

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:

            self.log_info(fl_ctx, "Beginning ScatterAndGather training phase.")
            self._phase = AppConstants.PHASE_TRAIN

            get_model_rate_task = Task(
                    name=self.task_get_model_rate,
                    data=DXO(data_kind="oo", data={'hello client' : 0}).to_shareable(),
                    props={},
                    timeout=self._train_timeout,
                    result_received_cb=self._process_get_model_rate_result,
                )

            self.broadcast_and_wait(
                task=get_model_rate_task,
                min_responses=self._min_clients,
                wait_time_after_min_received=self._wait_time_after_min_received,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )
            self.log_info(fl_ctx,f"client model rate: {self.client_modelrate}")
            
            if self._current_round is None:
                self._current_round = self._start_round
            while self._current_round < self._start_round + self._num_rounds:

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, f"Round {self._current_round} started.")

                since=time.time()
                
                weights = {k: v.cpu().numpy() for k, v in self.server.global_model.model.state_dict().items()}
                
                out_data={'weight':weights , 'current_round' : self._current_round}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=out_data)
                
                # Create train_task
                data_shareable:Shareable =outgoing_dxo.to_shareable()
                data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)
                
                train_task = Task(
                    name=self.train_task_name,
                    data=data_shareable,
                    props={},
                    timeout=self._train_timeout,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )

                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )
                
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return
                
                round_time = time.time() - since

                self.log_info(fl_ctx, "Start aggregation.")

                self.server.aggregate_weights(self.client_w, self.client_received_round)

                f1_score , val_loss , val_acc ,val_acc_5= self.evaluate(self.server.global_model.model,self.test_loader)

                resources_usage = self.resource_usage(round_time)

                self.loss_result.append(val_loss)
                self.top_1_result.append(val_acc)
                self.f1_score_result.append(f1_score)
                self.resources_usage_result.append(resources_usage)
                self.top_5_result.append(val_acc_5)

                self.log_info(fl_ctx,"[train round {}] loss: {:.3f}, acc: {:.3f} , f1_score :{:.3f} , resources_usage : {:.3f} , acc_5: {:.3f}".format(self._current_round,val_loss,val_acc,f1_score,resources_usage,val_acc_5))

                self.log_info(fl_ctx, "End aggregation.")

                np.save('/home/ly/result/loss.npy', self.loss_result)
                np.save('/home/ly/result/top_1.npy', self.top_1_result)
                np.save('/home/ly/result/f1_score.npy', self.f1_score_result)
                np.save('/home/ly/result/resources_usage.npy', self.resources_usage_result)
                np.save('/home/ly/result/top_5.npy', self.top_5_result)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, f"Round {self._current_round} finished.")
                self.log_info(fl_ctx,f"time : {time.time()-since}")
                if(self._current_round % 5 == 0):
                    torch.save(self.server.global_model.model.state_dict(), f'/home/ly/model/model_params_{self._current_round}.pth')
                self._current_round += 1
            #torch.save(self.server.global_model.model.state_dict(), f'/home/xugw/model/model_params_{self._current_round}.pth')
            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, "Finished ScatterAndGather Training.")
        except Exception as e:
            error_msg = f"Exception in ScatterAndGather control_flow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def stop_controller(self, fl_ctx: FLContext):
        self._phase = AppConstants.PHASE_FINISHED

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, None)
            if collector:
                if not isinstance(collector, GroupInfoCollector):
                    raise TypeError("collector must be GroupInfoCollector but got {}".format(type(collector)))

                collector.add_info(
                    group_name=self._name,
                    info={"phase": self._phase, "current_round": self._current_round, "num_rounds": self._num_rounds},
                )

    def _prepare_train_task_data(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        return

    def _process_train_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        result = client_task.result
        client_name = client_task.client.name

        self._accept_train_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

        # Cleanup task result
        client_task.result = None
    
    def _process_get_model_rate_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        result = client_task.result
        client_name = client_task.client.name

        self._accept_get_model_rate_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

        # Cleanup task result
        client_task.result = None

    def process_result_of_unknown_task(self, client: Client, task_name, client_task_id, result: Shareable, fl_ctx: FLContext) -> None:
        if self._phase == AppConstants.PHASE_TRAIN and task_name == self.train_task_name:
            self._accept_train_result(client_name=client.name, result=result, fl_ctx=fl_ctx)
            self.log_info(fl_ctx, f"Result of unknown task {task_name} sent to aggregator.")
        else:
            self.log_error(fl_ctx, "Ignoring result from unknown task.")

    def _accept_train_result(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> bool:

        rc = result.get_return_code()

        # Raise errors if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            if self.ignore_result_error:
                self.log_warning(
                    fl_ctx,
                    f"Ignore the train result from {client_name} at round {self._current_round}. Train result error code: {rc}",
                )
                return False
            else:
                self.system_panic(
                    f"Result from {client_name} is bad, error code: {rc}. "
                    f"{self.__class__.__name__} exiting at round {self._current_round}.",
                    fl_ctx=fl_ctx,
                )
                return False

        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)

        self.client_w[client_name]= {k: torch.as_tensor(v,device=self.device) for k, v in from_shareable(result).data['gradients'].items()}
        self.client_time[client_name] = from_shareable(result).data['train_time']

        return True
    
    def _accept_get_model_rate_result(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> bool:

        rc = result.get_return_code()

        # Raise errors if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            if self.ignore_result_error:
                self.log_warning(
                    fl_ctx,
                    f"Ignore the train result from {client_name} at round {self._current_round}. Train result error code: {rc}",
                )
                return False
            else:
                self.system_panic(
                    f"Result from {client_name} is bad, error code: {rc}. "
                    f"{self.__class__.__name__} exiting at round {self._current_round}.",
                    fl_ctx=fl_ctx,
                )
                return False

        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)

        self.client_modelrate[client_name] = from_shareable(result).data['model_rate']


        return True

    def _check_abort_signal(self, fl_ctx, abort_signal: Signal):
        if abort_signal.triggered:
            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, f"Abort signal received. Exiting at round {self._current_round}.")
            return True
        return False

    def get_persist_state(self, fl_ctx: FLContext) -> dict:
        return {
            "current_round": self._current_round,
            "start_round": self._start_round,
            "num_rounds": self._num_rounds,
            "global_weights": self._global_weights,
        }

    def restore(self, state_data: dict, fl_ctx: FLContext):
        try:
            self._current_round = state_data.get("current_round")
            self._start_round = state_data.get("start_round")
            self._num_rounds = state_data.get("num_rounds")
            self._global_weights = state_data.get("global_weights")
        finally:
            pass
    
    @torch.no_grad()
    def evaluate(self , model , data_loader):
        loss_function = torch.nn.CrossEntropyLoss()

        model.eval()

        accu_loss = torch.zeros(1).to(self.device)  # 累计损失

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
            pred_classes = torch.max(pred, dim=1)[1]

            labels = labels.to(torch.int64)

            correct_1 += (pred_classes == labels).sum().item()

            _, maxk = torch.topk(pred, 5, dim=-1)
            correct_5 += (labels.view(-1,1) == maxk).sum().item()

            predictions=torch.cat([predictions,pred_classes] , dim=0)
            #accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            loss = loss_function(pred, labels.to(self.device))
            accu_loss += loss

        f1 = f1_score(targets.cpu(), predictions.cpu(), average='weighted')
        return f1 , accu_loss.item() / sample_num , 1.0*correct_1/sample_num , 1.0*correct_5 / sample_num
    
    def resource_usage(self , round_time):
        all_time = round_time * len(self.client_time)
        real_time = sum(list(self.client_time.values()))
        
        return real_time / all_time
