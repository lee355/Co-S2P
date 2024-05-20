import torch
import copy
import sys
import os
sys.path.append(os.path.dirname(__file__))
from timm.optim.adan import Adan
import masked_VIT
import math
from bern import Bern

def get_server_model(drop, img_size, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim,device, segment_mask, self_distillation, model_rate=1):
    model, optimizer = None, None
    model = masked_VIT.ViT(training_mode="mask", image_size=img_size, patch_size=16, num_classes=num_classes, dim=embed_dim,
                               depth=math.floor(model_rate*transformer_depth), full_depth=transformer_depth, heads=transformer_head, 
                               mlp_dim=mlp_dim, self_distillation=False, channels=3).to(device)
    #model.apply(init_param)
    return model


def get_client_model(drop, img_size, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim, device, self_distillation, lr, weight_decay, no_prox, model_rate=1):
    model, optimizer = None, None
    model = masked_VIT.ViT(training_mode="mask", drop=drop, image_size=img_size, patch_size=16, num_classes=num_classes, dim=embed_dim, 
                               depth=math.floor(model_rate*transformer_depth), full_depth=transformer_depth, 
                               heads=transformer_head, mlp_dim=mlp_dim, self_distillation=False, channels=3).to(device)
    #model.apply(init_param)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

class FedModel:
    """
        Federated model.
    """
    def __init__(self, drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, lambda1, temperature, 
                 segment_mask=None, self_distillation=False, weight_decay=None, no_prox=None, lr=None, model_rate=1, device=None):
        
        self.lambda1 = lambda1
        self.temperature = temperature
        self.device = device
        self.model_rate = model_rate
        if lr is None:
            self.model = get_server_model(drop=drop, img_size=img_size, num_classes=num_classes, 
                                          embed_dim=embed_dim, transformer_depth=transformer_depth, transformer_head=transformer_head, 
                                          mlp_dim=mlp_dim, device=self.device, segment_mask=segment_mask, self_distillation=self_distillation, model_rate=self.model_rate)
        else:
            self.model, self.optimizer = get_client_model(drop=drop, img_size=img_size, num_classes=num_classes, 
                                                          embed_dim=embed_dim, transformer_depth=transformer_depth, transformer_head=transformer_head, 
                                                          mlp_dim=mlp_dim, device=self.device, self_distillation=self_distillation, 
                                                          lr=lr, weight_decay=weight_decay, no_prox=no_prox, model_rate=self.model_rate)

    
        self.model_size = self.compute_model_size()  # bit


    def compute_model_size(self):
        """
            Assume torch.FloatTensor --> 32 bit
        """
        tot_params = 0
        for param in self.model.parameters():
            tot_params += param.numel()
        return tot_params * 32

    def inference(self, x_input):
        with torch.no_grad():
            self.model.eval()
            return self.model(x_input)


    def train_mask(self, data_loader, local_sub_epoch):
        return self.perform_local_epochs(data_loader, local_sub_epoch)

    def compute_delta(self, data_loader):
        """
            In case of SignSGD or Fedavg, compute the gradient of the local model.
        """

        delta = dict()
        xt = dict()
        for k, v in self.model.named_parameters():
            delta[k] = torch.zeros_like(v)
            xt[k] = copy.deepcopy(v)

        # Update local model
        loss = self.perform_local_epochs(data_loader)
        for k, v in self.model.named_parameters():
            delta[k] = v - xt[k]

        # Error compensation
        if self.params.get('model').get('optimizer').get('type') == 'ef_sign_sgd':
            with torch.no_grad():
                for k, v in self.model.named_parameters():
                    delta[k] += self.beta_error * self.e[k]
                    self.e[k] = delta[k] - torch.sign(delta[k])

        return loss, delta

    def perform_local_epochs(self, data_loader, local_sub_epoch):
        """
            Compute local epochs, the training stategies depends on the adopted model.
        """
        loss = None
        for epoch in range(local_sub_epoch):
            running_loss = 0
            total = 0
            criterion = torch.nn.CrossEntropyLoss()
            correct = 0
            for batch_idx, (train_x, train_y) in enumerate(data_loader):
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                total += train_x.size(0)
                self.optimizer.zero_grad()
                y_pred = self.model(train_x, ths=None)
                loss = criterion(y_pred, train_y)
                running_loss += loss.item()
                _, pred_y = torch.max(y_pred.data, 1)
                correct += (pred_y == train_y).sum().item()
                loss.backward()
                self.optimizer.step()
            if epoch == local_sub_epoch-1:
                train_loss = running_loss / total
                accuracy = correct / total
                print("Epoch {}: train loss {}  -  Accuracy {}".format(epoch+1, train_loss, accuracy))
        return loss
    
    def mask_perform_local_epochs(self, data_loader, local_sub_epoch):
        loss = None
        loss1 = None
        for epoch in range(local_sub_epoch):
            running_loss = 0
            total = 0
            criterion = torch.nn.CrossEntropyLoss()
            correct = 0
            for batch_idx, (train_x, train_y) in enumerate(data_loader):
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                total += train_x.size(0)
                self.optimizer.zero_grad()
                y_pred = self.model(train_x, ths=None)
                train_y = train_y.to(torch.int64) 
                #performance
                loss = criterion(y_pred, train_y)
                #computation & communication
                flag = True
                mask_parameter_size = 0
                for name, parameter in self.model.named_parameters():
                    if 'mask' in name:
                        mask_parameter_size += parameter.numel()
                        s_m = torch.sigmoid(parameter)
                        g_m = Bern.apply(s_m)
                        if flag:
                            loss1 = torch.sum(g_m)
                            flag = False
                        else:
                            loss1 += torch.sum(g_m)
                loss1 = self.lambda1 * torch.abs(loss1/mask_parameter_size - self.model_rate)
                loss = loss + loss1
                running_loss += loss.item()
                _, pred_y = torch.max(y_pred.data, 1)
                correct += (pred_y == train_y).sum().item()
                loss.backward()
                self.optimizer.step()
            if epoch == local_sub_epoch-1:
                train_loss = running_loss / total
                accuracy = correct / total
                print("performance loss", loss.item())
                print("Computation & communication loss", loss1.item())
                print("MASK: train loss {}  -  Accuracy {}".format(train_loss, accuracy))
        return loss

    def set_weights(self, w):
        self.model.load_state_dict(
            copy.deepcopy(w), strict=False
        )

    def get_weights(self):
        return self.model.state_dict()

    def save(self, folderpath):
        torch.save(self.model.state_dict(), folderpath.joinpath("local_model"))

    def load(self, folderpath):
        self.model.load_state_dict(torch.load(folderpath.joinpath("local_model"),
                                              map_location=torch.device('cpu')))


