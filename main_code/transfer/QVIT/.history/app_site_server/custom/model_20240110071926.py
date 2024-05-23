import torch
import copy
import sys
import os
sys.path.append(os.path.dirname(__file__))
from timm.optim.adan import Adan
import math
from bern import Bern
from quantvit import *
from quantvit import QuantVisionTransformer
import torch.nn as nn
from functools import partial


def get_server_model(drop, img_size, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim,device, segment_mask, self_distillation, model_rate=1):
    model = None
    '''model = ViT(training_mode=training_mode, image_size=img_size[1], patch_size=8, num_classes=args.num_classes, dim=64, 
                               depth=math.floor(ratio*args.transformer_depth), full_depth=args.transformer_depth, 
                               heads=12, mlp_dim=args.mlp_dim, self_distillation=self_distillation, channels=img_size[0]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.local_lr)'''
    mlp_ratio = 1.0 * mlp_dim / embed_dim

    model = QuantVisionTransformer(img_size=img_size[1], patch_size=8, in_chans=img_size[0], num_classes=num_classes, 
                                   embed_dim=embed_dim, depth=transformer_depth, num_heads=transformer_head, mlp_ratio=mlp_ratio, 
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), wbits=4, abits=4).to(device)

    return model


def get_client_model(drop, img_size, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim, device, self_distillation, lr, weight_decay, no_prox, model_rate=1):
    model, optimizer = None, None
    '''model = ViT(training_mode=training_mode, image_size=img_size[1], patch_size=8, num_classes=args.num_classes, dim=64, 
                               depth=math.floor(ratio*args.transformer_depth), full_depth=args.transformer_depth, 
                               heads=12, mlp_dim=args.mlp_dim, self_distillation=self_distillation, channels=img_size[0]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.local_lr)'''

    mlp_ratio = 1.0 * mlp_dim / embed_dim
    model = QuantVisionTransformer(img_size=img_size[1], patch_size=8, in_chans=img_size[0], num_classes=num_classes, 
                                   embed_dim=embed_dim, depth=transformer_depth, num_heads=transformer_head, mlp_ratio=mlp_ratio, 
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), wbits=4, abits=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

    return model, optimizer

class FedModel:
    """
        Federated model.
    """
    def __init__(self, drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, lambda1, temperature, 
                 segment_mask=None, self_distillation=True, weight_decay=None, no_prox=None, lr=None, model_rate=1, device=None):
        
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
    
    #Compute kd loss
    def kd_loss_function(self, temperature, output, target_output):
        """
        para: output: middle ouptput logits.
        para: target_output: final output has divided by temperature and softmax.
        """
        output = output / temperature
        output_log_softmax = torch.log_softmax(output, dim=1)
        loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return loss_kd

    def train_weights(self, data_loader, local_sub_epoch):
        return self.weights_perform_local_epochs(data_loader, local_sub_epoch)

    def train_mask(self, data_loader, local_sub_epoch):
        return self.mask_perform_local_epochs(data_loader, local_sub_epoch)


    def weights_perform_local_epochs(self, data_loader, local_sub_epoch):
        for name,parameters in self.model.named_parameters():
            print(name,':',parameters.size())
        loss = None
        for epoch in range(local_sub_epoch):
            running_loss = 0
            total = 0
            criterion = torch.nn.CrossEntropyLoss().to(self.device)
            correct = 0
            for batch_idx, (train_x, train_y) in enumerate(data_loader):
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                total += train_x.size(0)
                
                self.optimizer.zero_grad()
                '''start = time.time()'''
                y_pred = self.model(train_x)
                '''end = time.time() - start
                print('Training complete in ', end)
                #train_y = train_y.to(torch.int64) 
                start = time.time()'''
                loss = criterion(y_pred[0], train_y.long())
                for i in range(1, len(y_pred)):
                    loss += criterion(y_pred[i], train_y.long())
                '''end = time.time() - start
                print('class loss complete in ', end)
                start = time.time()'''
                temperature = self.temperature
                densest_output = y_pred[-1] / temperature
                densest_output = torch.softmax(densest_output, dim=1)
                for i in range(len(y_pred)-1):
                    loss += self.kd_loss_function(temperature, y_pred[i], densest_output.detach()) * (temperature**2)
                '''end = time.time() - start
                print('kd loss complete in ', end)'''
                running_loss += loss.item()
                
                _, pred_y = torch.max(y_pred[-1].data, 1)
                correct += (pred_y == train_y).sum().item()
                '''start = time.time()'''
                loss.backward()
                self.optimizer.step()
                '''end = time.time() - start
                print('backward and optimizer loss complete in ', end)'''
            if epoch == local_sub_epoch-1:
                train_loss = running_loss / total
                accuracy = correct / total
                print("WEIGHT: train loss {}  -  Accuracy {}".format(train_loss, accuracy))
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
                #TODOcomputation & communication
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

