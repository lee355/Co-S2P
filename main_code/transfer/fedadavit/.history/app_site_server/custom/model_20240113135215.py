import torch
import copy
import sys
import os
sys.path.append(os.path.dirname(__file__))
from timm.optim.adan import Adan
import math
import torch.nn as nn
from functools import partial
from losses import AdaHeadLoss, AdaLoss, TeacherLoss
from ada_t2t_vit import AdaStepT2T_ViT

def get_server_model(drop, img_size, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim,device, segment_mask, self_distillation, model_rate=1):
    model = None
    '''model = ViT(training_mode=training_mode, image_size=img_size[1], patch_size=8, num_classes=args.num_classes, dim=64, 
                               depth=math.floor(ratio*args.transformer_depth), full_depth=args.transformer_depth, 
                               heads=12, mlp_dim=args.mlp_dim, self_distillation=self_distillation, channels=img_size[0]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.local_lr)'''
    mlp_ratio = 1.0 * mlp_dim / embed_dim

    model = QuantVisionTransformer(img_size=img_size, patch_size=16, in_chans=3, num_classes=num_classes, 
                                   embed_dim=embed_dim, depth=math.floor(transformer_depth*model_rate), num_heads=transformer_head, mlp_ratio=mlp_ratio, 
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), wbits=2, abits=2).to(device)

    return model


def get_client_model(drop, img_size, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim, device, self_distillation, lr, weight_decay, no_prox, model_rate=1):
    model, optimizer = None, None
    '''model = ViT(training_mode=training_mode, image_size=img_size[1], patch_size=8, num_classes=args.num_classes, dim=64, 
                               depth=math.floor(ratio*args.transformer_depth), full_depth=args.transformer_depth, 
                               heads=12, mlp_dim=args.mlp_dim, self_distillation=self_distillation, channels=img_size[0]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.local_lr)'''

    mlp_ratio = 1.0 * mlp_dim / embed_dim
    model = QuantVisionTransformer(img_size=img_size, patch_size=16, in_chans=3, num_classes=num_classes, 
                                   embed_dim=embed_dim, depth=math.floor(transformer_depth*model_rate), num_heads=transformer_head, mlp_ratio=mlp_ratio, 
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), wbits=2, abits=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

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
    
    def train_weights(self, data_loader, local_sub_epoch):
        return self.perform_local_epochs(data_loader)



    def perform_local_epochs(self, data_loader):
        """
            Compute local epochs, the training stategies depends on the adopted model.
        """
        loss = None
        train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
        # train_loss_fn = AdaHeadLoss(train_loss_fn, target_ratio=args.head_target_ratio,head_loss_ratio=args.head_ratio)
        criterion = AdaLoss(train_loss_fn, 
                            head_target_ratio=0.5, head_loss_ratio=2.0, 
                            layer_target_ratio=0.5, layer_loss_ratio=2.0,
                            head_diverse_ratio=0., layer_diverse_ratio=0.,
                            head_entropy_weight=0., layer_entropy_weight=0.,
                            head_minimal_weight=0., head_minimal=0.,
                            layer_minimal_weight=0., layer_minimal=0.,
                            token_target_ratio=0.5, token_loss_ratio=2.0, 
                            token_minimal=0., token_minimal_weight=0.)
        for epoch in range(self.args.local_ep):
            running_loss = 0
            total = 0
            #criterion = torch.nn.CrossEntropyLoss()
            correct = 0
            for batch_idx, (train_x, train_y) in enumerate(data_loader):
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                total += train_x.size(0)
                self.optimizer.zero_grad()
                y_pred = self.model(train_x, training=True, ret_attn_list=False)
                # loss = criterion(samples, outputs, targets) + bitops_scaler * (bitops - 21.455 * 1e9) ** 2
                # loss = criterion(samples, outputs, targets) + bitops_scaler * bitops
            
                train_y = train_y.to(torch.long) 
                loss, meta_loss = criterion(y_pred, train_y)
                running_loss += loss.item()
                _, pred_y = torch.max(y_pred[0].data, 1)
                correct += (pred_y == train_y).sum().item()
                loss.backward()
                self.optimizer.step()
            train_loss = running_loss / total
            accuracy = correct / total
            print("Epoch {}: train loss {}  -  Accuracy {}".format(epoch+1, train_loss, accuracy))
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


