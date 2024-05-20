import torch
import copy
import sys
import os
sys.path.append(os.path.dirname(__file__))
import Dense_ViT
from timm.optim.adan import Adan
import Dense_dynamic_ViT_v2, masked_dynamic_ViT_v2
import math

def get_server_model(drop, img_size, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim,device, segment_mask, self_distillation, model_rate=1):
    model, optimizer = None, None
    model = Dense_dynamic_ViT_v2.ViT(drop=drop, image_size=img_size, patch_size=16, num_classes=num_classes, dim=embed_dim,
                               depth=math.floor(model_rate*transformer_depth), full_depth=transformer_depth, heads=transformer_head, 
                               mlp_dim=mlp_dim, segment_mask=segment_mask, self_distillation=self_distillation, channels=3).to(device)
    #model.apply(init_param)
    return model


def get_client_model(drop, img_size, num_classes, embed_dim, transformer_depth, transformer_head, mlp_dim, device, self_distillation, lr, weight_decay, no_prox, model_rate=1):
    model, optimizer = None, None
    model = masked_dynamic_ViT_v2.ViT(training_mode="mask", drop=drop, image_size=img_size[1], patch_size=16, num_classes=num_classes, dim=embed_dim, 
                               depth=math.floor(model_rate*transformer_depth), full_depth=transformer_depth, 
                               heads=transformer_head, mlp_dim=mlp_dim, self_distillation=self_distillation, channels=img_size[0]).to(device)
    #model.apply(init_param)
    optimizer = Adan(model.parameters(), lr=lr, weight_decay=weight_decay, no_prox=no_prox)
    return model, optimizer

class FedModel:
    """
        Federated model.
    """
    def __init__(self, drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, segment_mask, self_distillation, 
                 weight_decay=None, no_prox=None, lr=None, model_rate=1, device=None):
        
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
                y_pred = self.model(train_x)
                train_y = train_y.to(torch.int64) 
                loss = criterion(y_pred, train_y)
                running_loss += loss.item()
                _, pred_y = torch.max(y_pred.data, 1)
                correct += (pred_y == train_y).sum().item()
                loss.backward()
                self.optimizer.step()
            if self.args.verbose:
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


