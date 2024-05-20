import torch
import copy
import sys
import os
sys.path.append(os.path.dirname(__file__))
import Dense_ViT
from timm.optim.adan import Adan


def get_server_model(drop, img_size,num_classes,embed_dim,transformer_depth,transformer_head, mlp_dim,device,model_rate=1):
    model, optimizer = None, None
    model = Dense_ViT.ViT(drop=drop, model_rate=model_rate, image_size=img_size, patch_size=16, num_classes=num_classes, dim=embed_dim,
                      depth=transformer_depth, heads=transformer_head, 
                      mlp_dim=mlp_dim).to(device)
    #model.apply(init_param)
    return model


def get_client_model(drop, img_size,num_classes,embed_dim,transformer_depth,transformer_head, mlp_dim,device,lr,weight_decay,no_prox,model_rate=1):
    model, optimizer = None, None
    model = Dense_ViT.ViT(drop=drop, model_rate=model_rate, image_size=img_size, patch_size=16, num_classes=num_classes, dim=embed_dim,
                      depth=transformer_depth, heads=transformer_head, 
                      mlp_dim=mlp_dim).to(device)
    #model.apply(init_param)
    optimizer = Adan(model.parameters(), lr=lr, weight_decay=weight_decay, no_prox=no_prox)
    return model, optimizer

class FedModel:
    """
        Federated model.
    """
    def __init__(self, drop, num_classes , embed_dim , transformer_depth , transformer_head, mlp_dim ,img_size, weight_decay=None, no_prox=None, lr=None, model_rate=1, device=None):
        self.device = device
        self.model_rate = model_rate
        if lr is None:
            self.model = get_server_model(drop=drop, img_size=img_size, num_classes=num_classes,
                                                         embed_dim=embed_dim, transformer_depth=transformer_depth, transformer_head=transformer_head,
                                                         mlp_dim=mlp_dim, device=self.device, model_rate=self.model_rate)
        else:
            self.model, self.optimizer = get_client_model(drop=drop, img_size=img_size, num_classes=num_classes,
                                                         embed_dim=embed_dim, transformer_depth=transformer_depth, transformer_head=transformer_head,
                                                         mlp_dim=mlp_dim, device=self.device, lr=lr, weight_decay=weight_decay, no_prox=no_prox, model_rate=self.model_rate)

    
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
        return self.weights_perform_local_epochs(data_loader, local_sub_epoch)


    '''def compute_delta(self, data_loader):
        """
            In case of SignSGD or Fedavg, compute the gradient of the local model.
        """
        delta = dict()
        xt = dict()
        for k, v in self.model.named_parameters():
            delta[k] = torch.zeros_like(v)
            xt[k] = copy.deepcopy(v)

        # Update local model
        loss = self.mask_perform_local_epochs(data_loader)
        for k, v in self.model.named_parameters():
            delta[k] = v - xt[k]

        # Error compensation
        if self.params.get('model').get('optimizer').get('type') == 'ef_sign_sgd':
            with torch.no_grad():
                for k, v in self.model.named_parameters():
                    delta[k] += self.beta_error * self.e[k]
                    self.e[k] = delta[k] - torch.sign(delta[k])

        return loss, delta'''

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


