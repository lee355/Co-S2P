# -*- coding: utf-8 -*-
# Python version: 3.8
import sys
import os
sys.path.append("..")
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from einops import rearrange
import torchvision
#from torch_scatter import  scatter_max
import numpy as np
from bern import Bern
import math
import timm
import torch.nn.init as init

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, training_mode="mask", bias=True, **kwargs):
        super(MaskedLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.is_bias = bias
        self.training_mode = training_mode
        arr_weights = None
        
        if training_mode == "mask":
            self.c = np.e * np.sqrt(1 / in_features)
            arr_weights = np.random.choice([-self.c, self.c], size=(out_features, in_features))
            self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=False, dtype=torch.float))
            init.kaiming_normal_(self.weight)
            self.weight.requires_grad = False    
            
            # Weights of Mask
            arr_masks = np.random.choice([-self.c, self.c], size=self.out_features)
            self.weight_mask = nn.Parameter(torch.tensor(arr_masks, requires_grad=True, dtype=torch.float))
            init.kaiming_normal_(self.weight_mask.unsqueeze(0))
            
            #self.weight_mask = nn.Parameter(torch.randn_like(self.weight, requires_grad=True))

            if self.is_bias == True:
                arr_bias = np.random.choice([-self.c, self.c], size=out_features)
                self.bias = nn.Parameter(torch.tensor(arr_bias, requires_grad=False, dtype=torch.float))
                init.kaiming_normal_(self.bias.unsqueeze(0))
                self.bias.requires_grad = False
                #self.bias_mask = nn.Parameter(torch.randn_like(self.bias, requires_grad=True))
        elif training_mode == "both":   
            self.c = np.e * np.sqrt(1 / in_features)
            arr_weights = np.random.choice([-self.c, self.c], size=(out_features, in_features))
            self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=True, dtype=torch.float))
            self.weight.requires_grad = True    
            # Weights of Mask
            arr_masks = np.random.choice([-self.c, self.c], size=self.out_features)
            self.weight_mask = nn.Parameter(torch.tensor(arr_masks, requires_grad=True, dtype=torch.float))
            #self.weight_mask = nn.Parameter(torch.randn_like(self.weight, requires_grad=True))

            if self.is_bias == True:
                arr_bias = np.random.choice([-self.c, self.c], size=out_features)
                self.bias = nn.Parameter(torch.tensor(arr_bias, requires_grad=True, dtype=torch.float))
                self.bias.requires_grad = True
                #self.bias_mask = nn.Parameter(torch.randn_like(self.bias, requires_grad=True))
        else:
            print("MaskedLinear init training mode error!!!")
            sys.exit(0)

    def forward(self, x, ths=None):
        if self.is_bias == True:
            if ths is None:
                # Generate probability of bernouli distributions
                #print(self.weight_mask)
                s_m = torch.sigmoid(self.weight_mask)
                bias_g_m = Bern.apply(s_m)
                g_m = bias_g_m.reshape((-1,1))
            else:
                nd_w_mask = torch.sigmoid(self.weight_mask)
                g_m = torch.where(nd_w_mask > ths, 1, 0).reshape((-1,1))   
            # Compute segment-wise product with mask
            effective_weight = g_m * self.weight
            effective_weight = effective_weight.to(x.device)
            effective_bias = bias_g_m * self.bias
            effective_bias = effective_bias.to(x.device)
            lin = F.linear(x, effective_weight, effective_bias)
        else:
            if ths is None:
                # Generate probability of bernouli distributions
                s_m = torch.sigmoid(self.weight_mask)
                g_m = Bern.apply(s_m).reshape((-1,1))
            else:
                nd_mask = torch.sigmoid(self.weight_mask)
                g_m = torch.where(nd_mask > ths, 1, 0).reshape((-1,1))
            effective_weight = g_m * self.weight
            lin = F.linear(x, effective_weight)

        # Apply the effective weight on the input data
        return lin

    def __str__(self):
        prod = torch.prod(*self.weight.shape).item()
        return 'Mask Layer: \n FC Weights: {}, {}, MASK: {}'.format(self.weight.sum(), torch.abs(self.weight).sum(),
                                                                    self.mask.sum() / prod)

class MaskedQKV(nn.Linear):
    def __init__(self, in_features, out_features, heads, bias=False, **kwargs):
        super(MaskedQKV, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.is_bias = bias
        self.heads = heads
        arr_weights = None
        
        self.c = np.e * np.sqrt(1 / in_features)
        arr_weights = np.random.choice([-self.c, self.c], size=(out_features, in_features))
        self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=False, dtype=torch.float))
        init.kaiming_normal_(self.weight)
        self.weight.requires_grad = False
            
        # Weights of Mask
        arr_masks = np.random.choice([-self.c, self.c], size=self.heads)
        self.weight_mask = nn.Parameter(torch.tensor(arr_masks, requires_grad=True, dtype=torch.float))
        init.kaiming_normal_(self.weight_mask.unsqueeze(0))
        #self.weight_mask = nn.Parameter(torch.randn_like(self.weight, requires_grad=True))

        '''if self.is_bias == True:
            arr_bias = np.random.choice([-self.c, self.c], size=out_features)
            self.bias = nn.Parameter(torch.tensor(arr_bias, requires_grad=False, dtype=torch.float))
            self.bias.requires_grad = False
            #self.bias_mask = nn.Parameter(torch.randn_like(self.bias, requires_grad=True))'''

    def forward(self, x, ths=None):
        if self.is_bias == True:
            if ths is None:
                # Generate probability of bernouli distributions
                #print(self.weight_mask)
                while True:
                    s_m = torch.sigmoid(self.weight_mask)
                    bias_g_m = Bern.apply(s_m)
                    if torch.sum(bias_g_m, dim=0) != 0:
                        break
                g_m = bias_g_m.reshape((-1,1))
            else:
                nd_w_mask = torch.sigmoid(self.weight_mask)
                g_m = torch.where(nd_w_mask > ths, 1, 0).reshape((-1,1))   
            # Compute segment-wise product with mask
            effective_weight = g_m * self.weight
            effective_weight = effective_weight.to(x.device)
            effective_bias = bias_g_m * self.bias
            effective_bias = effective_bias.to(x.device)
            lin = F.linear(x, effective_weight, effective_bias)
        else:
            if ths is None:
                while True:
                    s_m = torch.sigmoid(self.weight_mask)
                    g_m = Bern.apply(s_m)
                    if torch.sum(g_m, dim=0) != 0:
                        break
                g_m = g_m.unsqueeze(1).repeat(1, int(x.size(-1)/self.heads)).view(-1).repeat(3).reshape((-1,1))
            else:
                nd_mask = torch.sigmoid(self.weight_mask)
                g_m = torch.where(nd_mask > ths, 1, 0).reshape((-1,1))
            effective_weight = g_m * self.weight
            lin = F.linear(x, effective_weight)

        return lin

    def __str__(self):
        prod = torch.prod(*self.weight.shape).item()
        return 'Mask Layer: \n FC Weights: {}, {}, MASK: {}'.format(self.weight.sum(), torch.abs(self.weight).sum(),
                                                                    self.mask.sum() / prod)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        '''init.kaiming_normal_(self.norm.weight)
        init.kaiming_normal_(self.norm.bias)'''
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, training_mode, drop):
        super().__init__()
        '''
        self.net = nn.Sequential(
            MaskedLinear(dim, hidden_dim),
            nn.Dropout(0.5),
            nn.GELU(),
            MaskedLinear(hidden_dim, dim)
        )
        '''
        self.linear1 = MaskedLinear(dim, hidden_dim,  training_mode)
        self.drop = nn.Dropout(0.5) if drop == True else nn.Identity()
        self.gelu = nn.GELU()
        self.linear2 = MaskedLinear(hidden_dim, dim, training_mode)

    def forward(self, x, ths=None):
        x = self.linear1(x, ths=ths)
        x = self.drop(x)
        x = self.gelu(x)
        x = self.linear2(x, ths=ths)
        return x
        #return self.net(x, ths)dddd


class Attention(nn.Module):
    def __init__(self, dim, training_mode, drop, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        '''self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)'''

        self.to_qkv = MaskedQKV(dim, dim * 3, heads=self.heads, bias=False)  
        self.to_out = MaskedLinear(dim, dim, training_mode=training_mode)   
        self.dropout = nn.Dropout(p=0.2) if drop == True else nn.Identity()

    def forward(self, x, ths=None, transformer_mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        
        '''q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)'''

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale


        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out, ths=ths) 
        out = self.dropout(out)
        return out

'''

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, training_mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.training_mode = training_mode
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, self.training_mode, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, self.training_mode)))
            ]))

    def forward(self, x, ths=None, transformer_mask=None):
        for attn, ff in self.layers:
            x = attn(x, ths=ths, transformer_mask=transformer_mask)
            x = ff(x, ths=ths)
        return x
'''

class exist_classifier(nn.Module):
    def __init__(self, dim, mlp_dim, num_classes, training_mode):
        super().__init__()
        self.training_mode = training_mode
        #self.exist_linear1 = MaskedLinear(dim, mlp_dim)
        self.existnorm = nn.LayerNorm(dim)
        #self.drop = nn.Dropout(0.5)
        #self.gelu = nn.GELU()
        self.exist_linear = nn.Linear(dim, num_classes)  
        init.kaiming_normal_(self.exist_linear.weight)
        init.kaiming_normal_(self.exist_linear.bias.unsqueeze(0))
    def forward(self, x, ths=None):
        x = self.existnorm(x)
        #x = self.drop(x)
        #x = self.gelu(x)
        x = self.exist_linear(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, training_mode, drop, image_size, patch_size, num_classes, dim, depth, full_depth, heads, mlp_dim, self_distillation=True, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        self.training_mode = training_mode
        self.full_depth = full_depth
        self.self_distillation = self_distillation
        self.exist_classifiers_depth = []
        #generate exist classifiers
        for i in [0.25,0.5,0.75,1]:
            tmp_depth = math.floor(self.full_depth*i)
            if tmp_depth <= depth:
                self.exist_classifiers_depth.append(tmp_depth)
            else:
                break
        #print(self.exist_classifiers_depth)
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        init.kaiming_normal_(self.cls_token)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        init.kaiming_normal_(self.pos_embedding)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        init.kaiming_normal_(self.patch_to_embedding.weight)
        init.kaiming_normal_(self.patch_to_embedding.bias.unsqueeze(0))
        #self.transformer = Transformer(dim, depth, heads, mlp_dim, self.training_mode)

        self.trasnformer_block = nn.ModuleList([])
        for _ in range(depth):
            self.trasnformer_block.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, self.training_mode, drop, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, self.training_mode, drop)))
            ]))

        self.to_cls_token = nn.Identity()

        if len(self.exist_classifiers_depth) == 4:
            self.exist_classifiers4 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
        elif len(self.exist_classifiers_depth) == 3:
            self.exist_classifiers3 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
        elif len(self.exist_classifiers_depth) == 2:

            self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
        elif len(self.exist_classifiers_depth) == 1:
            self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)

    def forward(self, img, ths=None, transformer_mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        #x = self.transformer(x, ths, transformer_mask)
        
        for i, (attn, ff) in enumerate(self.trasnformer_block):
            x = attn(x, ths=ths, transformer_mask=transformer_mask)
            x = ff(x, ths=ths)

        y = eval("self.exist_classifiers"+str(len(self.exist_classifiers_depth)))(x)
        y = self.to_cls_token(y[:, 0])
        return y

class ViT_dist(nn.Module):
    def __init__(self, *, training_mode, drop, image_size, patch_size, num_classes, dim, depth, full_depth, heads, mlp_dim, self_distillation=True, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        self.training_mode = training_mode
        self.full_depth = full_depth
        self.self_distillation = self_distillation
        self.exist_classifiers_depth = []
        #generate exist classifiers
        for i in [0.25,0.5,0.75,1]:
            tmp_depth = math.floor(self.full_depth*i)
            if tmp_depth <= depth:
                self.exist_classifiers_depth.append(tmp_depth)
            else:
                break
        #print(self.exist_classifiers_depth)
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        #self.transformer = Transformer(dim, depth, heads, mlp_dim, self.training_mode)

        self.trasnformer_block = nn.ModuleList([])
        for _ in range(depth):
            self.trasnformer_block.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, self.training_mode, drop, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, self.training_mode, drop)))
            ]))

        self.to_cls_token = nn.Identity()

        #self.exist_classifiers = [exist_classifier(dim, mlp_dim, num_classes, self.training_mode) for _ in range(len(self.exist_classifiers_depth))]
        #self.exist_classifiers = nn.ModuleList([])
        '''for _ in range(len(self.exist_classifiers_depth)):
            self.exist_classifiers.append(nn.ModuleDict({'exist_linear1': MaskedLinear(dim, mlp_dim, self.training_mode),
                                                         'drop': nn.Dropout(0.5),
                                                         'gelu': nn.GELU(),
                                                         'exist_linear2': nn.Linear(mlp_dim, num_classes)}))
        self.exist_classifiers = nn.ModuleList([(exist_classifier(dim, mlp_dim, num_classes, self.training_mode)) 
                                                for _ in range(len(self.exist_classifiers_depth))])'''
        if self.self_distillation:
            if len(self.exist_classifiers_depth) == 4:
                self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
                self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
                self.exist_classifiers3 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
                self.exist_classifiers4 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
            elif len(self.exist_classifiers_depth) == 3:
                self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
                self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
                self.exist_classifiers3 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
            elif len(self.exist_classifiers_depth) == 2:
                self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
                self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
            elif len(self.exist_classifiers_depth) == 1:
                self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)
        else:
            self.exist_classifiers = exist_classifier(dim, mlp_dim, num_classes, self.training_mode)

    def forward(self, img, ths=None, transformer_mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        #x = self.transformer(x, ths, transformer_mask)
        
        x_outputs = []
        cnt=0
        for i, (attn, ff) in enumerate(self.trasnformer_block):
            x = attn(x, ths=ths, transformer_mask=transformer_mask)
            x = ff(x, ths=ths)
            if (i+1 == self.exist_classifiers_depth[cnt]) and self.self_distillation:
                y = eval("self.exist_classifiers"+str(cnt+1))(x)
                y = self.to_cls_token(y[:, 0])
                x_outputs.append(y)
                cnt += 1
        if not self.self_distillation:
            x = self.exist_classifiers(x)
            x = self.to_cls_token(x[:, 0])
            x_outputs.append(x)
        return x_outputs