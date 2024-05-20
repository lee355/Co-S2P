#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch import nn
from einops import rearrange
import torchvision
from torch_geometric.nn import GCNConv
#from torch_scatter import  scatter_max
import math

import torch
import torch.nn as nn
import torch.nn.init as init


class Residual(nn.Module):
    def __init__(self, fn, linear2_mask=None):
        super().__init__()
        self.fn = fn
        self.linear2_mask = linear2_mask

    def forward(self, x, **kwargs):
        if self.linear2_mask is not None:
            y = self.fn(x, **kwargs)
            #print(y)
            extended_tensor = torch.zeros((y.size(0), y.size(1), self.linear2_mask.size(0)), dtype=y.dtype).to(y.device)
            extended_tensor[:,:,self.linear2_mask.nonzero().squeeze()] = y
            #print(extended_tensor)
            return extended_tensor + x
        else:
            return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, linear1_size, drop, linear2_size):
        super().__init__()
        self.linear1 = nn.Linear(dim, linear1_size)
        init.kaiming_normal_(self.linear1.weight)
        init.kaiming_normal_(self.linear1.bias.unsqueeze(0))
        self.drop = nn.Dropout(0.5) if drop == True else nn.Identity()
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(linear1_size, linear2_size)
        init.kaiming_normal_(self.linear2.weight)
        init.kaiming_normal_(self.linear2.bias.unsqueeze(0))

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, drop, heads, qkv_szie, out_size):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        '''self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)'''

        self.to_qkv = nn.Linear(dim, qkv_szie, bias=False)
        init.kaiming_normal_(self.to_qkv.weight)
        self.to_out = nn.Linear(qkv_szie // 3, out_size)
        init.kaiming_normal_(self.to_out.weight)
        init.kaiming_normal_(self.to_out.bias.unsqueeze(0))
        self.dropout = nn.Dropout(p=0.2) if drop == True else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        '''q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)'''
        
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        return out


class exist_classifier(nn.Module):
    def __init__(self, dim, mlp_dim, num_classes):
        super().__init__()
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
    def __init__(self, *, drop, image_size, patch_size, num_classes, dim, depth, full_depth, heads, mlp_dim, segment_mask=None, self_distillation=True, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.full_depth = full_depth
        self.self_distillation = self_distillation
        self.exist_classifiers_depth = []
        
        self.segment_mask = segment_mask
        if segment_mask is not None:
            self.segment_output_size = []
            for value in segment_mask:
                self.segment_output_size.append(torch.sum(value).long().item())
            #print(self.segment_output_size)
        #generate exist classifiers
        for i in [0.25,0.5,0.75,1]:
            tmp_depth = math.floor(self.full_depth*i)
            if tmp_depth <= depth:
                self.exist_classifiers_depth.append(tmp_depth)
            else:
                break
        self.patch_size = patch_size
        
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        init.kaiming_normal_(self.cls_token)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim))
        init.kaiming_normal_(self.pos_embedding)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        init.kaiming_normal_(self.patch_to_embedding.weight)
        init.kaiming_normal_(self.patch_to_embedding.bias.unsqueeze(0))
        self.trasnformer_block = nn.ModuleList([])
        
        cnt = 0
        self.exist_classifiers_input_size = []
        if self.segment_mask is not None:
            for i in range(depth):
                if i == 0:
                    self.trasnformer_block.append(nn.ModuleList([
                        Residual(PreNorm(dim, Attention(dim, drop, heads[i], self.segment_output_size[cnt], 
                                                        self.segment_output_size[cnt+1])), self.segment_mask[cnt+1]),
                        Residual(PreNorm(dim, FeedForward(dim, self.segment_output_size[cnt+2], 
                                                          drop, self.segment_output_size[cnt+3])), self.segment_mask[cnt+3])
                    ]))
                else:
                    self.trasnformer_block.append(nn.ModuleList([
                        Residual(PreNorm(dim, Attention(dim, drop, heads[i], self.segment_output_size[cnt], 
                                                        self.segment_output_size[cnt+1])), self.segment_mask[cnt+1]),
                        Residual(PreNorm(dim, FeedForward(dim, self.segment_output_size[cnt+2], 
                                                          drop, self.segment_output_size[cnt+3])), self.segment_mask[cnt+3])
                    ]))
                if i+1 in self.exist_classifiers_depth:
                    self.exist_classifiers_input_size.append(self.segment_output_size[cnt+3])
                cnt += 4
            
            self.to_cls_token = nn.Identity()
            
            if self.self_distillation:
                if len(self.exist_classifiers_depth) == 4:
                    self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers3 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers4 = exist_classifier(dim, mlp_dim, num_classes)
                elif len(self.exist_classifiers_depth) == 3:
                    self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers3 = exist_classifier(dim, mlp_dim, num_classes)
                elif len(self.exist_classifiers_depth) == 2:
                    self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes)
                elif len(self.exist_classifiers_depth) == 1:
                    self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes)
            else:
                self.exist_classifiers = exist_classifier(dim, mlp_dim, num_classes)
        else:
            for i in range(depth):
                self.trasnformer_block.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, drop, heads, dim*3, dim))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, drop, dim)))
                ]))
            
            self.to_cls_token = nn.Identity()
            
            if self.self_distillation:
                if len(self.exist_classifiers_depth) == 4:
                    self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers3 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers4 = exist_classifier(dim, mlp_dim, num_classes)
                elif len(self.exist_classifiers_depth) == 3:
                    self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers3 = exist_classifier(dim, mlp_dim, num_classes)
                elif len(self.exist_classifiers_depth) == 2:
                    self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes)
                    self.exist_classifiers2 = exist_classifier(dim, mlp_dim, num_classes)
                elif len(self.exist_classifiers_depth) == 1:
                    self.exist_classifiers1 = exist_classifier(dim, mlp_dim, num_classes)
            else:
                self.exist_classifiers = exist_classifier(dim, mlp_dim, num_classes)
        
    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x_outputs = []
        cnt=0
        for i, (attn, ff) in enumerate(self.trasnformer_block):
            x = attn(x)
            x = ff(x)
            if (i+1 == self.exist_classifiers_depth[cnt]):
                y = eval("self.exist_classifiers"+str(cnt+1))(x)
                y = self.to_cls_token(y[:, 0])
                x_outputs.append(y)
                cnt += 1
        if not self.self_distillation:
            x = self.exist_classifiers(x)
            x = self.to_cls_token(x[:, 0])
            x_outputs.append(x)
        return x_outputs