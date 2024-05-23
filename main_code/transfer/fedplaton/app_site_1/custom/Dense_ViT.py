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

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.drop = nn.Dropout(0.5) if drop == True else nn.Identity()
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

# 
"""class Attention(nn.Module):
    def __init__(self, dim, drop, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=0.2) if drop == True else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        return out

"""
class exist_classifier(nn.Module):
    def __init__(self, dim, mlp_dim, num_classes):
        super().__init__()
        #self.exist_linear1 = MaskedLinear(dim, mlp_dim)
        self.existnorm = nn.LayerNorm(dim)
        #self.drop = nn.Dropout(0.5)
        #self.gelu = nn.GELU()
        self.exist_linear = nn.Linear(dim, num_classes)  
    def forward(self, x, ths=None):
        x = self.existnorm(x)
        #x = self.drop(x)
        #x = self.gelu(x)
        x = self.exist_linear(x)
        return x

class Attention(nn.Module):
    def __init__(self, embed_size,drop,num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.linear_q = nn.Linear(embed_size, embed_size)
        self.linear_k = nn.Linear(embed_size, embed_size)
        self.linear_v = nn.Linear(embed_size, embed_size)

        self.linear_o = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p=0.2) if drop == True else nn.Identity()

    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        # Linear transformation for query, key, and value
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)

        # Split the embeddings into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.head_dim ** 0.5

        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(energy, dim=-1)

        # Calculate output for each head
        x = torch.matmul(attention, V)

        # Concatenate heads and perform final linear transformation
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        x = self.linear_o(x)
        x = self.dropout(x)

        return x

# 
class ViT(nn.Module):
    def __init__(self, *, drop, model_rate, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        #self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.patch_to_embedding = PatchEmbed(img_size=image_size, patch_size=patch_size, in_c=channels, embed_dim=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.trasnformer_block = nn.ModuleList([])
        for _ in range(depth):
            self.trasnformer_block.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, drop, num_heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, drop)))
            ]))

        self.to_cls_token = nn.Identity()
        
        self.exist_classifiers = exist_classifier(dim, mlp_dim, num_classes)
        
    def forward(self, img, mask=None):
        p = self.patch_size

        #x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(img)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        for i, (attn, ff) in enumerate(self.trasnformer_block):
            #x = attn(x, x, x, transformer_mask=transformer_mask)
            x = attn(x, mask=None)
            x = ff(x)
            
        x = self.exist_classifiers(x)
        x = self.to_cls_token(x[:, 0])
        return x
    