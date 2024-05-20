from app_site_server.custom.quantvit import QuantVisionTransformer
from functools import partial
import torch
import torch.nn as nn

model = QuantVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=300, 
                                   embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), wbits=4, abits=4)
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
for name,parameter in  model.named_parameters():
    print(name,parameter.size())
print(len(model.state_dict().keys()))