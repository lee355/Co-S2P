from app_site_server.custom.quantvit import QuantVisionTransformer

model = QuantVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=300, 
                                   embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), wbits=4, abits=4).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)