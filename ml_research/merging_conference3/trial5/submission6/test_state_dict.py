import timm

model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
for name, param in model.state_dict().items():
    if "blocks.9" in name:
        print(name)
