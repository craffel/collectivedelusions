import timm

model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
print("Block 9 attention module:")
print(model.blocks[9].attn)
print("\nBlock 9 MLP module:")
print(model.blocks[9].mlp)
