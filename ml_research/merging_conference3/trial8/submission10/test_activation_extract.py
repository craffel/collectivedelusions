import torch
import timm

model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
model.eval()

# Fake input batch of size 2, 3 channels, 224x224
dummy_input = torch.randn(2, 3, 224, 224)

# Forward through embedding and blocks up to block_idx = 3 (layer 3)
x = model.patch_embed(dummy_input)
x = model.pos_drop(x)
if hasattr(model, 'patch_drop'):
    x = model.patch_drop(x)
x = model.norm_pre(x)

for i in range(3):
    x = model.blocks[i](x)

cls_token = x[:, 0]
print("Activation shape:", cls_token.shape)
assert cls_token.shape == (2, 192), f"Expected shape (2, 192), got {cls_token.shape}"
print("Successfully extracted 192-dimensional activation from real ViT-Tiny!")
