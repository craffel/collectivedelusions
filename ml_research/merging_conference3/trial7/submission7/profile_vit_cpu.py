import time
import torch
import torch.nn as nn
import torch.optim as optim
import timm

print("Profiling single training step of ViT-Tiny on CPU...")
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
for p in model.parameters():
    p.requires_grad = False

# LoRA blocks
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8):
        super().__init__()
        self.original_linear = original_linear
        out_features, in_features = original_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(4, r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(4, out_features, r))
        self.active_k = 0
    def forward(self, x):
        out = self.original_linear(x)
        lora_A_k = self.lora_A[self.active_k]
        lora_B_k = self.lora_B[self.active_k]
        lora_out = torch.matmul(torch.matmul(x, lora_A_k.t()), lora_B_k.t())
        return out + lora_out

for l in range(2, 12):
    model.blocks[l].attn.qkv = LoRALinear(model.blocks[l].attn.qkv)

lora_params = []
for l in range(2, 12):
    lora_params.append(model.blocks[l].attn.qkv.lora_A)
    lora_params.append(model.blocks[l].attn.qkv.lora_B)

optimizer = optim.AdamW(lora_params, lr=2e-3)
criterion = nn.CrossEntropyLoss()

imgs = torch.randn(16, 3, 224, 224)
t0 = time.time()
feats = model.forward_features(imgs)
pooled = model.forward_head(feats, pre_logits=True)
loss = pooled.mean()
loss.backward()
optimizer.step()
print(f"One step took {time.time() - t0:.4f} seconds!")
