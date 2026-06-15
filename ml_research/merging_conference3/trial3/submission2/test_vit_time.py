import torch
import torch.nn as nn
import torchvision.models as models
import time

print("Testing pre-trained ViT-B/16 CPU execution speed...")
start = time.time()
model = models.vit_b_16(weights='DEFAULT')
print(f"Loaded model in {time.time() - start:.2f} seconds.")

# Replace head to output 10 classes
model.heads.head = nn.Linear(768, 10)

# Dummy input representing batch_size=16, 3 channels, 224x224
x = torch.randn(16, 3, 224, 224)
start_fwd = time.time()
y = model(x)
print(f"Forward pass of batch size 16 took {time.time() - start_fwd:.2f} seconds.")

# Test backward pass on head + last block
# Freezing all but head and encoder block 11
for name, param in model.named_parameters():
    if "heads" in name or "encoder.layers.encoder_layer_11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
target = torch.randint(0, 10, (16,))

start_bwd = time.time()
out = model(x)
loss = loss_fn(out, target)
loss.backward()
opt.step()
print(f"Backward/step on head and last encoder block took {time.time() - start_bwd:.2f} seconds.")
